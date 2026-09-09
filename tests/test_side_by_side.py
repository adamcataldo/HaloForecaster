from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from halo import (
    crosslinear_configs,
    gaussian_head,
    gcgnet_configs,
    multi_head_gcgnet,
    multi_head_linear,
    multi_head_timexer,
    seeding,
    side_by_side,
    timexer_configs,
    tslib,
)
from halo.errors import ConfigurationError

SEED = 20260831
BATCH_SIZE = 3
MARKET = "NP"

CROSSLINEAR = "CrossLinear"
TIMEXER = "TimeXer"

LOCATION = gaussian_head.LOCATION
SCALE = gaussian_head.SCALE
OUTPUT_HEADS = gaussian_head.OUTPUT_HEADS
HEADS_PER_BACKBONE = side_by_side.HEADS_PER_BACKBONE

SMALL: dict[str, dict[str, Any]] = {
    CROSSLINEAR: {"d_model": 32, "d_ff": 64},
    TIMEXER: {"d_model": 64, "d_ff": 128, "e_layers": 2, "n_heads": 4},
}


def config(family: str, **overrides: Any) -> dict[str, Any]:
    published = (
        crosslinear_configs.exogenous_configs()
        if family == CROSSLINEAR
        else timexer_configs.exogenous_configs()
    )
    inherited = next(one for one in published if one["dataset_name"] == MARKET)
    return {**inherited, **SMALL[family], **overrides}


def backbone_module(family: str):
    return multi_head_linear if family == CROSSLINEAR else multi_head_timexer


def one_head_builder(family: str) -> Callable[[dict[str, Any]], nn.Module]:
    return backbone_module(family).builder()


def shared_trunk_builder(family: str) -> Callable[[dict[str, Any]], nn.Module]:
    return backbone_module(family).builder(
        output_heads=OUTPUT_HEADS,
        constraining_transformation=gaussian_head.scale_in_target_units,
        point_estimate=gaussian_head.location,
    )


def split_builder(family: str) -> Callable[[dict[str, Any]], nn.Module]:
    return side_by_side.builder(
        one_head_builder(family),
        output_heads=OUTPUT_HEADS,
        constraining_transformation=gaussian_head.scale_in_target_units,
        point_estimate=gaussian_head.location,
    )


def vendored_builder(family: str) -> Callable[[dict[str, Any]], nn.Module]:
    if family == CROSSLINEAR:
        return crosslinear_configs.build
    return tslib.model_builder(timexer_configs.MODEL_NAME)


def seeded(build: Callable[[dict[str, Any]], nn.Module], probe: dict[str, Any]):
    seeding.seed_everything(SEED)
    return build(probe).eval()


def probe_batch(probe: dict[str, Any]) -> tuple[Tensor, ...]:
    x_enc, x_mark_enc, x_dec, x_mark_dec = tslib.example_batch(
        probe, BATCH_SIZE, seeding.torch_generator(SEED)
    )
    spread = (
        torch.arange(float(BATCH_SIZE)).reshape(-1, 1, 1)
        + torch.arange(float(probe["seq_len"])).reshape(1, -1, 1)
        + torch.arange(float(probe["enc_in"])).reshape(1, 1, -1)
    )
    return x_enc + spread, x_mark_enc, x_dec, x_mark_dec


def stored_tensors_of(module: nn.Module) -> list[Tensor]:
    return [*module.parameters(), *module.buffers()]


def total_absolute_gradient_of(module: nn.Module) -> Tensor:
    return sum(
        (
            parameter.grad.abs().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        ),
        start=torch.zeros(()),
    )


def location_path_of(control: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor
        for name, tensor in control.state_dict().items()
        if not name.startswith(f"heads.{SCALE}.")
    }


@pytest.fixture(params=[CROSSLINEAR, TIMEXER], name="family")
def _family(request) -> str:
    return request.param


def test_the_probe_input_varies_across_batch_and_time_and_channel(family):
    x_enc, _, _, _ = probe_batch(config(family))

    assert (x_enc.std(dim=0) > 0).all()
    assert (x_enc.std(dim=1) > 0).all()
    assert (x_enc.std(dim=2) > 0).all()


def test_the_split_model_carries_one_whole_backbone_per_output_head(family):
    model = seeded(split_builder(family), config(family))

    assert len(model.backbones) == OUTPUT_HEADS


def test_no_stored_tensor_is_shared_between_the_two_branches(family):
    model = seeded(split_builder(family), config(family))

    location = stored_tensors_of(model.backbones[LOCATION])
    scale = stored_tensors_of(model.backbones[SCALE])

    assert location
    assert scale
    assert {id(tensor) for tensor in location}.isdisjoint(
        {id(tensor) for tensor in scale}
    )
    assert {tensor.data_ptr() for tensor in location if tensor.numel()}.isdisjoint(
        {tensor.data_ptr() for tensor in scale if tensor.numel()}
    )


def test_no_gradient_reaches_the_scale_branch_from_the_location_column(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    x_enc, x_mark_enc, _, _ = probe_batch(probe)

    model.predict(x_enc, x_mark_enc).parameters[..., LOCATION].sum().backward()

    assert total_absolute_gradient_of(model.backbones[LOCATION]) > 0
    assert total_absolute_gradient_of(model.backbones[SCALE]) == 0


def test_no_gradient_reaches_the_location_branch_from_the_scale_column(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    x_enc, x_mark_enc, _, _ = probe_batch(probe)

    model.predict(x_enc, x_mark_enc).parameters[..., SCALE].sum().backward()

    assert total_absolute_gradient_of(model.backbones[SCALE]) > 0
    assert total_absolute_gradient_of(model.backbones[LOCATION]) == 0


def test_the_location_branchs_tensors_correspond_one_for_one_with_the_controls(family):
    probe = config(family)
    control = seeded(shared_trunk_builder(family), probe)
    split = seeded(split_builder(family), probe)

    branch = split.backbones[LOCATION].state_dict()
    inherited = location_path_of(control)

    assert set(branch) == set(inherited)
    assert len(branch) == len(inherited)
    assert len(inherited) < len(control.state_dict())


def test_the_location_branchs_tensors_are_already_equal_to_the_controls(family):
    probe = config(family)
    control = seeded(shared_trunk_builder(family), probe)
    split = seeded(split_builder(family), probe)

    inherited = location_path_of(control)

    for name, tensor in split.backbones[LOCATION].state_dict().items():
        assert torch.equal(tensor, inherited[name]), name


def test_the_two_branches_start_from_different_weights(family):
    split = seeded(split_builder(family), config(family))

    location = split.backbones[LOCATION].state_dict()
    scale = split.backbones[SCALE].state_dict()

    assert set(location) == set(scale)
    assert any(not torch.equal(location[name], scale[name]) for name in location)


def test_both_branches_read_the_same_window_statistics_off_the_same_input(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    x_enc, x_mark_enc, _, _ = probe_batch(probe)

    with torch.no_grad():
        branches = [
            backbone.raw_heads(x_enc, x_mark_enc) for backbone in model.backbones
        ]

    assert torch.equal(branches[LOCATION].window_mean, branches[SCALE].window_mean)
    assert torch.equal(branches[LOCATION].window_stdev, branches[SCALE].window_stdev)


def test_the_scale_that_reaches_the_objective_is_positive(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    x_enc, x_mark_enc, _, _ = probe_batch(probe)

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)

    assert forecast.parameters.shape == (BATCH_SIZE, probe["pred_len"], OUTPUT_HEADS)
    assert (forecast.parameters[..., SCALE:] > 0).all()


def test_the_scale_carries_the_windows_standard_deviation(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    x_enc, x_mark_enc, _, _ = probe_batch(probe)

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)
        branch = model.backbones[SCALE].raw_heads(x_enc, x_mark_enc)

    assert (branch.window_stdev > 1).any()
    assert torch.allclose(
        forecast.parameters[..., SCALE:], branch.window_stdev * softplus(branch.raw)
    )


def test_the_point_estimate_is_destandardized_exactly_once(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    x_enc, x_mark_enc, _, _ = probe_batch(probe)

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)
        branch = model.backbones[LOCATION].raw_heads(x_enc, x_mark_enc)

    assert torch.equal(forecast.parameters[..., LOCATION : LOCATION + 1], branch.raw)
    assert torch.allclose(
        forecast.point_estimate, branch.raw * branch.window_stdev + branch.window_mean
    )


def test_the_harness_sees_one_forecast_from_the_two_branches(family):
    probe = config(family)
    model = seeded(split_builder(family), probe)
    batch = probe_batch(probe)

    with torch.no_grad():
        prediction = model(*batch)

    assert prediction.shape == (BATCH_SIZE, probe["pred_len"], 1)


def test_one_backbone_and_both_callables_defaulted_reproduces_the_vendored_forecast(
    family,
):
    probe = config(family)
    vendored = seeded(vendored_builder(family), probe)
    ours = seeded(side_by_side.builder(one_head_builder(family)), probe)
    batch = probe_batch(probe)

    with torch.no_grad():
        theirs = vendored(*batch)
        mine = ours(*batch)

    assert mine.shape == theirs.shape
    assert torch.allclose(mine, theirs, rtol=1e-5, atol=1e-6)


def test_a_model_with_no_branches_is_refused_at_construction():
    with pytest.raises(ConfigurationError, match="at least one projection"):
        side_by_side.SideBySide(backbones=[])


@pytest.mark.parametrize("output_heads", [0, -1])
def test_a_builder_asked_for_no_branches_is_refused_at_construction(
    family, output_heads
):
    build = side_by_side.builder(one_head_builder(family), output_heads=output_heads)

    with pytest.raises(ConfigurationError, match="at least one projection"):
        build(config(family))


def test_a_branch_that_exposes_no_bank_of_heads_is_refused(family):
    probe = config(family)
    seeding.seed_everything(SEED)

    with pytest.raises(ConfigurationError, match="no bank of heads"):
        side_by_side.SideBySide(
            backbones=[one_head_builder(family)(probe), nn.Linear(1, 1)]
        )


def test_a_branch_that_answers_only_in_finished_forecasts_is_refused(family):
    seeding.seed_everything(SEED)
    inherited = next(
        one
        for one in gcgnet_configs.exogenous_configs()
        if one["dataset_name"] == MARKET
    )
    build = multi_head_gcgnet.builder()
    branch = cast(multi_head_gcgnet.MultiHeadGCGNet, build(inherited))

    assert len(branch.heads) == HEADS_PER_BACKBONE
    assert not hasattr(branch, "raw_heads")

    with pytest.raises(ConfigurationError, match="cannot report their raw heads"):
        side_by_side.SideBySide(backbones=[branch, build(inherited)])


def test_branches_drawn_from_two_backbones_are_refused():
    seeding.seed_everything(SEED)
    linear = one_head_builder(CROSSLINEAR)(config(CROSSLINEAR))
    timexer = one_head_builder(TIMEXER)(config(TIMEXER))

    with pytest.raises(ConfigurationError, match="not MultiHeadTimeXer"):
        side_by_side.SideBySide(backbones=[timexer, linear])


def test_branches_that_would_standardize_unalike_are_refused():
    build = one_head_builder(TIMEXER)
    seeding.seed_everything(SEED)
    standardizing = build(config(TIMEXER, use_norm=True))
    raw = build(config(TIMEXER, use_norm=False))

    with pytest.raises(ConfigurationError, match="configured unlike branch"):
        side_by_side.SideBySide(backbones=[standardizing, raw])


def doubled(parameters: Tensor, window_mean: Tensor, window_stdev: Tensor) -> Tensor:
    return parameters * 2


@pytest.mark.parametrize(
    "carried",
    [{"constraining_transformation": doubled}, {"point_estimate": softplus}],
)
def test_a_branch_carrying_an_output_stage_of_its_own_is_refused(family, carried):
    probe = config(family)
    seeding.seed_everything(SEED)
    build = backbone_module(family).builder(**carried)

    with pytest.raises(ConfigurationError, match="constraint or a point estimate"):
        side_by_side.SideBySide(backbones=[build(probe), build(probe)])


def test_a_branch_whose_output_stage_changes_nothing_is_accepted(family):
    probe = config(family)
    seeding.seed_everything(SEED)
    build = backbone_module(family).builder(
        constraining_transformation=gaussian_head.scale_in_target_units,
        point_estimate=gaussian_head.location,
    )

    model = side_by_side.SideBySide(backbones=[build(probe), build(probe)])

    assert len(model.backbones) == OUTPUT_HEADS


def test_a_branch_carrying_more_than_one_head_is_refused(family):
    probe = config(family)
    seeding.seed_everything(SEED)
    build = backbone_module(family).builder(output_heads=OUTPUT_HEADS)

    with pytest.raises(ConfigurationError, match="one whole network per column"):
        side_by_side.SideBySide(backbones=[build(probe), build(probe)])


def test_branches_whose_heads_would_not_stack_are_refused(family):
    build = one_head_builder(family)
    seeding.seed_everything(SEED)
    shorter = build(config(family, pred_len=12))
    longer = build(config(family))

    with pytest.raises(ConfigurationError, match="would not stack"):
        side_by_side.SideBySide(backbones=[shorter, longer])


@pytest.mark.parametrize(
    "parameter",
    ["output_heads", "constraining_transformation", "point_estimate"],
)
def test_a_config_asking_for_a_constructor_parameter_is_refused_not_ignored(
    family, parameter
):
    probe = config(family)
    probe[parameter] = 2 if parameter == "output_heads" else softplus

    with pytest.raises(ConfigurationError, match=parameter):
        split_builder(family)(probe)
