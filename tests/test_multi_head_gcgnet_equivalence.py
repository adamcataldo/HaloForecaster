from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from halo import (
    gaussian_head,
    gcgnet_adapter,
    gcgnet_configs,
    multi_head_gcgnet,
    multi_head_gcgnet_configs,
    registry,
    seeding,
    settings,
)
from halo.multi_head_gcgnet import MultiHeadGCGNet

SETTING = settings.SHORT_TERM_EXOGENOUS
SEED = 20260827
BATCH_SIZE = 3
MARKET = "NP"

VENDORED_HEAD = "head."
OUR_HEAD = "heads.0."
VENDORED_PREFIX = "inner."

SMALL: dict[str, Any] = {"d_model": 64, "d_ff": 32, "dropout": 0.1}


def config(**overrides: Any) -> dict[str, Any]:
    inherited = next(
        config
        for config in gcgnet_configs.exogenous_configs()
        if config["dataset_name"] == MARKET
    )
    return {**inherited, **SMALL, **overrides}


def vendored_and_ours(
    config: dict[str, Any],
) -> tuple[gcgnet_adapter.GCGNetAdapter, MultiHeadGCGNet]:
    seeding.seed_everything(SEED)
    vendored = cast(gcgnet_adapter.GCGNetAdapter, gcgnet_configs.build(config)).eval()
    seeding.seed_everything(SEED)
    ours = cast(MultiHeadGCGNet, multi_head_gcgnet.builder()(config)).eval()
    return vendored, ours


def under_the_vendored_name(key: str) -> str:
    named = key.replace(OUR_HEAD, VENDORED_HEAD, 1) if key.startswith(OUR_HEAD) else key
    return VENDORED_PREFIX + named


def varying_across_batch_and_time_and_channel(
    config: dict[str, Any], length: int, offset: int
) -> torch.Tensor:
    window = torch.randn(
        BATCH_SIZE,
        length,
        config["enc_in"],
        generator=seeding.torch_generator(SEED + offset),
    )
    spread = (
        torch.arange(float(BATCH_SIZE)).reshape(-1, 1, 1)
        + torch.arange(float(length)).reshape(1, -1, 1) / length
        + torch.arange(float(config["enc_in"])).reshape(1, 1, -1)
    )
    return window + spread


def history(config: dict[str, Any]) -> torch.Tensor:
    return varying_across_batch_and_time_and_channel(config, config["seq_len"], 0)


def future(config: dict[str, Any]) -> torch.Tensor:
    return varying_across_batch_and_time_and_channel(config, config["pred_len"], 1)


def elsewhere(config: dict[str, Any]) -> torch.Tensor:
    return future(config) * 100 + 50


@pytest.fixture(params=[True, False], ids=["standardized", "unstandardized"])
def use_norm(request) -> bool:
    return request.param


def test_the_probe_input_varies_across_batch_and_time_and_channel():
    window = history(config())

    assert (window.std(dim=0) > 0).all()
    assert (window.std(dim=1) > 0).all()
    assert (window.std(dim=2) > 0).all()


def test_the_probe_stays_inside_the_range_the_unstandardized_pipeline_carries():
    probe = config(use_norm=False)
    vendored, _ = vendored_and_ours(probe)

    with torch.no_grad():
        seeding.seed_everything(SEED)
        theirs, auxiliary = vendored.forecast_with_auxiliary(
            history(probe), future(probe)
        )

    assert torch.isfinite(theirs).all()
    assert torch.isfinite(auxiliary)


def test_the_probe_raises_gcgnets_dropout_so_evaluation_mode_is_measurable():
    assert config()["dropout"] > 0
    assert gcgnet_configs.EXOGENOUS_CONSTANTS["dropout"] == 0


def test_the_stored_tensors_correspond_one_for_one_in_both_directions(use_norm):
    vendored, ours = vendored_and_ours(config(use_norm=use_norm))

    theirs = vendored.state_dict()
    mine = {under_the_vendored_name(key): t for key, t in ours.state_dict().items()}

    assert len(mine) == len(ours.state_dict())
    assert set(mine) == set(theirs)
    assert len(mine) == len(theirs)


def test_corresponding_tensors_are_already_equal_without_anything_being_copied(
    use_norm,
):
    vendored, ours = vendored_and_ours(config(use_norm=use_norm))

    theirs = vendored.state_dict()
    mine = {under_the_vendored_name(key): t for key, t in ours.state_dict().items()}

    for name, tensor in mine.items():
        assert torch.equal(tensor, theirs[name]), name


def test_one_head_and_both_callables_defaulted_reproduces_gcgnets_forecast(use_norm):
    probe = config(use_norm=use_norm)
    vendored, ours = vendored_and_ours(probe)
    window, ahead = history(probe), future(probe)

    with torch.no_grad():
        seeding.seed_everything(SEED)
        theirs, _ = vendored.forecast_with_auxiliary(window, ahead)
        seeding.seed_everything(SEED)
        mine, _ = ours.predict_with_auxiliary(window, ahead)

    assert mine.point_estimate.shape == theirs.shape
    assert torch.allclose(mine.point_estimate, theirs, rtol=1e-5, atol=1e-6)


def test_one_head_and_both_callables_defaulted_reproduces_gcgnets_auxiliary(use_norm):
    probe = config(use_norm=use_norm)
    vendored, ours = vendored_and_ours(probe)
    window, ahead = history(probe), future(probe)

    with torch.no_grad():
        seeding.seed_everything(SEED)
        _, theirs = vendored.forecast_with_auxiliary(window, ahead)
        seeding.seed_everything(SEED)
        _, mine = ours.predict_with_auxiliary(window, ahead)

    assert torch.isfinite(mine)
    assert torch.allclose(mine, theirs, rtol=1e-5, atol=1e-6)


def test_the_plain_forward_pass_agrees_with_an_absent_future(use_norm):
    probe = config(use_norm=use_norm)
    _, ours = vendored_and_ours(probe)
    window = history(probe)
    absent = torch.zeros(BATCH_SIZE, probe["pred_len"], probe["enc_in"])

    with torch.no_grad():
        seeding.seed_everything(SEED)
        plain = ours(window, None, None, None)
        seeding.seed_everything(SEED)
        forecast, _ = ours.predict_with_auxiliary(window, absent)

    assert torch.equal(plain, forecast.point_estimate)


def published_forecast(model: MultiHeadGCGNet, window, ahead) -> torch.Tensor:
    with torch.no_grad():
        seeding.seed_everything(SEED)
        forecast, _ = model.predict_with_auxiliary(window, ahead)
    return forecast.point_estimate


def test_an_evaluation_forecast_does_not_move_when_the_future_is_replaced():
    probe = config()
    seeding.seed_everything(SEED)
    model = cast(
        MultiHeadGCGNet,
        registry.entry(SETTING.name, multi_head_gcgnet_configs.MODEL_NAME).build(probe),
    ).eval()
    window = history(probe)

    assert torch.equal(
        published_forecast(model, window, future(probe)),
        published_forecast(model, window, elsewhere(probe)),
    )


def test_the_published_head_count_and_callables_are_what_that_forecast_used():
    probe = config()
    seeding.seed_everything(SEED)
    model = registry.entry(SETTING.name, multi_head_gcgnet_configs.MODEL_NAME).build(
        probe
    )

    assert isinstance(model, MultiHeadGCGNet)
    assert len(model.heads) == multi_head_gcgnet_configs.OUTPUT_HEADS
    assert model.constraining_transformation is gaussian_head.scale_in_target_units
    assert model.point_estimate is gaussian_head.location
