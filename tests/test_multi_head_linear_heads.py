from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from torch.nn.functional import softplus

from halo import multi_head_linear, multi_head_linear_configs, seeding
from halo.errors import ConfigurationError
from halo.multi_head_linear import MultiHeadLinear

SEED = 20260827
BATCH_SIZE = 3
SEQ_LEN = 48
PRED_LEN = 8
CHANNELS = 3

SHAPE: dict[str, Any] = {
    "seq_len": SEQ_LEN,
    "pred_len": PRED_LEN,
    "patch_len": 12,
    "dec_in": CHANNELS,
    "d_model": 16,
    "d_ff": 32,
    "alpha": 1.0,
    "beta": 1.0,
}


def build(**overrides: Any) -> MultiHeadLinear:
    seeding.seed_everything(SEED)
    return MultiHeadLinear(**{**SHAPE, **overrides}).eval()


def inputs() -> tuple[Tensor, Tensor]:
    generator = seeding.torch_generator(SEED)
    noise = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, generator=generator)
    per_channel_scale = torch.tensor([1.0, 4.0, 9.0])
    per_channel_shift = torch.tensor([0.0, 10.0, -5.0])
    x_enc = noise * per_channel_scale + per_channel_shift
    x_mark_enc = torch.randn(BATCH_SIZE, SEQ_LEN, 4, generator=generator)
    return x_enc, x_mark_enc


def unbiased_statistics_of(channels: Tensor) -> tuple[Tensor, Tensor]:
    mean = channels.mean(dim=1, keepdim=True)
    stdev = channels.std(dim=1, keepdim=True, correction=1)
    return mean, stdev


def population_deviation_of(channels: Tensor) -> Tensor:
    return channels.std(dim=1, keepdim=True, correction=0)


def second_head(parameters: Tensor) -> Tensor:
    return parameters[..., 1:2]


def softplus_everywhere(raw: Tensor, mean: Tensor, stdev: Tensor) -> Tensor:
    return softplus(raw)


def the_statistics_alone(raw: Tensor, mean: Tensor, stdev: Tensor) -> Tensor:
    return (mean + stdev).expand_as(raw)


class Recording:
    def __init__(self, inner=None) -> None:
        self.inner = inner
        self.parameters: Tensor
        self.mean: Tensor
        self.stdev: Tensor

    def __call__(self, parameters: Tensor, mean: Tensor, stdev: Tensor) -> Tensor:
        self.parameters, self.mean, self.stdev = parameters, mean, stdev
        return parameters if self.inner is None else self.inner(parameters, mean, stdev)


@pytest.mark.parametrize("output_heads", [1, 2, 4])
def test_the_forecast_carries_one_parameter_per_output_head(output_heads):
    model = build(output_heads=output_heads)
    x_enc, x_mark_enc = inputs()

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)

    assert forecast.parameters.shape == (BATCH_SIZE, PRED_LEN, output_heads)
    assert forecast.point_estimate.shape == (BATCH_SIZE, PRED_LEN, 1)


@pytest.mark.parametrize("output_heads", [1, 2, 4])
def test_the_harness_sees_one_forecast_however_many_heads_there_are(output_heads):
    model = build(output_heads=output_heads)
    x_enc, x_mark_enc = inputs()

    with torch.no_grad():
        prediction = model(x_enc, x_mark_enc, x_enc, x_mark_enc)

    assert prediction.shape == (BATCH_SIZE, PRED_LEN, 1)


def test_the_constraining_transformation_reaches_every_parameter():
    x_enc, x_mark_enc = inputs()
    unconstrained = build(output_heads=3)
    constrained = build(output_heads=3, constraining_transformation=softplus_everywhere)

    with torch.no_grad():
        loose = unconstrained.predict(x_enc, x_mark_enc).parameters
        tight = constrained.predict(x_enc, x_mark_enc).parameters

    assert (loose < 0).any()
    assert (tight > 0).all()
    assert torch.allclose(tight, softplus(loose), atol=1e-6)


def test_the_supplied_reduction_rather_than_the_first_head_is_destandardized():
    x_enc, x_mark_enc = inputs()
    model = build(output_heads=2, point_estimate=second_head)
    mean, stdev = unbiased_statistics_of(x_enc[:, :, -1:])

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)

    assert torch.allclose(
        forecast.point_estimate, forecast.parameters[..., 1:2] * stdev + mean
    )
    assert not torch.allclose(
        forecast.point_estimate, forecast.parameters[..., :1] * stdev + mean
    )


def test_the_constraint_is_handed_the_target_channels_own_mean_and_standard_deviation():
    x_enc, x_mark_enc = inputs()
    recording = Recording()
    model = build(output_heads=2, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc, x_mark_enc)

    mean, stdev = unbiased_statistics_of(x_enc[:, :, -1:])

    assert recording.mean.shape == (BATCH_SIZE, 1, 1)
    assert recording.stdev.shape == (BATCH_SIZE, 1, 1)
    assert torch.allclose(recording.mean, mean)
    assert torch.allclose(recording.stdev, stdev)


def test_the_deviation_handed_over_is_the_unbiased_one_and_not_the_population_one():
    x_enc, x_mark_enc = inputs()
    recording = Recording()
    model = build(output_heads=2, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc, x_mark_enc)

    _, unbiased = unbiased_statistics_of(x_enc[:, :, -1:])
    population = population_deviation_of(x_enc[:, :, -1:])

    assert not torch.allclose(unbiased, population)
    assert torch.allclose(recording.stdev, unbiased)
    assert not torch.allclose(recording.stdev, population)


@pytest.mark.parametrize("other", [0, 1])
def test_the_statistics_handed_over_are_no_other_channels(other):
    x_enc, x_mark_enc = inputs()
    recording = Recording()
    model = build(output_heads=2, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc, x_mark_enc)

    mean, stdev = unbiased_statistics_of(x_enc[:, :, other : other + 1])

    assert not torch.allclose(recording.mean, mean)
    assert not torch.allclose(recording.stdev, stdev)


def test_the_statistics_broadcast_against_the_heads():
    x_enc, x_mark_enc = inputs()
    recording = Recording()
    model = build(output_heads=4, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc, x_mark_enc)

    scaled = recording.stdev * recording.parameters + recording.mean

    assert scaled.shape == recording.parameters.shape


def test_no_gradient_reaches_the_model_through_either_statistic():
    x_enc, x_mark_enc = inputs()
    x_enc.requires_grad_(True)
    recording = Recording()
    model = build(output_heads=1, constraining_transformation=recording)

    model.predict(x_enc, x_mark_enc)

    assert not recording.mean.requires_grad
    assert recording.mean.grad_fn is None
    assert not recording.stdev.requires_grad
    assert recording.stdev.grad_fn is None


def test_a_forecast_made_only_of_the_statistics_carries_no_gradient_at_all():
    x_enc, x_mark_enc = inputs()
    x_enc.requires_grad_(True)
    model = build(output_heads=1, constraining_transformation=the_statistics_alone)

    forecast = model.predict(x_enc, x_mark_enc)

    assert not forecast.parameters.requires_grad
    assert not forecast.point_estimate.requires_grad


def test_the_heads_still_carry_gradient_back_into_the_pipeline():
    x_enc, x_mark_enc = inputs()
    model = build(output_heads=2)

    model.predict(x_enc, x_mark_enc).point_estimate.sum().backward()

    embedding = model.value_embedding.linear[1].weight

    assert embedding.grad is not None
    assert embedding.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_heads": 0}, "at least"),
        ({"output_heads": -1}, "at least"),
        ({"dec_in": 0}, "no target channel"),
    ],
)
def test_a_configuration_that_would_fail_silently_is_refused_at_construction(
    overrides, message
):
    with pytest.raises(ConfigurationError, match=message):
        build(**overrides)


@pytest.mark.parametrize("patch_len", [7, 17, 32])
def test_a_patch_length_that_does_not_divide_the_lookback_is_padded_not_refused(
    patch_len,
):
    model = build(patch_len=patch_len, output_heads=2)
    x_enc, x_mark_enc = inputs()

    assert SEQ_LEN % patch_len != 0
    assert model.value_embedding.pad_num > 0

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)

    assert forecast.parameters.shape == (BATCH_SIZE, PRED_LEN, 2)
    assert torch.isfinite(forecast.point_estimate).all()


@pytest.mark.parametrize(
    "parameter",
    ["output_heads", "constraining_transformation", "point_estimate"],
)
def test_a_config_asking_for_a_constructor_parameter_is_refused_not_ignored(parameter):
    config = dict(multi_head_linear_configs.exogenous_search_space()[0])
    config[parameter] = 2 if parameter == "output_heads" else softplus

    with pytest.raises(ConfigurationError, match=parameter):
        multi_head_linear.builder()(config)


@pytest.mark.parametrize("features", ["M", "S"])
def test_a_config_outside_the_short_term_exogenous_setting_is_refused(features):
    config = dict(multi_head_linear_configs.exogenous_search_space()[0])
    config["features"] = features

    with pytest.raises(ConfigurationError, match="head bank has already"):
        multi_head_linear.builder()(config)


def test_the_short_term_exogenous_setting_is_what_the_builder_accepts():
    config = dict(multi_head_linear_configs.exogenous_search_space()[0])

    assert config["features"] == multi_head_linear.EXOGENOUS_FEATURES
    assert multi_head_linear.builder()(config) is not None
