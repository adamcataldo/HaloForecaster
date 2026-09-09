from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from torch.nn.functional import softplus

from halo import (
    gcgnet_configs,
    multi_head_gcgnet,
    multi_head_gcgnet_configs,
    seeding,
)
from halo.errors import ConfigurationError
from halo.multi_head_gcgnet import MultiHeadGCGNet

SEED = 20260827
BATCH_SIZE = 3
SEQ_LEN = 48
PRED_LEN = 8
CHANNELS = 3

SHAPE: dict[str, Any] = {
    "seq_len": SEQ_LEN,
    "pred_len": PRED_LEN,
    "patch_len": 12,
    "enc_in": CHANNELS,
    "series_dim": 1,
    "d_model": 16,
    "d_ff": 32,
    "n_heads": 4,
    "e_layers": 1,
    "rank": 4,
    "dropout": 0.0,
}


def build(**overrides: Any) -> MultiHeadGCGNet:
    seeding.seed_everything(SEED)
    return MultiHeadGCGNet(**{**SHAPE, **overrides}).eval()


def inputs() -> tuple[Tensor, Tensor]:
    generator = seeding.torch_generator(SEED)
    noise = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, generator=generator)
    per_channel_scale = torch.tensor([1.0, 4.0, 9.0])
    per_channel_shift = torch.tensor([0.0, 10.0, -5.0])
    x_enc = noise * per_channel_scale + per_channel_shift
    future = torch.randn(BATCH_SIZE, PRED_LEN, CHANNELS, generator=generator)
    return x_enc, future


def statistics_of(channels: Tensor) -> tuple[Tensor, Tensor]:
    mean = channels.mean(dim=1, keepdim=True)
    stdev = torch.sqrt(
        channels.var(dim=1, keepdim=True, correction=0)
        + multi_head_gcgnet.STANDARDIZATION_EPSILON
    )
    return mean, stdev


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
    x_enc, _ = inputs()

    with torch.no_grad():
        forecast = model.predict(x_enc)

    assert forecast.parameters.shape == (BATCH_SIZE, PRED_LEN, output_heads)
    assert forecast.point_estimate.shape == (BATCH_SIZE, PRED_LEN, 1)


@pytest.mark.parametrize("output_heads", [1, 2, 4])
def test_the_harness_sees_one_forecast_however_many_heads_there_are(output_heads):
    model = build(output_heads=output_heads)
    x_enc, _ = inputs()

    with torch.no_grad():
        prediction = model(x_enc, None, None, None)

    assert prediction.shape == (BATCH_SIZE, PRED_LEN, 1)


@pytest.mark.parametrize("output_heads", [1, 2, 4])
def test_the_auxiliary_number_travels_beside_the_forecast_whatever_the_head_count(
    output_heads,
):
    model = build(output_heads=output_heads)
    x_enc, future = inputs()

    with torch.no_grad():
        forecast, auxiliary = model.predict_with_auxiliary(x_enc, future)

    assert forecast.parameters.shape[-1] == output_heads
    assert auxiliary.shape == ()
    assert torch.isfinite(auxiliary)


def test_the_constraining_transformation_reaches_every_parameter():
    x_enc, _ = inputs()
    unconstrained = build(output_heads=3)
    constrained = build(output_heads=3, constraining_transformation=softplus_everywhere)

    with torch.no_grad():
        seeding.seed_everything(SEED)
        loose = unconstrained.predict(x_enc).parameters
        seeding.seed_everything(SEED)
        tight = constrained.predict(x_enc).parameters

    assert (loose < 0).any()
    assert (tight > 0).all()
    assert torch.allclose(tight, softplus(loose), atol=1e-6)


def test_the_supplied_reduction_rather_than_the_first_head_is_destandardized():
    x_enc, _ = inputs()
    model = build(output_heads=2, point_estimate=second_head)
    mean, stdev = statistics_of(x_enc[:, :, -1:])

    with torch.no_grad():
        forecast = model.predict(x_enc)

    assert torch.allclose(
        forecast.point_estimate, forecast.parameters[..., 1:2] * stdev + mean
    )
    assert not torch.allclose(
        forecast.point_estimate, forecast.parameters[..., :1] * stdev + mean
    )


def test_the_constraint_is_handed_the_target_channels_own_mean_and_standard_deviation():
    x_enc, _ = inputs()
    recording = Recording()
    model = build(output_heads=2, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc)

    mean, stdev = statistics_of(x_enc[:, :, -1:])

    assert recording.mean.shape == (BATCH_SIZE, 1, 1)
    assert recording.stdev.shape == (BATCH_SIZE, 1, 1)
    assert torch.allclose(recording.mean, mean)
    assert torch.allclose(recording.stdev, stdev)


@pytest.mark.parametrize("other", [0, 1])
def test_the_statistics_handed_over_are_no_other_channels(other):
    x_enc, _ = inputs()
    recording = Recording()
    model = build(output_heads=2, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc)

    mean, stdev = statistics_of(x_enc[:, :, other : other + 1])

    assert not torch.allclose(recording.mean, mean)
    assert not torch.allclose(recording.stdev, stdev)


def test_the_statistics_broadcast_against_the_heads():
    x_enc, _ = inputs()
    recording = Recording()
    model = build(output_heads=4, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc)

    scaled = recording.stdev * recording.parameters + recording.mean

    assert scaled.shape == recording.parameters.shape


def test_the_statistics_are_zero_and_one_when_the_standardization_is_off():
    x_enc, _ = inputs()
    recording = Recording()
    model = build(use_norm=False, output_heads=2, constraining_transformation=recording)

    with torch.no_grad():
        model.predict(x_enc)

    assert recording.mean.shape == (BATCH_SIZE, 1, 1)
    assert torch.equal(recording.mean, torch.zeros_like(recording.mean))
    assert torch.equal(recording.stdev, torch.ones_like(recording.stdev))


def test_no_gradient_reaches_the_model_through_either_statistic():
    x_enc, _ = inputs()
    x_enc.requires_grad_(True)
    recording = Recording()
    model = build(output_heads=1, constraining_transformation=recording)

    model.predict(x_enc)

    assert not recording.mean.requires_grad
    assert recording.mean.grad_fn is None
    assert not recording.stdev.requires_grad
    assert recording.stdev.grad_fn is None


def test_a_forecast_made_only_of_the_statistics_carries_no_gradient_at_all():
    x_enc, _ = inputs()
    x_enc.requires_grad_(True)
    model = build(output_heads=1, constraining_transformation=the_statistics_alone)

    forecast = model.predict(x_enc)

    assert not forecast.parameters.requires_grad
    assert not forecast.point_estimate.requires_grad


def test_the_heads_still_carry_gradient_back_into_the_pipeline():
    x_enc, _ = inputs()
    model = build(output_heads=1)

    model.predict(x_enc).point_estimate.sum().backward()

    assert model.patch_embedding.value_embedding.weight.grad is not None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_heads": 0}, "at least"),
        ({"patch_len": 13}, "does not divide into patches"),
        ({"series_dim": 2}, "indistinguishable"),
    ],
)
def test_a_configuration_that_would_fail_silently_is_refused_at_construction(
    overrides, message
):
    with pytest.raises(ConfigurationError, match=message):
        build(**overrides)


@pytest.mark.parametrize(
    "parameter",
    ["output_heads", "constraining_transformation", "point_estimate"],
)
def test_a_config_asking_for_a_constructor_parameter_is_refused_not_ignored(parameter):
    config = dict(multi_head_gcgnet_configs.exogenous_search_space()[0])
    config[parameter] = 2 if parameter == "output_heads" else softplus

    with pytest.raises(ConfigurationError, match=parameter):
        multi_head_gcgnet.builder()(config)


@pytest.mark.parametrize("features", ["M", "S"])
def test_a_config_outside_the_short_term_exogenous_setting_is_refused(features):
    config = dict(multi_head_gcgnet_configs.exogenous_search_space()[0])
    config["features"] = features

    with pytest.raises(ConfigurationError, match="head bank has already"):
        multi_head_gcgnet.builder()(config)


def test_a_config_asking_to_read_the_true_future_is_refused_naming_table_three():
    config = dict(multi_head_gcgnet_configs.exogenous_search_space()[0])
    config[multi_head_gcgnet.USE_FUTURE_EXOG_KEY] = True

    with pytest.raises(ConfigurationError, match="Table 3"):
        multi_head_gcgnet.builder()(config)


def test_the_key_the_builder_refuses_is_the_one_gcgnets_configs_spell():
    assert multi_head_gcgnet.USE_FUTURE_EXOG_KEY == gcgnet_configs.USE_FUTURE_EXOG_KEY


def test_the_short_term_exogenous_setting_is_what_the_builder_accepts():
    config = dict(multi_head_gcgnet_configs.exogenous_search_space()[0])

    assert config["features"] == multi_head_gcgnet.EXOGENOUS_FEATURES
    assert multi_head_gcgnet.builder()(config) is not None
