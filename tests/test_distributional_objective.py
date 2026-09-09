from __future__ import annotations

from typing import cast

import pytest
import torch

from halo import (
    beta_nll,
    crosslinear_sbs_configs,
    gaussian_head,
    gcgnet_configs,
    multi_head_gcgnet_configs,
    multi_head_linear_configs,
    multi_head_timexer_configs,
    multi_head_timexer_sweep_configs,
    objectives,
    registry,
    seeding,
    settings,
    timexer_sbs_configs,
    tslib,
)
from halo.forecast import Forecast
from halo.multi_head_timexer import MultiHeadTimeXer

SETTING = settings.SHORT_TERM_EXOGENOUS
MULTI_HEAD = multi_head_timexer_configs.MODEL_NAME
MULTI_HEAD_SWEEP = multi_head_timexer_sweep_configs.MODEL_NAME
MULTI_HEAD_GCGNET = multi_head_gcgnet_configs.MODEL_NAME

MULTI_HEAD_LINEAR = multi_head_linear_configs.MODEL_NAME

CROSSLINEAR_SBS = crosslinear_sbs_configs.MODEL_NAME

TIMEXER_SBS = timexer_sbs_configs.MODEL_NAME

BATCH = 2
SEQ_LEN = 8
LABEL_LEN = 2
PRED_LEN = 3
CHANNELS = 3
SEED = 20260818
DEVICE = torch.device("cpu")

CONFIG = {"features": "MS", "pred_len": PRED_LEN, "label_len": LABEL_LEN}

TARGET = 2.0
RAW_LOCATION = -5.0
SCALE = 2.0
POINT_ESTIMATE = 3.0


def horizon(value):
    return torch.full((BATCH, PRED_LEN, 1), value)


def channel_marked(length):
    return torch.arange(float(CHANNELS)).expand(BATCH, length, CHANNELS)


def batch():
    return (
        torch.zeros(BATCH, SEQ_LEN, CHANNELS),
        channel_marked(LABEL_LEN + PRED_LEN),
        torch.zeros(BATCH, SEQ_LEN, 4),
        torch.zeros(BATCH, LABEL_LEN + PRED_LEN, 4),
    )


class TwoHeaded(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forecast = Forecast(
            parameters=torch.cat([horizon(RAW_LOCATION), horizon(SCALE)], dim=-1),
            point_estimate=horizon(POINT_ESTIMATE),
        )

    def predict(self, x_enc, x_mark_enc):
        return self.forecast


def loss_of(mean, variance):
    return float(
        beta_nll.beta_nll_loss(horizon(mean), horizon(variance), horizon(TARGET))
    )


def stepped_on():
    return float(gaussian_head.beta_nll_objective(TwoHeaded(), batch(), CONFIG, DEVICE))


def test_the_objective_is_beta_nll_of_the_point_estimate_against_the_second_head():
    assert stepped_on() == pytest.approx(loss_of(POINT_ESTIMATE, SCALE**2))


def test_the_location_is_the_point_estimate_not_the_first_head_before_its_inverse():
    assert stepped_on() != pytest.approx(loss_of(RAW_LOCATION, SCALE**2))


def test_exchanging_the_location_and_the_scale_is_a_different_number():
    assert stepped_on() != pytest.approx(loss_of(SCALE, POINT_ESTIMATE**2))


def test_the_harness_channel_slice_over_the_head_axis_would_take_the_scale():
    sliced = objectives.scored(TwoHeaded().forecast.parameters, CONFIG)

    assert torch.equal(sliced, horizon(SCALE))
    assert stepped_on() != pytest.approx(loss_of(SCALE, SCALE**2))


def test_the_target_is_the_scored_column_the_rest_of_the_harness_uses():
    prepared = objectives.prepare(batch(), CONFIG, DEVICE)

    assert torch.equal(prepared.target, horizon(TARGET))


def test_the_reduction_is_the_first_head():
    parameters = torch.tensor([[[-4.0, -1.0, 0.5]]])

    assert torch.equal(gaussian_head.location(parameters), parameters[..., :1])


def test_every_model_declares_the_objective_its_own_module_defines():
    declared = {
        gcgnet_configs.MODEL_NAME: gcgnet_configs.gcgnet_objective,
        MULTI_HEAD: gaussian_head.beta_nll_objective,
        MULTI_HEAD_SWEEP: gaussian_head.beta_nll_objective,
        MULTI_HEAD_GCGNET: multi_head_gcgnet_configs.laplace_nll_objective,
        MULTI_HEAD_LINEAR: gaussian_head.beta_nll_objective,
        CROSSLINEAR_SBS: gaussian_head.beta_nll_objective,
        TIMEXER_SBS: gaussian_head.beta_nll_objective,
    }

    for (_, model_name), entry in registry.MODEL_REGISTRY.items():
        assert entry.objective is declared.get(model_name, objectives.squared_error)


def multi_head_config():
    entry = registry.entry(SETTING.name, MULTI_HEAD)
    config = dict(entry.configs()[0])
    config.update(d_model=32, d_ff=64, e_layers=1, n_heads=4)
    return config


def test_the_multi_head_transformation_leaves_the_location_in_standardized_units():
    parameters = torch.tensor([[[-4.0, -1.0]]])
    stdev = torch.tensor([[[7.0]]])

    constrained = gaussian_head.scale_in_target_units(
        parameters, torch.zeros_like(stdev), stdev
    )

    assert torch.equal(constrained[..., 0], parameters[..., 0])


def test_the_multi_head_scale_is_the_windows_deviation_times_a_softplus():
    parameters = torch.tensor([[[-4.0, -1.0]]])
    stdev = torch.tensor([[[7.0]]])

    constrained = gaussian_head.scale_in_target_units(
        parameters, torch.zeros_like(stdev), stdev
    )

    assert (constrained[..., 1:] > 0).all()
    assert torch.allclose(
        constrained[..., 1:],
        stdev * torch.nn.functional.softplus(parameters[..., 1:]),
    )


def test_the_multi_head_scale_carries_the_window_and_the_bare_softplus_does_not():
    parameters = torch.tensor([[[-4.0, -1.0]]])
    stdev = torch.tensor([[[7.0]]])

    constrained = gaussian_head.scale_in_target_units(
        parameters, torch.zeros_like(stdev), stdev
    )

    assert not torch.allclose(
        constrained[..., 1:], torch.nn.functional.softplus(parameters[..., 1:])
    )


def test_the_registered_multi_head_model_ends_in_a_location_and_a_scale():
    entry = registry.entry(SETTING.name, MULTI_HEAD)

    model = cast(MultiHeadTimeXer, entry.build(multi_head_config()))

    assert len(model.heads) == 2


def test_the_registered_multi_head_model_scores_under_the_shared_beta_nll_objective():
    config = multi_head_config()
    entry = registry.entry(SETTING.name, MULTI_HEAD)
    seeding.seed_everything(SEED)
    model = cast(MultiHeadTimeXer, entry.build(config)).eval()
    x_enc, x_mark_enc, x_dec, x_mark_dec = tslib.example_batch(
        config, BATCH, seeding.torch_generator(SEED)
    )

    with torch.no_grad():
        forecast = model.predict(x_enc, x_mark_enc)
        forward = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert forecast.parameters.shape[-1] == 2
    assert (forecast.parameters[..., 1] > 0).all()
    assert torch.equal(forward, forecast.point_estimate)
    assert entry.objective is gaussian_head.beta_nll_objective
