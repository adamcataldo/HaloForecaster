from __future__ import annotations

from typing import cast

import pytest
import torch

from halo import (
    laplace_nll,
    multi_head_gcgnet_configs,
    objectives,
    registry,
    seeding,
    settings,
)
from halo.forecast import Forecast
from halo.multi_head_gcgnet import MultiHeadGCGNet

SETTING = settings.SHORT_TERM_EXOGENOUS
MODEL = multi_head_gcgnet_configs.MODEL_NAME

BATCH = 2
SEQ_LEN = 8
LABEL_LEN = 2
PRED_LEN = 3
CHANNELS = 3
SEED = 20260827
DEVICE = torch.device("cpu")

CONFIG = {"features": "MS", "pred_len": PRED_LEN, "label_len": LABEL_LEN}

TARGET = 2.0
RAW_LOCATION = -5.0
SCALE = 3.0
POINT_ESTIMATE = 3.5
AUXILIARY = 0.75


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


class TwoHeadedWithAnAuxiliary(torch.nn.Module):
    def __init__(self, auxiliary: float = AUXILIARY):
        super().__init__()
        self.forecast = Forecast(
            parameters=torch.cat([horizon(RAW_LOCATION), horizon(SCALE)], dim=-1),
            point_estimate=horizon(POINT_ESTIMATE),
        )
        self.auxiliary = torch.tensor(auxiliary)

    def predict_with_auxiliary(self, x_enc, future):
        return self.forecast, self.auxiliary


def loss_of(location, scale):
    return float(
        laplace_nll.laplace_nll_loss(horizon(location), horizon(scale), horizon(TARGET))
    )


def stepped_on(auxiliary: float = AUXILIARY):
    return float(
        multi_head_gcgnet_configs.laplace_nll_objective(
            TwoHeadedWithAnAuxiliary(auxiliary), batch(), CONFIG, DEVICE
        )
    )


def test_the_likelihood_is_the_log_scale_plus_the_absolute_residual_over_it():
    location = torch.tensor([[[1.0, -2.0]]])
    scale = torch.tensor([[[2.0, 0.5]]])
    target = torch.tensor([[[4.0, -1.0]]])

    measured = laplace_nll.laplace_nll_loss(location, scale, target)

    expected = (torch.log(scale) + (target - location).abs() / scale).mean()
    assert torch.allclose(measured, expected)


def test_the_likelihood_averages_rather_than_sums_over_the_window_and_horizon():
    location = torch.zeros(4, 6, 1)
    scale = torch.ones(4, 6, 1)
    target = torch.full((4, 6, 1), 2.0)

    doubled = laplace_nll.laplace_nll_loss(
        torch.cat([location, location]),
        torch.cat([scale, scale]),
        torch.cat([target, target]),
    )

    assert torch.allclose(
        doubled, laplace_nll.laplace_nll_loss(location, scale, target)
    )


@pytest.mark.parametrize("scale", [0.25, 1.0, 7.0])
def test_at_a_fixed_scale_it_ranks_locations_the_way_absolute_error_does(scale):
    target = torch.tensor([[[1.0, -3.0, 0.5]]])
    near = torch.tensor([[[1.2, -3.1, 0.4]]])
    far = torch.tensor([[[4.0, 2.0, -6.0]]])
    fixed = torch.full_like(target, scale)

    nearer = laplace_nll.laplace_nll_loss(near, fixed, target)
    further = laplace_nll.laplace_nll_loss(far, fixed, target)

    assert nearer < further
    assert (near - target).abs().mean() < (far - target).abs().mean()


def test_a_scale_small_enough_to_overflow_the_quotient_is_floored():
    location = torch.zeros(1, 1, 1)
    target = torch.ones(1, 1, 1)
    floored = laplace_nll.laplace_nll_loss(
        location, torch.full((1, 1, 1), 1e-30), target
    )

    assert torch.isfinite(floored)
    assert torch.allclose(
        floored,
        laplace_nll.laplace_nll_loss(
            location, torch.full((1, 1, 1), laplace_nll.SCALE_FLOOR), target
        ),
    )


def test_a_floored_scale_still_backpropagates_a_finite_gradient():
    scale = torch.full((1, 1, 1), 1e-30, requires_grad=True)
    laplace_nll.laplace_nll_loss(
        torch.zeros(1, 1, 1), scale, torch.ones(1, 1, 1)
    ).backward()

    assert scale.grad is not None
    assert torch.isfinite(scale.grad).all()


def test_the_objective_is_the_likelihood_of_the_point_estimate_and_the_second_head():
    assert stepped_on() == pytest.approx(
        loss_of(POINT_ESTIMATE, SCALE) + AUXILIARY, rel=1e-6
    )


def test_the_location_is_the_point_estimate_not_the_raw_first_head():
    assert stepped_on() != pytest.approx(loss_of(RAW_LOCATION, SCALE) + AUXILIARY)


def test_exchanging_the_location_and_the_scale_is_a_different_number():
    assert stepped_on() != pytest.approx(loss_of(SCALE, POINT_ESTIMATE) + AUXILIARY)


def test_the_harness_channel_slice_over_the_head_axis_would_take_the_scale():
    sliced = objectives.scored(TwoHeadedWithAnAuxiliary().forecast.parameters, CONFIG)

    assert torch.equal(sliced, horizon(SCALE))
    assert stepped_on() != pytest.approx(loss_of(SCALE, SCALE) + AUXILIARY)


def test_the_models_own_number_is_added_with_no_coefficient():
    without = stepped_on(0.0)

    assert stepped_on(AUXILIARY) - without == pytest.approx(AUXILIARY, rel=1e-6)
    assert stepped_on(2 * AUXILIARY) - without == pytest.approx(2 * AUXILIARY, rel=1e-6)


def test_the_target_is_the_scored_column_the_rest_of_the_harness_uses():
    prepared = objectives.prepare(batch(), CONFIG, DEVICE)

    assert torch.equal(prepared.target, horizon(TARGET))


def small_config():
    entry = registry.entry(SETTING.name, MODEL)
    config = dict(entry.trials()[0])
    config.update(d_model=32, d_ff=32)
    return config


def test_the_registered_model_declares_the_laplace_objective_its_module_defines():
    entry = registry.entry(SETTING.name, MODEL)

    assert entry.objective is multi_head_gcgnet_configs.laplace_nll_objective
    assert entry.device == multi_head_gcgnet_configs.DEVICE


def test_the_registered_model_ends_in_a_location_and_a_positive_scale():
    config = small_config()
    seeding.seed_everything(SEED)
    model = cast(
        MultiHeadGCGNet, registry.entry(SETTING.name, MODEL).build(config)
    ).eval()
    x_enc = torch.randn(
        BATCH,
        config["seq_len"],
        config["enc_in"],
        generator=seeding.torch_generator(SEED),
    )

    with torch.no_grad():
        seeding.seed_everything(SEED)
        forecast = model.predict(x_enc)
        seeding.seed_everything(SEED)
        forward = model(x_enc, None, None, None)

    assert len(model.heads) == 2
    assert forecast.parameters.shape[-1] == 2
    assert (forecast.parameters[..., 1] > 0).all()
    assert torch.equal(forward, forecast.point_estimate)


def test_the_registered_model_steps_on_a_finite_number():
    config = small_config()
    seeding.seed_everything(SEED)
    entry = registry.entry(SETTING.name, MODEL)
    model = entry.build(config)
    sample = (
        torch.randn(
            BATCH,
            config["seq_len"],
            config["enc_in"],
            generator=seeding.torch_generator(SEED),
        ),
        torch.randn(
            BATCH,
            config["label_len"] + config["pred_len"],
            config["enc_in"],
            generator=seeding.torch_generator(SEED + 1),
        ),
        torch.zeros(BATCH, config["seq_len"], 4),
        torch.zeros(BATCH, config["label_len"] + config["pred_len"], 4),
    )

    total = entry.objective(model, sample, config, DEVICE)

    assert torch.isfinite(total)
