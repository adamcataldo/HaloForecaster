from __future__ import annotations

from typing import Any, cast

import torch

from halo import (
    crosslinear_configs,
    gaussian_head,
    multi_head_linear_configs,
    registry,
    seeding,
    settings,
    tslib,
)
from halo.multi_head_linear import MultiHeadLinear

SETTING = settings.SHORT_TERM_EXOGENOUS
SEED = 20260827
BATCH_SIZE = 3
MARKET = "NP"

SMALL: dict[str, Any] = {"d_model": 32, "d_ff": 64}


def config() -> dict[str, Any]:
    inherited = next(
        config
        for config in crosslinear_configs.exogenous_configs()
        if config["dataset_name"] == MARKET
    )
    return {**inherited, **SMALL}


def published_model(config: dict[str, Any]) -> MultiHeadLinear:
    seeding.seed_everything(SEED)
    built = registry.entry(SETTING.name, multi_head_linear_configs.MODEL_NAME).build(
        config
    )
    return cast(MultiHeadLinear, built).eval()


def batch(config: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tslib.example_batch(config, BATCH_SIZE, seeding.torch_generator(SEED))


def elsewhere(window: torch.Tensor) -> torch.Tensor:
    return window * 100 + 50


def test_the_published_head_count_and_callables_are_what_the_registry_builds():
    model = published_model(config())

    assert isinstance(model, MultiHeadLinear)
    assert len(model.heads) == multi_head_linear_configs.OUTPUT_HEADS
    assert model.constraining_transformation is gaussian_head.scale_in_target_units
    assert model.point_estimate is gaussian_head.location


def test_the_replacement_future_is_a_different_future():
    _, _, x_dec, x_mark_dec = batch(config())

    assert not torch.allclose(x_dec, elsewhere(x_dec))
    assert not torch.allclose(x_mark_dec, elsewhere(x_mark_dec))


def test_an_evaluation_forecast_does_not_move_when_the_future_windows_are_replaced():
    probe = config()
    model = published_model(probe)
    x_enc, x_mark_enc, x_dec, x_mark_dec = batch(probe)

    with torch.no_grad():
        given = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        replaced = model(x_enc, x_mark_enc, elsewhere(x_dec), elsewhere(x_mark_dec))

    assert torch.equal(given, replaced)


def test_the_distributional_parameters_do_not_move_either():
    probe = config()
    model = published_model(probe)
    x_enc, x_mark_enc, _, _ = batch(probe)

    with torch.no_grad():
        given = model.predict(x_enc, x_mark_enc)
        replaced = model.predict(x_enc, elsewhere(x_mark_enc))

    assert given.parameters.shape[-1] == multi_head_linear_configs.OUTPUT_HEADS
    assert torch.equal(given.parameters, replaced.parameters)
    assert torch.equal(given.point_estimate, replaced.point_estimate)


def test_a_replaced_lookback_does_move_the_forecast_so_the_probe_can_detect_movement():
    probe = config()
    model = published_model(probe)
    x_enc, x_mark_enc, x_dec, x_mark_dec = batch(probe)

    with torch.no_grad():
        given = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        replaced = model(elsewhere(x_enc), x_mark_enc, x_dec, x_mark_dec)

    assert not torch.allclose(given, replaced)
