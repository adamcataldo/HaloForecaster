from __future__ import annotations

from typing import cast

import pytest
import torch

from halo import gcgnet_adapter, gcgnet_configs, objectives, registry, seeding
from halo.errors import ConfigurationError

SETTING = "short_term_exogenous"
SEED = 20260820
BATCH_SIZE = 4
CPU = torch.device("cpu")


def config_for(market: str = "NP") -> dict:
    return next(
        config
        for config in gcgnet_configs.exogenous_configs()
        if config["dataset_name"] == market
    )


def adapter_with(use_future_exog: bool, config: dict) -> gcgnet_adapter.GCGNetAdapter:
    seeding.seed_everything(SEED)
    return (
        gcgnet_adapter.GCGNetAdapter(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            patch_len=config["patch_len"],
            enc_in=config["enc_in"],
            series_dim=config["series_dim"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            n_heads=config["n_heads"],
            e_layers=config["e_layers"],
            rank=config["rank"],
            dropout=config["dropout"],
            use_norm=config["use_norm"],
            use_future_exog=use_future_exog,
        )
        .float()
        .eval()
    )


def history(config: dict) -> torch.Tensor:
    return torch.randn(
        BATCH_SIZE,
        config["seq_len"],
        config["enc_in"],
        generator=seeding.torch_generator(SEED),
    )


def two_different_futures(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (BATCH_SIZE, config["pred_len"], config["enc_in"])
    zeros = torch.zeros(*shape)
    elsewhere = (
        torch.randn(*shape, generator=seeding.torch_generator(SEED + 1)) * 100 + 50
    )
    return zeros, elsewhere


def forecast_from(model: gcgnet_adapter.GCGNetAdapter, x_enc, future) -> torch.Tensor:
    with torch.no_grad():
        seeding.seed_everything(SEED)
        forecast, _ = model.forecast_with_auxiliary(x_enc, future)
    return forecast


def test_an_evaluation_forecast_does_not_move_when_the_future_is_replaced():
    config = config_for()
    model = adapter_with(False, config)
    x_enc = history(config)
    zeros, elsewhere = two_different_futures(config)

    assert torch.equal(
        forecast_from(model, x_enc, zeros),
        forecast_from(model, x_enc, elsewhere),
    )


def test_reading_the_future_exogenous_window_would_move_that_forecast():
    config = config_for()
    model = adapter_with(True, config)
    x_enc = history(config)
    zeros, elsewhere = two_different_futures(config)

    assert not torch.equal(
        forecast_from(model, x_enc, zeros),
        forecast_from(model, x_enc, elsewhere),
    )


def test_the_plain_forward_pass_agrees_with_an_absent_future():
    config = config_for()
    model = adapter_with(False, config)
    x_enc = history(config)
    zeros, _ = two_different_futures(config)

    with torch.no_grad():
        seeding.seed_everything(SEED)
        plain = model(x_enc, None, None, None)

    assert torch.equal(plain, forecast_from(model, x_enc, zeros))


def test_a_configuration_asking_to_read_the_future_is_refused():
    config = dict(config_for())
    config[gcgnet_configs.USE_FUTURE_EXOG_KEY] = True

    with pytest.raises(ConfigurationError, match="Table 3"):
        gcgnet_configs.build(config)


def test_the_scored_column_is_rotated_to_the_front_for_the_model():
    window = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    rotated = gcgnet_adapter.endogenous_first(window, 1)

    assert torch.equal(rotated, torch.tensor([[[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]]))


def test_the_objective_adds_the_models_own_number_to_its_absolute_error():
    config = config_for()
    seeding.seed_everything(SEED)
    model = cast(
        gcgnet_adapter.GCGNetAdapter, registry.entry(SETTING, "GCGNet").build(config)
    )
    batch = (
        history(config),
        torch.randn(
            BATCH_SIZE,
            config["label_len"] + config["pred_len"],
            config["enc_in"],
            generator=seeding.torch_generator(SEED + 2),
        ),
        torch.randn(BATCH_SIZE, config["seq_len"], 4),
        torch.randn(BATCH_SIZE, config["label_len"] + config["pred_len"], 4),
    )

    seeding.seed_everything(SEED)
    total = gcgnet_configs.gcgnet_objective(model, batch, config, CPU)

    seeding.seed_everything(SEED)
    prepared = objectives.prepare(batch, config, CPU)
    forecast, auxiliary = model.forecast_with_auxiliary(prepared.x_enc, prepared.future)
    error = torch.nn.functional.l1_loss(
        objectives.scored(forecast, config), prepared.target
    )

    assert torch.isfinite(total)
    assert torch.allclose(total, error + auxiliary)


def test_the_future_window_carries_every_column_and_the_target_is_its_last():
    config = config_for()
    batch = (
        history(config),
        torch.randn(
            BATCH_SIZE,
            config["label_len"] + config["pred_len"],
            config["enc_in"],
            generator=seeding.torch_generator(SEED + 3),
        ),
        torch.randn(BATCH_SIZE, config["seq_len"], 4),
        torch.randn(BATCH_SIZE, config["label_len"] + config["pred_len"], 4),
    )

    prepared = objectives.prepare(batch, config, CPU)

    assert prepared.future.shape == (
        BATCH_SIZE,
        config["pred_len"],
        config["enc_in"],
    )
    assert torch.equal(prepared.future[:, :, -1:], prepared.target)
