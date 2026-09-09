from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn.functional import l1_loss

from halo import gcgnet_adapter, objectives, scheduling, settings
from halo.data import DATASETS
from halo.errors import ConfigurationError

MODEL_NAME = "GCGNet"

SERIES_DIM = 1

USE_FUTURE_EXOG_KEY = "use_future_exog"


def build(config: dict[str, Any]) -> torch.nn.Module:
    use_future_exog = config[USE_FUTURE_EXOG_KEY]
    if use_future_exog:
        raise ConfigurationError(
            f"config asks for {USE_FUTURE_EXOG_KEY}=True, which lets GCGNet "
            "read the true future covariates over the horizon it is "
            "forecasting. That is upstream's headline setting and it is not a "
            "setting anything else in this table runs under. We run the "
            "paper's Table 3 protocol, where the future exogenous variables "
            "are unavailable and the model's own generator forecasts them."
        )
    return gcgnet_adapter.GCGNetAdapter(
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
    ).float()


def gcgnet_objective(
    model: torch.nn.Module,
    batch: tuple[Tensor, ...],
    config: dict[str, Any],
    device: torch.device,
) -> Tensor:
    prepared = objectives.prepare(batch, config, device)
    adapter = cast(gcgnet_adapter.GCGNetAdapter, model)
    forecast, auxiliary = adapter.forecast_with_auxiliary(
        prepared.x_enc, prepared.future
    )
    error = l1_loss(objectives.scored(forecast, config), prepared.target)
    return error + auxiliary


EXOGENOUS_CONSTANTS: dict[str, Any] = {
    "setting": settings.SHORT_TERM_EXOGENOUS.name,
    "model": MODEL_NAME,
    "task_name": "long_term_forecast",
    "features": "MS",
    "seq_len": settings.SHORT_TERM_EXOGENOUS.seq_len,
    "label_len": settings.SHORT_TERM_EXOGENOUS.label_len,
    "target": "OT",
    "embed": "timeF",
    "patch_len": 24,
    "series_dim": SERIES_DIM,
    "e_layers": 1,
    "n_heads": 4,
    "rank": 4,
    "dropout": 0,
    "use_norm": True,
    USE_FUTURE_EXOG_KEY: False,
    "train_epochs": 50,
    "patience": 5,
    "lradj": "type3",
    "num_workers": 0,
    "seasonal_patterns": None,
    "augmentation_ratio": 0,
}

EXOGENOUS_TABLE: dict[str, tuple[int, int, int, float]] = {
    "NP": (512, 256, 32, 1e-4),
    "PJM": (512, 512, 32, 1e-3),
    "BE": (256, 512, 32, 1e-3),
    "FR": (64, 128, 32, 1e-3),
    "DE": (64, 64, 32, 1e-3),
}

EXOGENOUS_PARALLELISM: dict[str, int] = {
    "NP": 8,
    "PJM": 8,
    "BE": 8,
    "FR": 8,
    "DE": 8,
}

_markets = frozenset(settings.SHORT_TERM_EXOGENOUS.datasets)
if EXOGENOUS_PARALLELISM.keys() != _markets:
    raise ValueError(
        "every market of the "
        f"{settings.SHORT_TERM_EXOGENOUS.name} setting needs a parallelism "
        f"number, and every number needs a market: {sorted(_markets)} are "
        f"run but {sorted(EXOGENOUS_PARALLELISM)} carry numbers."
    )


def exogenous_configs() -> list[dict[str, Any]]:
    setting = settings.SHORT_TERM_EXOGENOUS
    out: list[dict[str, Any]] = []
    for name in setting.datasets:
        spec = DATASETS[name]
        for horizon in setting.horizons:
            d_model, d_ff, batch_size, learning_rate = EXOGENOUS_TABLE[name]
            out.append(
                {
                    **EXOGENOUS_CONSTANTS,
                    "dataset_name": name,
                    "data": spec.data,
                    "data_path": spec.data_path,
                    "freq": spec.freq,
                    "enc_in": spec.n_vars,
                    "dec_in": spec.n_vars,
                    "c_out": SERIES_DIM,
                    "pred_len": horizon,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    scheduling.PARALLELISM_KEY: EXOGENOUS_PARALLELISM[name],
                }
            )
    return out
