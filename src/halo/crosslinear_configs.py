from __future__ import annotations

from typing import Any

import torch

from halo import crosslinear, scheduling, settings, tslib
from halo.data import DATASETS

MODEL_NAME = "CrossLinear"

SERIES_DIM = 1


def build(config: dict[str, Any]) -> torch.nn.Module:
    return crosslinear.model_class()(tslib.as_namespace(config)).float()


EXOGENOUS_CONSTANTS: dict[str, Any] = {
    "setting": settings.SHORT_TERM_EXOGENOUS.name,
    "model": MODEL_NAME,
    "task_name": "long_term_forecast",
    "features": "MS",
    "seq_len": settings.SHORT_TERM_EXOGENOUS.seq_len,
    "label_len": settings.SHORT_TERM_EXOGENOUS.label_len,
    "target": "OT",
    "embed": "timeF",
    "train_epochs": 50,
    "patience": 5,
    "lradj": "type3",
    "num_workers": 0,
    "seasonal_patterns": None,
    "augmentation_ratio": 0,
}

EXOGENOUS_TABLE: dict[str, tuple[int, int, int, float, float, int, float]] = {
    "NP": (16, 768, 2048, 1.0, 1.0, 8, 1e-3),
    "PJM": (16, 1024, 4096, 1.0, 1.0, 4, 1e-3),
    "BE": (24, 1024, 2048, 1.0, 1.0, 8, 1e-3),
    "FR": (24, 1024, 2048, 2.0, 1.0, 16, 1e-4),
    "DE": (16, 1024, 2048, 2.0, 1.0, 16, 1e-3),
}

EXOGENOUS_PARALLELISM: dict[str, int] = {
    "NP": 8,
    "PJM": 8,
    "BE": 8,
    "FR": 8,
    "DE": 8,
}

_markets = frozenset(settings.SHORT_TERM_EXOGENOUS.datasets)
if EXOGENOUS_TABLE.keys() != _markets:
    raise ValueError(
        "every market of the "
        f"{settings.SHORT_TERM_EXOGENOUS.name} setting needs a row in "
        f"{MODEL_NAME}'s table, and every row needs a market: "
        f"{sorted(_markets)} are run but {sorted(EXOGENOUS_TABLE)} carry rows."
    )
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
            (
                patch_len,
                d_model,
                d_ff,
                alpha,
                beta,
                batch_size,
                learning_rate,
            ) = EXOGENOUS_TABLE[name]
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
                    "patch_len": patch_len,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "alpha": alpha,
                    "beta": beta,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    scheduling.PARALLELISM_KEY: EXOGENOUS_PARALLELISM[name],
                }
            )
    return out
