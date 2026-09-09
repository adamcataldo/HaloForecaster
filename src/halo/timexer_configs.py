from __future__ import annotations

from typing import Any

from halo import settings
from halo.data import DATASETS

MODEL_NAME = "TimeXer"


EXOGENOUS_CONSTANTS: dict[str, Any] = {
    "setting": settings.SHORT_TERM_EXOGENOUS.name,
    "task_name": "long_term_forecast",
    "features": "MS",
    "seq_len": settings.SHORT_TERM_EXOGENOUS.seq_len,
    "label_len": settings.SHORT_TERM_EXOGENOUS.label_len,
    "target": "OT",
    "n_heads": 8,
    "dropout": 0.1,
    "patch_len": 24,
    "use_norm": 1,
    "factor": 1,
    "embed": "timeF",
    "activation": "gelu",
    "train_epochs": 50,
    "patience": 5,
    "lradj": "type3",
    "num_workers": 0,
    "seasonal_patterns": None,
    "augmentation_ratio": 0,
}

EXOGENOUS_TABLE: dict[str, tuple[int, int, int, int]] = {
    "NP": (3, 512, 512, 4),
    "PJM": (3, 512, 2048, 16),
    "BE": (2, 512, 512, 16),
    "FR": (2, 512, 2048, 16),
    "DE": (1, 512, 2048, 4),
}

EXOGENOUS_LEARNING_RATE = 1e-4

EXOGENOUS_PARALLELISM = 8


def exogenous_configs() -> list[dict[str, Any]]:
    setting = settings.SHORT_TERM_EXOGENOUS
    out: list[dict[str, Any]] = []
    for name in setting.datasets:
        spec = DATASETS[name]
        for horizon in setting.horizons:
            e_layers, d_model, d_ff, batch_size = EXOGENOUS_TABLE[name]
            out.append(
                {
                    **EXOGENOUS_CONSTANTS,
                    "model": MODEL_NAME,
                    "dataset_name": name,
                    "data": spec.data,
                    "data_path": spec.data_path,
                    "freq": spec.freq,
                    "enc_in": spec.n_vars,
                    "dec_in": spec.n_vars,
                    "c_out": 1,
                    "pred_len": horizon,
                    "e_layers": e_layers,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "batch_size": batch_size,
                    "learning_rate": EXOGENOUS_LEARNING_RATE,
                    "parallelism": EXOGENOUS_PARALLELISM,
                }
            )
    return out
