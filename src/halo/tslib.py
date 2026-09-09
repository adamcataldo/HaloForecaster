from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

from halo import vendor

N_TIME_FEATURES = 4


def tslib_root() -> Path:
    return vendor.tree(vendor.TIME_SERIES_LIBRARY)


def add_to_sys_path() -> Path:
    root = vendor.ensure_tree(vendor.TIME_SERIES_LIBRARY)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ensure_available() -> Path:
    return add_to_sys_path()


def data_provider() -> ModuleType:
    add_to_sys_path()
    from data_provider import data_factory

    return data_factory


def model_module(model_name: str) -> ModuleType:
    add_to_sys_path()
    import importlib

    return importlib.import_module(f"models.{model_name}")


def model_builder(model_name: str) -> Callable[[dict[str, Any]], torch.nn.Module]:
    def build(config: dict[str, Any]) -> torch.nn.Module:
        module = model_module(model_name)
        return module.Model(as_namespace(config)).float()

    return build


def adjust_learning_rate():
    add_to_sys_path()
    from utils.tools import adjust_learning_rate as fn

    return fn


def as_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def example_batch(
    config: dict[str, Any],
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    seq_len = config["seq_len"]
    dec_len = config["label_len"] + config["pred_len"]
    return (
        randn(batch_size, seq_len, config["enc_in"]),
        randn(batch_size, seq_len, N_TIME_FEATURES),
        randn(batch_size, dec_len, config["dec_in"]),
        randn(batch_size, dec_len, N_TIME_FEATURES),
    )
