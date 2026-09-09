from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from halo import vendor


def gcgnet_root() -> Path:
    return vendor.tree(vendor.GCGNET)


def add_to_sys_path() -> Path:
    root = vendor.ensure_tree(vendor.GCGNET)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def model_module() -> ModuleType:
    add_to_sys_path()
    import importlib

    return importlib.import_module("ts_benchmark.baselines.GCGNet.models.gcgnet_model")


def model_class() -> type:
    return model_module().GCGNetModel
