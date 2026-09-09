from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from halo import vendor


def crosslinear_root() -> Path:
    return vendor.tree(vendor.CROSSLINEAR)


def model_module() -> ModuleType:
    vendor.ensure_tree(vendor.CROSSLINEAR)
    vendor.add_vendor_dir_to_sys_path()
    return importlib.import_module(f"{vendor.CROSSLINEAR}.models.CrossLinear")


def model_class() -> type:
    return model_module().Model
