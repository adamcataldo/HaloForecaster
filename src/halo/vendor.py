from __future__ import annotations

import os
import sys
from pathlib import Path

from halo.errors import ConfigurationError

VENDOR_DIR_ENV_VAR = "HALO_VENDOR_DIR"

TIME_SERIES_LIBRARY = "time_series_library"
TIMEXER = "timexer"
GCGNET = "gcgnet"
CROSSLINEAR = "crosslinear"


def vendor_dir() -> Path:
    override = os.environ.get(VENDOR_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "third_party"


def tree(name: str) -> Path:
    return vendor_dir() / name


def ensure_tree(name: str) -> Path:
    root = tree(name)
    if not root.is_dir():
        raise ConfigurationError(
            f"Vendored {name} not found at {root}. Fetch it with `uv run get_deps.py`."
        )
    return root


def add_vendor_dir_to_sys_path() -> Path:
    root = vendor_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
