from __future__ import annotations

import os
import shlex
from pathlib import Path

from halo import splits
from halo.errors import ConfigurationError
from halo.splits import Split

__all__ = [
    "DB_NAME",
    "LEGACY_DB_NAME",
    "RESULTS_DIR_ENV_VAR",
    "ConfigurationError",
    "db_path",
    "results_dir",
]

RESULTS_DIR_ENV_VAR = "HALO_RESULTS_DIR"

DB_NAME = "results.db"

LEGACY_DB_NAME = "long_term_results.db"


def results_dir() -> Path:
    configured = os.environ.get(RESULTS_DIR_ENV_VAR, "").strip()
    if not configured:
        raise ConfigurationError(
            f"${RESULTS_DIR_ENV_VAR} is not set. It must name a directory "
            "outside this repository, where the results database and its CSVs "
            "are written -- results outlive the branch that produced them, and "
            "a database does not belong in version control. For example:\n"
            f"    export {RESULTS_DIR_ENV_VAR}=~/some/results/dir"
        )
    directory = Path(configured).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def db_path(split: Split) -> Path:
    directory = results_dir()
    path = directory / f"{split.file_prefix}{DB_NAME}"
    legacy = directory / LEGACY_DB_NAME
    if split == splits.VALIDATION and not path.exists() and legacy.exists():
        raise ConfigurationError(
            f"{directory} holds {LEGACY_DB_NAME}, the old name for the results "
            f"database, and no {DB_NAME}. Rename it -- starting a fresh "
            "database instead would report no results and re-run every finished "
            "run in it:\n"
            f"    mv {shlex.quote(str(legacy))} {shlex.quote(str(path))}"
        )
    return path
