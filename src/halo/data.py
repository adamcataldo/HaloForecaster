from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from halo import vendor
from halo.errors import ConfigurationError


@dataclass(frozen=True)
class DatasetSpec:
    name: str

    data: str

    subdir: str

    data_path: str

    n_vars: int

    freq: str


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec("NP", "custom", "dataset/EPF", "NP.csv", 3, "h"),
        DatasetSpec("PJM", "custom", "dataset/EPF", "PJM.csv", 3, "h"),
        DatasetSpec("BE", "custom", "dataset/EPF", "BE.csv", 3, "h"),
        DatasetSpec("FR", "custom", "dataset/EPF", "FR.csv", 3, "h"),
        DatasetSpec("DE", "custom", "dataset/EPF", "DE.csv", 3, "h"),
    )
}

DATASET_NAMES: tuple[str, ...] = tuple(DATASETS)


def _ensured_root_of(name: str) -> Path:
    return vendor.ensure_tree(vendor.TIMEXER) / DATASETS[name].subdir


def ensure_sources(names: list[str] | tuple[str, ...]) -> None:
    for name in names:
        _ensured_root_of(name)


def fetch(names: list[str] | tuple[str, ...]) -> dict[str, str]:
    specs = [DATASETS[name] for name in names]
    roots = {name: str(_ensured_root_of(name)) for name in names}

    _verify(specs, roots)
    return roots


def _verify(specs: list[DatasetSpec], roots: dict[str, str]) -> None:
    missing = [
        f"{spec.name}: {Path(roots[spec.name]) / spec.data_path}"
        for spec in specs
        if not (Path(roots[spec.name]) / spec.data_path).is_file()
    ]
    if missing:
        raise ConfigurationError(
            "these dataset files are not on disk:\n  "
            + "\n  ".join(missing)
            + "\nThey come from `uv run get_deps.py`. Refusing to continue, "
            "because the vendored loaders would silently download unpinned "
            "replacements and the results would not be reproducible."
        )
