from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    name: str

    loader_flag: str

    file_prefix: str

    paper_names: bool


VALIDATION = Split("validation", "val", "", paper_names=False)

TEST = Split("test", "test", "test_", paper_names=True)

SPLITS: dict[str, Split] = {split.name: split for split in (VALIDATION, TEST)}

DEFAULT = VALIDATION
