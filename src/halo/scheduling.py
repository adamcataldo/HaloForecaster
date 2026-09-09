from __future__ import annotations

from collections.abc import Iterable
from math import lcm
from typing import Any

PARALLELISM_KEY = "parallelism"

SLOT_RESOURCE = "halo_slots"


def parallelism_of(config: dict[str, Any]) -> int:
    value = config.get(PARALLELISM_KEY)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            f"config {config.get('dataset_name')}/S={config.get('pred_len')} has "
            f"{PARALLELISM_KEY}={value!r}; expected a positive integer"
        )
    return value


def slot_costs(configs: Iterable[dict[str, Any]]) -> tuple[int, list[int]]:
    parallelisms = [parallelism_of(config) for config in configs]
    if not parallelisms:
        return 1, []
    budget = lcm(*parallelisms)
    return budget, [budget // p for p in parallelisms]


def admits(budget: int, spent: int, cost: int) -> bool:
    return spent + cost <= budget
