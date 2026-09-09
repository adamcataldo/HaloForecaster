from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Protocol

GRID_POINT_KEY = "grid_point"


class Scored(Protocol):
    @property
    def grid_point(self) -> str: ...

    @property
    def mse(self) -> float: ...

    @property
    def seed(self) -> int: ...


def grid_point_of(config: dict[str, Any]) -> str:
    value = config.get(GRID_POINT_KEY)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"config {config.get('dataset_name')}/S={config.get('pred_len')} has "
            f"{GRID_POINT_KEY}={value!r}; expected a non-empty string. A trial is "
            "keyed by its label, so a search space without one cannot be resumed "
            "and cannot be read back."
        )
    return value


def best[ScoredT: Scored](trials: Iterable[ScoredT]) -> ScoredT:
    candidates = list(trials)
    if not candidates:
        raise ValueError("no trials to choose between")
    return min(candidates, key=lambda trial: (trial.mse, trial.grid_point))


def winner[ScoredT: Scored](
    trials: Iterable[ScoredT], *, expected: Collection[str], seed: int
) -> ScoredT | None:
    wanted = set(expected)
    at_seed = [
        trial for trial in trials if trial.seed == seed and trial.grid_point in wanted
    ]
    if not wanted or wanted - {trial.grid_point for trial in at_seed}:
        return None
    return best(at_seed)
