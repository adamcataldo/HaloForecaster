from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from halo import (
    crosslinear_configs,
    gaussian_head,
    multi_head_linear,
    scheduling,
    search,
    seeding,
    settings,
)

MODEL_NAME = "MultiHeadLinear"

SEED_IDENTITY = crosslinear_configs.MODEL_NAME

OUTPUT_HEADS = gaussian_head.OUTPUT_HEADS


build = multi_head_linear.builder(
    output_heads=OUTPUT_HEADS,
    constraining_transformation=gaussian_head.scale_in_target_units,
    point_estimate=gaussian_head.location,
)


def _column(index: int) -> tuple[Any, ...]:
    return tuple(row[index] for row in crosslinear_configs.EXOGENOUS_TABLE.values())


def _distinct_values_of(index: int) -> tuple[Any, ...]:
    return tuple(sorted(set(_column(index))))


def _largest_value_of(index: int) -> Any:
    return max(_column(index))


def _most_common_value_of(index: int) -> Any:
    counted = Counter(_column(index))
    return max(sorted(counted), key=counted.__getitem__)


PATCH_LENGTHS: tuple[int, ...] = _distinct_values_of(0)

LATENT_WIDTHS: tuple[int, ...] = _distinct_values_of(1)

FEED_FORWARD_WIDTHS: tuple[int, ...] = _distinct_values_of(2)

ALPHAS: tuple[float, ...] = _distinct_values_of(3)

BATCH_SIZE: int = _largest_value_of(5)

LEARNING_RATE: float = _most_common_value_of(6)


@dataclass(frozen=True)
class SearchPoint:
    patch_len: int
    d_model: int
    d_ff: int
    alpha: float

    @property
    def label(self) -> str:
        return f"P{self.patch_len}-D{self.d_model}-F{self.d_ff}-A{self.alpha:g}"

    def as_config(self) -> dict[str, Any]:
        return {
            "patch_len": self.patch_len,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "alpha": self.alpha,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            search.GRID_POINT_KEY: self.label,
        }


EXOGENOUS_GRID: tuple[SearchPoint, ...] = tuple(
    SearchPoint(
        patch_len=patch_len,
        d_model=d_model,
        d_ff=d_ff,
        alpha=alpha,
    )
    for patch_len in PATCH_LENGTHS
    for d_model in LATENT_WIDTHS
    for d_ff in FEED_FORWARD_WIDTHS
    for alpha in ALPHAS
)


def _refuse_colliding_labels(grid: Sequence[SearchPoint]) -> None:
    labels = [point.label for point in grid]
    if len(set(labels)) != len(labels):
        collided = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(
            f"{collided} are the label of more than one point in the grid. A "
            "trial is stored under its label, so two points sharing one are "
            "recorded as one point measured twice -- half the search would go "
            "missing from the winner check rather than fail. The mixing "
            "coefficient is the field to look at: a formatter that renders two "
            "of its values the same way collapses them here."
        )


_refuse_colliding_labels(EXOGENOUS_GRID)


EXOGENOUS_PARALLELISM: dict[str, int] = {
    "NP": 8,
    "PJM": 8,
    "BE": 8,
    "FR": 8,
    "DE": 8,
}


def _refuse_unnumbered_markets(
    parallelism: dict[str, int], markets: Collection[str]
) -> None:
    if parallelism.keys() != frozenset(markets):
        raise ValueError(
            "every market of the "
            f"{settings.SHORT_TERM_EXOGENOUS.name} setting needs a parallelism "
            f"number, and every number needs a market: {sorted(markets)} are "
            f"searched but {sorted(parallelism)} carry numbers."
        )


_refuse_unnumbered_markets(
    EXOGENOUS_PARALLELISM, settings.SHORT_TERM_EXOGENOUS.datasets
)


def _configs_at(
    points_for: Callable[[str], Sequence[SearchPoint]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for inherited in crosslinear_configs.exogenous_configs():
        name = inherited["dataset_name"]
        for point in points_for(name):
            out.append(
                {
                    **inherited,
                    "model": MODEL_NAME,
                    seeding.SEED_IDENTITY_KEY: SEED_IDENTITY,
                    scheduling.PARALLELISM_KEY: EXOGENOUS_PARALLELISM[name],
                    **point.as_config(),
                }
            )
    return out


def exogenous_search_space() -> list[dict[str, Any]]:
    return _configs_at(lambda name: EXOGENOUS_GRID)


EXOGENOUS_TABLE: dict[str, SearchPoint] = {
    "NP": SearchPoint(patch_len=16, d_model=1024, d_ff=2048, alpha=2.0),
    "PJM": SearchPoint(patch_len=16, d_model=1024, d_ff=2048, alpha=2.0),
    "BE": SearchPoint(patch_len=16, d_model=768, d_ff=4096, alpha=2.0),
    "FR": SearchPoint(patch_len=16, d_model=768, d_ff=2048, alpha=2.0),
    "DE": SearchPoint(patch_len=16, d_model=1024, d_ff=4096, alpha=1.0),
}


def _refuse_unsettled_markets(
    table: dict[str, SearchPoint], markets: Collection[str]
) -> None:
    if table.keys() != frozenset(markets):
        raise ValueError(
            "every market of the "
            f"{settings.SHORT_TERM_EXOGENOUS.name} setting needs a settled "
            "configuration, and every configuration needs a market: "
            f"{sorted(markets)} are searched but {sorted(table)} carry "
            "configurations."
        )


_refuse_unsettled_markets(EXOGENOUS_TABLE, settings.SHORT_TERM_EXOGENOUS.datasets)


def _winners_the_grid_does_not_contain(
    table: dict[str, SearchPoint], grid: Collection[SearchPoint]
) -> dict[str, str]:
    searched = frozenset(grid)
    return {name: point.label for name, point in table.items() if point not in searched}


def _refuse_winners_the_grid_does_not_contain(
    table: dict[str, SearchPoint], grid: Collection[SearchPoint]
) -> None:
    stray = _winners_the_grid_does_not_contain(table, grid)
    if stray:
        raise ValueError(
            f"{stray} name configurations this search never ran. Every entry "
            "of this table was chosen by the sweep, so it has to be one of "
            "the grid's points -- an entry outside it means the axes moved "
            "and the winners were never re-chosen."
        )


_refuse_winners_the_grid_does_not_contain(EXOGENOUS_TABLE, EXOGENOUS_GRID)


def exogenous_configs() -> list[dict[str, Any]]:
    return _configs_at(lambda name: (EXOGENOUS_TABLE[name],))
