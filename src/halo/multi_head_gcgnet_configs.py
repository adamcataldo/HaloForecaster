from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from halo import (
    gaussian_head,
    gcgnet_configs,
    laplace_nll,
    multi_head_gcgnet,
    objectives,
    scheduling,
    search,
    seeding,
    settings,
)

MODEL_NAME = "MultiHeadGCGNet"

SEED_IDENTITY = gcgnet_configs.MODEL_NAME

SCALE = gaussian_head.SCALE

OUTPUT_HEADS = gaussian_head.OUTPUT_HEADS

DEVICE = torch.device("cpu")


build = multi_head_gcgnet.builder(
    output_heads=OUTPUT_HEADS,
    constraining_transformation=gaussian_head.scale_in_target_units,
    point_estimate=gaussian_head.location,
)


def laplace_nll_objective(
    model: torch.nn.Module,
    batch: tuple[Tensor, ...],
    config: dict[str, Any],
    device: torch.device,
) -> Tensor:
    prepared = objectives.prepare(batch, config, device)
    forecast, auxiliary = cast(
        multi_head_gcgnet.MultiHeadGCGNet, model
    ).predict_with_auxiliary(prepared.x_enc, prepared.future)
    scale = forecast.parameters[..., SCALE : SCALE + 1]
    likelihood = laplace_nll.laplace_nll_loss(
        forecast.point_estimate, scale, prepared.target
    )
    return likelihood + auxiliary


@dataclass(frozen=True)
class SearchPoint:
    d_model: int
    d_ff: int
    learning_rate: float

    @property
    def label(self) -> str:
        return f"D{self.d_model}-F{self.d_ff}-R{self.learning_rate:g}"

    def as_config(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "learning_rate": self.learning_rate,
            search.GRID_POINT_KEY: self.label,
        }


LATENT_WIDTHS: tuple[int, ...] = tuple(
    sorted({row[0] for row in gcgnet_configs.EXOGENOUS_TABLE.values()})
)

FEED_FORWARD_WIDTHS: tuple[int, ...] = tuple(
    sorted({row[1] for row in gcgnet_configs.EXOGENOUS_TABLE.values()})
)

LEARNING_RATES: tuple[float, ...] = tuple(
    sorted({row[3] for row in gcgnet_configs.EXOGENOUS_TABLE.values()})
)

EXOGENOUS_GRID: tuple[SearchPoint, ...] = tuple(
    SearchPoint(d_model=d_model, d_ff=d_ff, learning_rate=learning_rate)
    for d_model in LATENT_WIDTHS
    for d_ff in FEED_FORWARD_WIDTHS
    for learning_rate in LEARNING_RATES
)


def _refuse_colliding_labels(grid: Sequence[SearchPoint]) -> None:
    labels = [point.label for point in grid]
    if len(set(labels)) != len(labels):
        collided = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(
            f"{collided} are the label of more than one point in the grid. A "
            "trial is stored under its label, so two points sharing one are "
            "recorded as one point measured twice -- half the search would go "
            "missing from the winner check rather than fail. The learning rate "
            "is the field to look at: a formatter that renders two rates the "
            "same way collapses them here."
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
    for inherited in gcgnet_configs.exogenous_configs():
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
    "NP": SearchPoint(d_model=64, d_ff=512, learning_rate=1e-3),
    "PJM": SearchPoint(d_model=64, d_ff=256, learning_rate=1e-3),
    "BE": SearchPoint(d_model=256, d_ff=256, learning_rate=1e-3),
    "FR": SearchPoint(d_model=64, d_ff=512, learning_rate=1e-3),
    "DE": SearchPoint(d_model=512, d_ff=512, learning_rate=1e-4),
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
