from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import replace
from typing import Any

from halo import (
    multi_head_timexer_configs,
    scheduling,
    settings,
    timexer_configs,
)
from halo.multi_head_timexer_configs import GridPoint

MODEL_NAME = "MultiHeadTimeXer_sweep"


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


EXOGENOUS_ANCHORS: dict[str, GridPoint] = {
    name: GridPoint(e_layers=e_layers, d_ff=d_ff, batch_size=batch_size)
    for name, (
        e_layers,
        _,
        d_ff,
        batch_size,
    ) in timexer_configs.EXOGENOUS_TABLE.items()
}


def _steps_down_from(anchor: GridPoint) -> tuple[GridPoint, ...]:
    return (
        replace(anchor, e_layers=anchor.e_layers - 1),
        replace(anchor, d_ff=anchor.d_ff // 2),
        replace(anchor, batch_size=anchor.batch_size // 2),
    )


def _steps_up_from(anchor: GridPoint) -> tuple[GridPoint, ...]:
    return (
        replace(anchor, e_layers=anchor.e_layers + 1),
        replace(anchor, d_ff=anchor.d_ff * 2),
        replace(anchor, batch_size=anchor.batch_size * 2),
    )


def _steps_from(anchor: GridPoint) -> tuple[GridPoint, ...]:
    along_each_axis = zip(_steps_down_from(anchor), _steps_up_from(anchor), strict=True)
    return (anchor, *(point for axis in along_each_axis for point in axis))


def _sits_on_an_axis_floor(point: GridPoint) -> bool:
    return min(point.e_layers, point.d_ff, point.batch_size) < 1


def neighbourhood(anchor: GridPoint) -> tuple[GridPoint, ...]:
    return tuple(
        point for point in _steps_from(anchor) if not _sits_on_an_axis_floor(point)
    )


EXOGENOUS_GRIDS: dict[str, tuple[GridPoint, ...]] = {
    name: neighbourhood(anchor) for name, anchor in EXOGENOUS_ANCHORS.items()
}


def _steps_every_star_must_carry(anchor: GridPoint) -> tuple[GridPoint, ...]:
    return (
        anchor,
        *_steps_up_from(anchor),
        *(
            point
            for point in _steps_down_from(anchor)
            if not _sits_on_an_axis_floor(point)
        ),
    )


def _stars_missing_a_step_no_axis_floor_forbids(
    anchors: dict[str, GridPoint], grids: dict[str, tuple[GridPoint, ...]]
) -> dict[str, list[str]]:
    missing = {
        name: sorted(
            point.label
            for point in _steps_every_star_must_carry(anchor)
            if point not in grids[name]
        )
        for name, anchor in anchors.items()
    }
    return {name: labels for name, labels in missing.items() if labels}


def _refuse_stars_missing_a_step_no_axis_floor_forbids(
    anchors: dict[str, GridPoint], grids: dict[str, tuple[GridPoint, ...]]
) -> None:
    missing = _stars_missing_a_step_no_axis_floor_forbids(anchors, grids)
    if missing:
        raise ValueError(
            f"{missing} are absent from the star at a market whose anchor "
            "sits clear of that axis's floor. A point is dropped only where "
            "the downward step is arithmetically impossible -- an anchor "
            "already at the smallest layer count, width or batch size the "
            "architecture can be built at, and then only along the axis that "
            "sits on its floor. Anywhere else a dropped point means the anchor "
            "moved and the star was never rebuilt."
        )


_refuse_stars_missing_a_step_no_axis_floor_forbids(EXOGENOUS_ANCHORS, EXOGENOUS_GRIDS)


def _configs_at(
    points_for: Callable[[str], Sequence[GridPoint]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for inherited in multi_head_timexer_configs.exogenous_configs():
        name = inherited["dataset_name"]
        for point in points_for(name):
            out.append(
                {
                    **inherited,
                    "model": MODEL_NAME,
                    scheduling.PARALLELISM_KEY: EXOGENOUS_PARALLELISM[name],
                    **point.as_config(),
                }
            )
    return out


def exogenous_search_space() -> list[dict[str, Any]]:
    return _configs_at(lambda name: EXOGENOUS_GRIDS[name])


EXOGENOUS_TABLE: dict[str, GridPoint] = {
    "NP": GridPoint(e_layers=3, d_ff=512, batch_size=2),
    "PJM": GridPoint(e_layers=2, d_ff=2048, batch_size=16),
    "BE": GridPoint(e_layers=2, d_ff=256, batch_size=16),
    "FR": GridPoint(e_layers=2, d_ff=2048, batch_size=32),
    "DE": GridPoint(e_layers=2, d_ff=2048, batch_size=4),
}


def _refuse_unsettled_markets(
    table: dict[str, GridPoint], markets: Collection[str]
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


def _winners_no_star_contains(
    table: dict[str, GridPoint], grids: dict[str, tuple[GridPoint, ...]]
) -> dict[str, str]:
    return {
        name: point.label
        for name, point in table.items()
        if point not in grids.get(name, ())
    }


def _refuse_winners_no_star_contains(
    table: dict[str, GridPoint], grids: dict[str, tuple[GridPoint, ...]]
) -> None:
    stray = _winners_no_star_contains(table, grids)
    if stray:
        raise ValueError(
            f"{stray} name configurations no star in this search contains. "
            "Every entry of this table was chosen by the sweep, so it has to "
            "be one of the points the sweep ran -- an entry outside the star "
            "means the anchors moved and the winners were never re-chosen."
        )


_refuse_winners_no_star_contains(EXOGENOUS_TABLE, EXOGENOUS_GRIDS)


def exogenous_configs() -> list[dict[str, Any]]:
    return _configs_at(lambda name: (EXOGENOUS_TABLE[name],))
