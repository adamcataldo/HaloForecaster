from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from halo import (
    gaussian_head,
    multi_head_timexer,
    multi_head_timexer_configs,
    multi_head_timexer_sweep_configs,
    scheduling,
    settings,
    side_by_side,
    side_by_side_configs,
)
from halo.multi_head_timexer_configs import GridPoint

MODEL_NAME = "TimeXer_sbs"

CONTROL = multi_head_timexer_sweep_configs.MODEL_NAME

SEED_IDENTITY = multi_head_timexer_configs.SEED_IDENTITY

OUTPUT_HEADS = gaussian_head.OUTPUT_HEADS

STAR_SIZE = 5


build = side_by_side.builder(
    multi_head_timexer.builder(),
    output_heads=OUTPUT_HEADS,
    constraining_transformation=gaussian_head.scale_in_target_units,
    point_estimate=gaussian_head.location,
)


EXOGENOUS_PARALLELISM: dict[str, int] = {
    "NP": 8,
    "PJM": 8,
    "BE": 8,
    "FR": 8,
    "DE": 8,
}


side_by_side_configs.refuse_unnumbered_markets(
    EXOGENOUS_PARALLELISM, settings.SHORT_TERM_EXOGENOUS.datasets
)


EXOGENOUS_ANCHORS: dict[str, GridPoint] = dict(
    multi_head_timexer_sweep_configs.EXOGENOUS_TABLE
)


def star(anchor: GridPoint) -> tuple[GridPoint, ...]:
    return tuple(
        point
        for point in multi_head_timexer_sweep_configs.neighbourhood(anchor)
        if point.batch_size == anchor.batch_size
    )


EXOGENOUS_GRIDS: dict[str, tuple[GridPoint, ...]] = {
    name: star(anchor) for name, anchor in EXOGENOUS_ANCHORS.items()
}


def _stars_that_are_not_the_stated_size(
    grids: dict[str, tuple[GridPoint, ...]],
) -> dict[str, int]:
    return {
        name: len(points) for name, points in grids.items() if len(points) != STAR_SIZE
    }


def _refuse_stars_that_are_not_the_stated_size(
    grids: dict[str, tuple[GridPoint, ...]],
) -> None:
    wrong = _stars_that_are_not_the_stated_size(grids)
    if wrong:
        raise ValueError(
            f"{wrong} count points where every star here carries {STAR_SIZE}: "
            "the anchor, the layer count one step either way, and the "
            "feed-forward width halved and doubled. The batch size is held at "
            "the anchor's own, so it contributes no point, and no anchor this "
            "search is centred on sits on a floor, so no downward step is "
            "dropped. A star of another size means the anchors moved onto a "
            "floor, or an axis was added or taken away upstream, and the trial "
            "count this search is budgeted at no longer holds."
        )


_refuse_stars_that_are_not_the_stated_size(EXOGENOUS_GRIDS)


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


side_by_side_configs.refuse_trials_that_depart_from_the_control(
    MODEL_NAME,
    CONTROL,
    exogenous_search_space(),
    multi_head_timexer_sweep_configs.exogenous_search_space(),
)


side_by_side_configs.refuse_a_space_that_omits_the_controls_published_configuration(
    MODEL_NAME,
    CONTROL,
    exogenous_search_space(),
    multi_head_timexer_sweep_configs.exogenous_configs(),
    EXOGENOUS_PARALLELISM,
)


EXOGENOUS_TABLE: dict[str, GridPoint] = {
    "NP": GridPoint(e_layers=2, d_ff=512, batch_size=2),
    "PJM": GridPoint(e_layers=2, d_ff=4096, batch_size=16),
    "BE": GridPoint(e_layers=2, d_ff=256, batch_size=16),
    "FR": GridPoint(e_layers=2, d_ff=1024, batch_size=32),
    "DE": GridPoint(e_layers=2, d_ff=4096, batch_size=4),
}


side_by_side_configs.refuse_unsettled_markets(
    EXOGENOUS_TABLE, settings.SHORT_TERM_EXOGENOUS.datasets
)


side_by_side_configs.refuse_winners_the_search_never_ran(
    MODEL_NAME, EXOGENOUS_TABLE, EXOGENOUS_GRIDS
)


def exogenous_configs() -> list[dict[str, Any]]:
    return _configs_at(lambda name: (EXOGENOUS_TABLE[name],))
