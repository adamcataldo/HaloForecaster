from __future__ import annotations

from typing import Any

from halo import (
    gaussian_head,
    multi_head_linear,
    multi_head_linear_configs,
    search,
    settings,
    side_by_side,
    side_by_side_configs,
)
from halo.multi_head_linear_configs import SearchPoint

MODEL_NAME = "CrossLinear_sbs"

CONTROL = multi_head_linear_configs.MODEL_NAME

SEED_IDENTITY = multi_head_linear_configs.SEED_IDENTITY

OUTPUT_HEADS = gaussian_head.OUTPUT_HEADS


build = side_by_side.builder(
    multi_head_linear.builder(),
    output_heads=OUTPUT_HEADS,
    constraining_transformation=gaussian_head.scale_in_target_units,
    point_estimate=gaussian_head.location,
)


EXOGENOUS_GRID: tuple[SearchPoint, ...] = multi_head_linear_configs.EXOGENOUS_GRID


EXOGENOUS_GRIDS: dict[str, tuple[SearchPoint, ...]] = dict.fromkeys(
    settings.SHORT_TERM_EXOGENOUS.datasets, EXOGENOUS_GRID
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


def exogenous_search_space() -> list[dict[str, Any]]:
    return side_by_side_configs.at_this_model(
        multi_head_linear_configs.exogenous_search_space(),
        MODEL_NAME,
        EXOGENOUS_PARALLELISM,
    )


side_by_side_configs.refuse_trials_that_depart_from_the_control(
    MODEL_NAME,
    CONTROL,
    exogenous_search_space(),
    multi_head_linear_configs.exogenous_search_space(),
)


side_by_side_configs.refuse_a_space_that_omits_the_controls_published_configuration(
    MODEL_NAME,
    CONTROL,
    exogenous_search_space(),
    multi_head_linear_configs.exogenous_configs(),
    EXOGENOUS_PARALLELISM,
)


EXOGENOUS_TABLE: dict[str, SearchPoint] = {
    "NP": SearchPoint(patch_len=24, d_model=768, d_ff=2048, alpha=2.0),
    "PJM": SearchPoint(patch_len=24, d_model=1024, d_ff=2048, alpha=2.0),
    "BE": SearchPoint(patch_len=16, d_model=1024, d_ff=2048, alpha=1.0),
    "FR": SearchPoint(patch_len=16, d_model=1024, d_ff=2048, alpha=2.0),
    "DE": SearchPoint(patch_len=16, d_model=1024, d_ff=2048, alpha=1.0),
}


side_by_side_configs.refuse_unsettled_markets(
    EXOGENOUS_TABLE, settings.SHORT_TERM_EXOGENOUS.datasets
)


side_by_side_configs.refuse_winners_the_search_never_ran(
    MODEL_NAME, EXOGENOUS_TABLE, EXOGENOUS_GRIDS
)


def _the_controls_configurations_at(
    table: dict[str, SearchPoint],
) -> list[dict[str, Any]]:
    chosen = {(name, point.label) for name, point in table.items()}
    return [
        config
        for config in multi_head_linear_configs.exogenous_search_space()
        if (config["dataset_name"], search.grid_point_of(config)) in chosen
    ]


def exogenous_configs() -> list[dict[str, Any]]:
    return side_by_side_configs.at_this_model(
        _the_controls_configurations_at(EXOGENOUS_TABLE),
        MODEL_NAME,
        EXOGENOUS_PARALLELISM,
    )
