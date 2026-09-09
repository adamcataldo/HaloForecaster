from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from halo import (
    gaussian_head,
    multi_head_timexer,
    search,
    seeding,
    timexer_configs,
)

MODEL_NAME = "MultiHeadTimeXer"

UNEARNED_TABLE_FROM = timexer_configs.MODEL_NAME

SEED_IDENTITY = timexer_configs.MODEL_NAME

SCALE = gaussian_head.SCALE

OUTPUT_HEADS = gaussian_head.OUTPUT_HEADS


@dataclass(frozen=True)
class GridPoint:
    e_layers: int
    d_ff: int
    batch_size: int

    @property
    def label(self) -> str:
        return f"L{self.e_layers}-F{self.d_ff}-B{self.batch_size}"

    def as_config(self) -> dict[str, Any]:
        return {
            "e_layers": self.e_layers,
            "d_ff": self.d_ff,
            "batch_size": self.batch_size,
            search.GRID_POINT_KEY: self.label,
        }


build = multi_head_timexer.builder(
    output_heads=OUTPUT_HEADS,
    constraining_transformation=gaussian_head.scale_in_target_units,
    point_estimate=gaussian_head.location,
)


def exogenous_configs() -> list[dict[str, Any]]:
    return [
        {
            **config,
            "model": MODEL_NAME,
            seeding.SEED_IDENTITY_KEY: SEED_IDENTITY,
        }
        for config in timexer_configs.exogenous_configs()
    ]


def _fields_that_differ(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in ours.keys() | theirs.keys() if ours.get(key) != theirs.get(key)
    )


_inherited_from = {
    (config["dataset_name"], config["pred_len"]): config
    for config in timexer_configs.exogenous_configs()
}

_departures = {
    f"{config['dataset_name']}/S={config['pred_len']}": _fields_that_differ(
        config,
        _inherited_from.get((config["dataset_name"], config["pred_len"]), {}),
    )
    for config in exogenous_configs()
}

_unearned = {
    where: differing
    for where, differing in _departures.items()
    if differing != ["model", seeding.SEED_IDENTITY_KEY]
}
if _unearned:
    raise ValueError(
        f"{MODEL_NAME} departs from {UNEARNED_TABLE_FROM} at {_unearned}. This "
        "model has never been searched: every trial behind these five "
        f"configurations chose them for {UNEARNED_TABLE_FROM} under squared "
        "error, and this model inherits them unearned so that the difference "
        "between the two rows is the extra head and the objective and nothing "
        "else. Only the published name and the declared seed identity may "
        "differ."
    )
