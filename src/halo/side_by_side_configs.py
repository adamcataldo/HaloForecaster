from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol

from halo import scheduling, search, settings

FIELDS_A_SPLIT_MAY_MOVE = frozenset({"model", scheduling.PARALLELISM_KEY})


def at_this_model(
    control_configs: Sequence[dict[str, Any]],
    model_name: str,
    parallelism: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            **config,
            "model": model_name,
            scheduling.PARALLELISM_KEY: parallelism[config["dataset_name"]],
        }
        for config in control_configs
    ]


class LabelledPoint(Protocol):
    @property
    def label(self) -> str: ...


def refuse_unnumbered_markets(
    parallelism: dict[str, int], markets: Collection[str]
) -> None:
    if parallelism.keys() != frozenset(markets):
        raise ValueError(
            "every market of the "
            f"{settings.SHORT_TERM_EXOGENOUS.name} setting needs a parallelism "
            f"number, and every number needs a market: {sorted(markets)} are "
            f"searched but {sorted(parallelism)} carry numbers."
        )


def _point_of(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config["dataset_name"],
        config["pred_len"],
        config[search.GRID_POINT_KEY],
    )


def _where(config: dict[str, Any]) -> str:
    market, horizon, label = _point_of(config)
    return f"{market}/S={horizon}/{label}"


def _fields_that_differ(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in ours.keys() | theirs.keys() if ours.get(key) != theirs.get(key)
    )


def _departures_from_the_control(
    trials: Sequence[dict[str, Any]], control_trials: Sequence[dict[str, Any]]
) -> dict[str, list[str]]:
    control_at = {_point_of(trial): trial for trial in control_trials}
    shared = [trial for trial in trials if _point_of(trial) in control_at]
    differing = {
        _where(trial): _fields_that_differ(trial, control_at[_point_of(trial)])
        for trial in shared
    }
    return {
        where: sorted(frozenset(fields) - FIELDS_A_SPLIT_MAY_MOVE)
        for where, fields in differing.items()
        if not frozenset(fields) <= FIELDS_A_SPLIT_MAY_MOVE
    }


def refuse_trials_that_depart_from_the_control(
    model_name: str,
    control_name: str,
    trials: Sequence[dict[str, Any]],
    control_trials: Sequence[dict[str, Any]],
) -> None:
    departures = _departures_from_the_control(trials, control_trials)
    if departures:
        raise ValueError(
            f"{model_name} departs from {control_name} at {departures}. These "
            "two rows are read against each other, and the whole reading is "
            "that one splits the trunk and the other shares it. A trial is "
            "composed from the control's own configuration at the same point "
            "rather than transcribed beside it, so the two can differ in the "
            "published name and the measured parallelism number and in "
            "nothing else -- a third field moving turns the comparison into a "
            "second experiment nobody named."
        )


def _published_configurations_the_space_omits(
    trials: Sequence[dict[str, Any]],
    control_configs: Sequence[dict[str, Any]],
    model_name: str,
    parallelism: dict[str, int],
) -> list[str]:
    wanted = at_this_model(control_configs, model_name, parallelism)
    return sorted(_where(config) for config in wanted if config not in trials)


def refuse_a_space_that_omits_the_controls_published_configuration(
    model_name: str,
    control_name: str,
    trials: Sequence[dict[str, Any]],
    control_configs: Sequence[dict[str, Any]],
    parallelism: dict[str, int],
) -> None:
    missing = _published_configurations_the_space_omits(
        trials, control_configs, model_name, parallelism
    )
    if missing:
        raise ValueError(
            f"{model_name} never runs {missing}, which is where {control_name} "
            "publishes. The number this model is read against is the control's "
            "published row, so the control's configuration has to be a point "
            "this model actually visits -- otherwise the two rows are compared "
            "across a gap neither of them searched, and the split is confused "
            "with the search that chose the shapes."
        )


def refuse_unsettled_markets(
    table: Mapping[str, LabelledPoint], markets: Collection[str]
) -> None:
    if table.keys() != frozenset(markets):
        raise ValueError(
            "every searched market needs a settled configuration, and every "
            f"configuration needs a market: {sorted(markets)} are searched "
            f"but {sorted(table)} carry configurations."
        )


def _winners_the_search_never_ran(
    table: Mapping[str, LabelledPoint],
    searched: Mapping[str, Collection[LabelledPoint]],
) -> dict[str, str]:
    return {
        name: point.label
        for name, point in table.items()
        if point not in searched.get(name, ())
    }


def refuse_winners_the_search_never_ran(
    model_name: str,
    table: Mapping[str, LabelledPoint],
    searched: Mapping[str, Collection[LabelledPoint]],
) -> None:
    stray = _winners_the_search_never_ran(table, searched)
    if stray:
        raise ValueError(
            f"{stray} name configurations {model_name} never ran. Every entry "
            "of this table was chosen by the sweep, so it has to be one of "
            "the points the sweep visited -- an entry outside them means the "
            "space moved and the winners were never re-chosen."
        )
