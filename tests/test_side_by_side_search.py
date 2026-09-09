from __future__ import annotations

import re
from dataclasses import replace

import pytest

from halo import (
    crosslinear_sbs_configs,
    multi_head_linear_configs,
    multi_head_timexer_configs,
    multi_head_timexer_sweep_configs,
    registry,
    scheduling,
    search,
    seeding,
    settings,
    side_by_side,
    side_by_side_configs,
    timexer_sbs_configs,
)

SHORT_TERM = settings.SHORT_TERM_EXOGENOUS

CROSSLINEAR_SBS = crosslinear_sbs_configs.MODEL_NAME
TIMEXER_SBS = timexer_sbs_configs.MODEL_NAME

CROSSLINEAR_SBS_SWEEP_SIZE = 80
CROSSLINEAR_SBS_GRID_SIZE = 16

TIMEXER_SBS_SWEEP_SIZE = 25

SPLIT_MODELS = (crosslinear_sbs_configs, timexer_sbs_configs)

CONTROLS = {
    CROSSLINEAR_SBS: multi_head_linear_configs,
    TIMEXER_SBS: multi_head_timexer_sweep_configs,
}


def _split(module):
    return pytest.param(module, id=module.MODEL_NAME)


EACH_SPLIT_MODEL = pytest.mark.parametrize(
    "split", [_split(module) for module in SPLIT_MODELS]
)


def _trials_by_market(module) -> dict[str, list[dict]]:
    trials: dict[str, list[dict]] = {}
    for trial in module.exogenous_search_space():
        trials.setdefault(trial["dataset_name"], []).append(trial)
    return trials


def _fields_that_differ(ours: dict, theirs: dict) -> set[str]:
    return {
        key for key in ours.keys() | theirs.keys() if ours.get(key) != theirs.get(key)
    }


def _by_point(configs: list[dict]) -> dict[tuple, dict]:
    return {
        (c["dataset_name"], c["pred_len"], search.grid_point_of(c)): c for c in configs
    }


@EACH_SPLIT_MODEL
def test_each_split_model_is_settled_and_keeps_the_space_it_was_settled_from(split):
    entry = registry.entry(SHORT_TERM.name, split.MODEL_NAME)

    assert not entry.is_tunable
    assert entry.has_search_space
    assert entry.trials() == split.exogenous_search_space()
    assert entry.configs() == split.exogenous_configs()
    assert entry.known_configs() == entry.configs()


@EACH_SPLIT_MODEL
def test_every_trial_names_this_model_and_not_the_control_it_is_read_against(split):
    for trial in split.exogenous_search_space():
        assert trial["model"] == split.MODEL_NAME
        assert trial["model"] != split.CONTROL


@EACH_SPLIT_MODEL
def test_every_trial_derives_the_seed_identity_its_control_declares(split):
    control = CONTROLS[split.MODEL_NAME]
    declared = {
        seeding.seed_identity_of(trial) for trial in control.exogenous_search_space()
    }

    assert declared == {split.SEED_IDENTITY}
    for trial in split.exogenous_search_space():
        assert seeding.seed_identity_of(trial) == split.SEED_IDENTITY


@EACH_SPLIT_MODEL
def test_every_trial_is_schedulable_and_can_be_read_back(split):
    for trial in split.exogenous_search_space():
        assert search.grid_point_of(trial)
        assert scheduling.parallelism_of(trial) >= 1


@EACH_SPLIT_MODEL
def test_each_searched_market_runs_at_exactly_one_parallelism_number(split):
    carried: dict[str, set[int]] = {}
    for trial in split.exogenous_search_space():
        carried.setdefault(trial["dataset_name"], set()).add(
            scheduling.parallelism_of(trial)
        )

    assert carried == {
        name: {number} for name, number in split.EXOGENOUS_PARALLELISM.items()
    }


@EACH_SPLIT_MODEL
def test_the_parallelism_mapping_names_every_market_and_nothing_else(split):
    assert split.EXOGENOUS_PARALLELISM.keys() == frozenset(SHORT_TERM.datasets)


@EACH_SPLIT_MODEL
def test_no_market_sweeps_one_grid_point_twice(split):
    for name, trials in _trials_by_market(split).items():
        labels = [search.grid_point_of(trial) for trial in trials]

        assert len(labels) == len(set(labels)), name


@EACH_SPLIT_MODEL
def test_each_markets_trials_stay_contiguous_and_in_the_settings_order(split):
    markets = [trial["dataset_name"] for trial in split.exogenous_search_space()]
    first_seen = list(dict.fromkeys(markets))

    assert first_seen == list(SHORT_TERM.datasets)
    assert markets == sorted(markets, key=first_seen.index)


@EACH_SPLIT_MODEL
def test_every_trial_forecasts_the_one_horizon_this_setting_scores(split):
    horizons = {trial["pred_len"] for trial in split.exogenous_search_space()}

    assert horizons == set(SHORT_TERM.horizons)


@EACH_SPLIT_MODEL
def test_a_trial_differs_from_its_controls_trial_only_in_the_name_and_the_number(split):
    control = CONTROLS[split.MODEL_NAME]
    theirs = _by_point(control.exogenous_search_space())
    ours = _by_point(split.exogenous_search_space())
    shared = ours.keys() & theirs.keys()

    assert shared
    for point in shared:
        assert _fields_that_differ(ours[point], theirs[point]) <= {
            "model",
            scheduling.PARALLELISM_KEY,
        }


@EACH_SPLIT_MODEL
def test_the_space_runs_the_configuration_its_control_publishes(split):
    control = CONTROLS[split.MODEL_NAME]
    ours = _by_point(split.exogenous_search_space())

    published = control.exogenous_configs()
    assert len(published) == len(SHORT_TERM.datasets)

    for config in published:
        point = (
            config["dataset_name"],
            config["pred_len"],
            search.grid_point_of(config),
        )

        assert point in ours
        assert _fields_that_differ(ours[point], config) <= {
            "model",
            scheduling.PARALLELISM_KEY,
        }


def test_the_crosslinear_sweep_is_eighty_trials_sixteen_a_market():
    trials = crosslinear_sbs_configs.exogenous_search_space()
    counts = {
        name: len([t for t in trials if t["dataset_name"] == name])
        for name in SHORT_TERM.datasets
    }

    assert len(trials) == CROSSLINEAR_SBS_SWEEP_SIZE
    assert counts == dict.fromkeys(SHORT_TERM.datasets, CROSSLINEAR_SBS_GRID_SIZE)


def test_the_crosslinear_grid_is_the_one_its_control_owns_rather_than_a_copy():
    assert (
        crosslinear_sbs_configs.EXOGENOUS_GRID
        is multi_head_linear_configs.EXOGENOUS_GRID
    )
    assert len(crosslinear_sbs_configs.EXOGENOUS_GRID) == CROSSLINEAR_SBS_GRID_SIZE


def test_every_market_runs_every_point_of_the_controls_full_factorial():
    for name, trials in _trials_by_market(crosslinear_sbs_configs).items():
        labels = {search.grid_point_of(trial) for trial in trials}

        assert labels == {
            point.label for point in multi_head_linear_configs.EXOGENOUS_GRID
        }, name


def test_the_crosslinear_sweep_holds_the_batch_size_and_learning_rate_still():
    for trial in crosslinear_sbs_configs.exogenous_search_space():
        assert trial["batch_size"] == multi_head_linear_configs.BATCH_SIZE
        assert trial["learning_rate"] == multi_head_linear_configs.LEARNING_RATE


def test_the_timexer_sweep_is_twenty_five_trials_five_a_market():
    trials = timexer_sbs_configs.exogenous_search_space()
    counts = {
        name: len([t for t in trials if t["dataset_name"] == name])
        for name in SHORT_TERM.datasets
    }

    assert len(trials) == TIMEXER_SBS_SWEEP_SIZE
    assert counts == dict.fromkeys(SHORT_TERM.datasets, timexer_sbs_configs.STAR_SIZE)


def test_every_timexer_star_is_centred_on_its_controls_published_winner():
    assert (
        timexer_sbs_configs.EXOGENOUS_ANCHORS
        == multi_head_timexer_sweep_configs.EXOGENOUS_TABLE
    )


def test_every_timexer_star_is_the_anchor_and_one_step_along_each_of_two_axes():
    for name, anchor in timexer_sbs_configs.EXOGENOUS_ANCHORS.items():
        assert set(timexer_sbs_configs.EXOGENOUS_GRIDS[name]) == {
            anchor,
            replace(anchor, e_layers=anchor.e_layers - 1),
            replace(anchor, e_layers=anchor.e_layers + 1),
            replace(anchor, d_ff=anchor.d_ff // 2),
            replace(anchor, d_ff=anchor.d_ff * 2),
        }, name


def test_no_timexer_star_drops_a_point_because_no_anchor_sits_on_a_floor():
    for name, anchor in timexer_sbs_configs.EXOGENOUS_ANCHORS.items():
        assert anchor.e_layers - 1 >= 1, name
        assert anchor.d_ff // 2 >= 1, name
        assert len(timexer_sbs_configs.EXOGENOUS_GRIDS[name]) == (
            timexer_sbs_configs.STAR_SIZE
        ), name


def test_the_timexer_sweep_keeps_each_markets_own_batch_size():
    batches = {
        name: {trial["batch_size"] for trial in trials}
        for name, trials in _trials_by_market(timexer_sbs_configs).items()
    }

    assert batches == {"NP": {2}, "PJM": {16}, "BE": {16}, "FR": {32}, "DE": {4}}
    for name, anchor in timexer_sbs_configs.EXOGENOUS_ANCHORS.items():
        assert batches[name] == {anchor.batch_size}


def test_the_timexer_sweep_holds_the_learning_rate_its_control_holds():
    rates = {
        trial["learning_rate"] for trial in timexer_sbs_configs.exogenous_search_space()
    }

    assert rates == {
        trial["learning_rate"]
        for trial in multi_head_timexer_configs.exogenous_configs()
    }


@EACH_SPLIT_MODEL
def test_a_market_left_without_a_parallelism_number_is_refused_at_import(split):
    numbered = split.EXOGENOUS_PARALLELISM

    with pytest.raises(ValueError, match="needs a parallelism"):
        side_by_side_configs.refuse_unnumbered_markets(
            {name: number for name, number in numbered.items() if name != "DE"},
            SHORT_TERM.datasets,
        )

    with pytest.raises(ValueError, match="needs a parallelism"):
        side_by_side_configs.refuse_unnumbered_markets(
            {**numbered, "Atlantis": 1}, SHORT_TERM.datasets
        )

    side_by_side_configs.refuse_unnumbered_markets(numbered, SHORT_TERM.datasets)


@EACH_SPLIT_MODEL
def test_a_trial_moving_a_third_field_is_refused_at_import(split):
    control = CONTROLS[split.MODEL_NAME]
    theirs = control.exogenous_search_space()
    ours = split.exogenous_search_space()
    shared = _by_point(ours).keys() & _by_point(theirs).keys()

    assert shared
    strayed = [
        {**trial, "learning_rate": trial["learning_rate"] * 10}
        if (trial["dataset_name"], trial["pred_len"], search.grid_point_of(trial))
        in shared
        else trial
        for trial in ours
    ]

    side_by_side_configs.refuse_trials_that_depart_from_the_control(
        split.MODEL_NAME, split.CONTROL, ours, theirs
    )
    with pytest.raises(ValueError, match=re.escape("learning_rate")):
        side_by_side_configs.refuse_trials_that_depart_from_the_control(
            split.MODEL_NAME, split.CONTROL, strayed, theirs
        )


@EACH_SPLIT_MODEL
def test_a_parallelism_number_of_its_own_is_not_a_departure(split):
    control = CONTROLS[split.MODEL_NAME]
    remeasured = [
        {**trial, scheduling.PARALLELISM_KEY: 1}
        for trial in split.exogenous_search_space()
    ]

    side_by_side_configs.refuse_trials_that_depart_from_the_control(
        split.MODEL_NAME, split.CONTROL, remeasured, control.exogenous_search_space()
    )


@EACH_SPLIT_MODEL
def test_a_space_that_never_runs_the_controls_published_point_is_refused(split):
    control = CONTROLS[split.MODEL_NAME]
    published = control.exogenous_configs()
    ours = split.exogenous_search_space()
    at_be = {
        (c["dataset_name"], c["pred_len"], search.grid_point_of(c))
        for c in published
        if c["dataset_name"] == "BE"
    }
    thinned = [
        trial
        for trial in ours
        if (trial["dataset_name"], trial["pred_len"], search.grid_point_of(trial))
        not in at_be
    ]

    side_by_side_configs.refuse_a_space_that_omits_the_controls_published_configuration(
        split.MODEL_NAME,
        split.CONTROL,
        ours,
        published,
        split.EXOGENOUS_PARALLELISM,
    )
    with pytest.raises(ValueError, match="BE"):
        side_by_side_configs.refuse_a_space_that_omits_the_controls_published_configuration(
            split.MODEL_NAME,
            split.CONTROL,
            thinned,
            published,
            split.EXOGENOUS_PARALLELISM,
        )


def test_a_timexer_star_of_another_size_is_refused_at_import():
    grids = timexer_sbs_configs.EXOGENOUS_GRIDS

    timexer_sbs_configs._refuse_stars_that_are_not_the_stated_size(grids)
    with pytest.raises(ValueError, match="FR"):
        timexer_sbs_configs._refuse_stars_that_are_not_the_stated_size(
            {**grids, "FR": grids["FR"][:-1]}
        )


@EACH_SPLIT_MODEL
def test_each_split_model_builds_two_branches_standing_side_by_side(split):
    entry = registry.entry(SHORT_TERM.name, split.MODEL_NAME)
    config = dict(entry.trials()[0])
    config.update(d_model=32, d_ff=64, e_layers=1, n_heads=4, patch_len=24)

    model = entry.build(config)

    assert isinstance(model, side_by_side.SideBySide)
    assert len(model.backbones) == split.OUTPUT_HEADS
    assert model.backbones[0] is not model.backbones[1]


def test_each_table_names_the_winners_its_own_search_chose():
    labelled = {
        module.MODEL_NAME: {
            name: point.label for name, point in module.EXOGENOUS_TABLE.items()
        }
        for module in SPLIT_MODELS
    }

    assert labelled == {
        CROSSLINEAR_SBS: {
            "NP": "P24-D768-F2048-A2",
            "PJM": "P24-D1024-F2048-A2",
            "BE": "P16-D1024-F2048-A1",
            "FR": "P16-D1024-F2048-A2",
            "DE": "P16-D1024-F2048-A1",
        },
        TIMEXER_SBS: {
            "NP": "L2-F512-B2",
            "PJM": "L2-F4096-B16",
            "BE": "L2-F256-B16",
            "FR": "L2-F1024-B32",
            "DE": "L2-F4096-B4",
        },
    }


def test_every_timexer_winner_took_two_encoder_layers():
    for point in timexer_sbs_configs.EXOGENOUS_TABLE.values():
        assert point.e_layers == 2


@EACH_SPLIT_MODEL
def test_each_table_chooses_one_configuration_for_every_market(split):
    configs = split.exogenous_configs()

    assert [c["dataset_name"] for c in configs] == list(SHORT_TERM.datasets)
    assert set(split.EXOGENOUS_TABLE) == set(SHORT_TERM.datasets)
    assert {c["pred_len"] for c in configs} == {24}


@EACH_SPLIT_MODEL
def test_every_winner_is_a_point_the_split_model_actually_ran(split):
    for name, point in split.EXOGENOUS_TABLE.items():
        assert point in split.EXOGENOUS_GRIDS[name]


@EACH_SPLIT_MODEL
def test_every_settled_configuration_is_one_of_the_trials_it_was_chosen_from(split):
    trials = _by_point(split.exogenous_search_space())

    for config in split.exogenous_configs():
        point = (
            config["dataset_name"],
            config["pred_len"],
            search.grid_point_of(config),
        )
        assert trials[point] == config


@EACH_SPLIT_MODEL
def test_each_settled_configuration_names_this_model_and_its_own_parallelism(split):
    for config in split.exogenous_configs():
        assert config["model"] == split.MODEL_NAME
        assert (
            config[scheduling.PARALLELISM_KEY]
            == split.EXOGENOUS_PARALLELISM[config["dataset_name"]]
        )


@EACH_SPLIT_MODEL
def test_a_settled_configuration_departs_from_its_control_only_in_name_and_number(
    split,
):
    control = CONTROLS[split.MODEL_NAME]
    theirs = _by_point(control.exogenous_search_space())
    ours = _by_point(split.exogenous_configs())
    shared = ours.keys() & theirs.keys()

    assert shared
    for point in shared:
        assert _fields_that_differ(ours[point], theirs[point]) <= {
            "model",
            scheduling.PARALLELISM_KEY,
        }


@EACH_SPLIT_MODEL
def test_a_winner_the_search_never_ran_is_refused_at_import(split):
    table = split.EXOGENOUS_TABLE
    strayed = {**table, "BE": replace(table["BE"], d_ff=table["BE"].d_ff * 8)}

    side_by_side_configs.refuse_winners_the_search_never_ran(
        split.MODEL_NAME, table, split.EXOGENOUS_GRIDS
    )
    with pytest.raises(ValueError, match=re.escape(strayed["BE"].label)):
        side_by_side_configs.refuse_winners_the_search_never_ran(
            split.MODEL_NAME, strayed, split.EXOGENOUS_GRIDS
        )


@EACH_SPLIT_MODEL
def test_a_market_left_without_a_settled_configuration_is_refused_at_import(split):
    table = split.EXOGENOUS_TABLE

    with pytest.raises(ValueError, match="carry configurations"):
        side_by_side_configs.refuse_unsettled_markets(
            {name: p for name, p in table.items() if name != "FR"},
            SHORT_TERM.datasets,
        )


@EACH_SPLIT_MODEL
def test_a_settled_configuration_at_a_misspelt_market_is_refused_at_import(split):
    table = split.EXOGENOUS_TABLE
    misspelt = {
        **{name: p for name, p in table.items() if name != "FR"},
        "FRA": table["FR"],
    }

    with pytest.raises(ValueError, match="carry configurations"):
        side_by_side_configs.refuse_unsettled_markets(misspelt, SHORT_TERM.datasets)
