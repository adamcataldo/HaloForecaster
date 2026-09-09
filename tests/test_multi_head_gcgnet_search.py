from __future__ import annotations

import re
from dataclasses import replace

import pytest

from halo import (
    gcgnet_configs,
    multi_head_gcgnet_configs,
    registry,
    scheduling,
    search,
    seeding,
    settings,
)
from halo.multi_head_gcgnet_configs import SearchPoint

SHORT_TERM = settings.SHORT_TERM_EXOGENOUS
MODEL = multi_head_gcgnet_configs.MODEL_NAME

SEARCHED_FIELDS = ("d_model", "d_ff", "learning_rate")


def _without_the_parallelism_number(config: dict) -> dict:
    return {
        key: value for key, value in config.items() if key != scheduling.PARALLELISM_KEY
    }


def _trials_by_market() -> dict[str, list[dict]]:
    trials: dict[str, list[dict]] = {}
    for trial in multi_head_gcgnet_configs.exogenous_search_space():
        trials.setdefault(trial["dataset_name"], []).append(trial)
    return trials


def test_the_axes_are_the_distinct_values_gcgnets_own_table_takes():
    assert multi_head_gcgnet_configs.LATENT_WIDTHS == (64, 256, 512)
    assert multi_head_gcgnet_configs.FEED_FORWARD_WIDTHS == (64, 128, 256, 512)
    assert multi_head_gcgnet_configs.LEARNING_RATES == (1e-4, 1e-3)

    table = gcgnet_configs.EXOGENOUS_TABLE.values()
    assert set(multi_head_gcgnet_configs.LATENT_WIDTHS) == {row[0] for row in table}
    assert set(multi_head_gcgnet_configs.FEED_FORWARD_WIDTHS) == {
        row[1] for row in table
    }
    assert set(multi_head_gcgnet_configs.LEARNING_RATES) == {row[3] for row in table}


def test_the_grid_is_the_twenty_four_points_of_the_stated_cross_product():
    grid = multi_head_gcgnet_configs.EXOGENOUS_GRID

    assert len(grid) == 24
    assert set(grid) == {
        SearchPoint(d_model=d_model, d_ff=d_ff, learning_rate=learning_rate)
        for d_model in multi_head_gcgnet_configs.LATENT_WIDTHS
        for d_ff in multi_head_gcgnet_configs.FEED_FORWARD_WIDTHS
        for learning_rate in multi_head_gcgnet_configs.LEARNING_RATES
    }


def test_every_market_is_searched_over_the_same_twenty_four_points():
    trials = _trials_by_market()

    assert trials.keys() == set(SHORT_TERM.datasets)
    for market_trials in trials.values():
        assert len(market_trials) == 24
        assert {search.grid_point_of(trial) for trial in market_trials} == {
            point.label for point in multi_head_gcgnet_configs.EXOGENOUS_GRID
        }


def test_the_whole_sweep_is_a_hundred_and_twenty_trials():
    assert len(multi_head_gcgnet_configs.exogenous_search_space()) == 120


def test_the_upstream_configuration_of_every_market_falls_inside_the_grid():
    for d_model, d_ff, _, learning_rate in gcgnet_configs.EXOGENOUS_TABLE.values():
        assert (
            SearchPoint(d_model=d_model, d_ff=d_ff, learning_rate=learning_rate)
            in multi_head_gcgnet_configs.EXOGENOUS_GRID
        )


def test_a_trial_at_a_markets_upstream_point_is_gcgnets_run_but_for_three_fields():
    published = {c["dataset_name"]: c for c in gcgnet_configs.exogenous_configs()}
    swept = _trials_by_market()

    for name, (
        d_model,
        d_ff,
        _,
        learning_rate,
    ) in gcgnet_configs.EXOGENOUS_TABLE.items():
        upstream = SearchPoint(d_model=d_model, d_ff=d_ff, learning_rate=learning_rate)
        at_upstream = [
            t for t in swept[name] if search.grid_point_of(t) == upstream.label
        ]
        assert len(at_upstream) == 1

        ours = _without_the_parallelism_number(at_upstream[0])
        theirs = _without_the_parallelism_number(published[name])
        differing = sorted(
            key
            for key in ours.keys() | theirs.keys()
            if ours.get(key) != theirs.get(key)
        )

        assert differing == sorted(
            [search.GRID_POINT_KEY, "model", seeding.SEED_IDENTITY_KEY]
        )


def test_every_trial_carries_gcgnets_constants_and_departs_only_where_it_searches():
    published = {c["dataset_name"]: c for c in gcgnet_configs.exogenous_configs()}

    for trial in multi_head_gcgnet_configs.exogenous_search_space():
        ours = _without_the_parallelism_number(trial)
        theirs = _without_the_parallelism_number(published[trial["dataset_name"]])
        differing = {
            key
            for key in ours.keys() | theirs.keys()
            if ours.get(key) != theirs.get(key)
        }

        assert differing <= {
            search.GRID_POINT_KEY,
            "model",
            seeding.SEED_IDENTITY_KEY,
            *SEARCHED_FIELDS,
        }


def test_every_trial_derives_the_seed_it_shares_with_gcgnet():
    for trial in multi_head_gcgnet_configs.exogenous_search_space():
        assert seeding.seed_identity_of(trial) == gcgnet_configs.MODEL_NAME
        assert trial["model"] == MODEL


def test_every_trial_is_schedulable_and_can_be_read_back():
    for trial in multi_head_gcgnet_configs.exogenous_search_space():
        assert search.grid_point_of(trial)
        assert scheduling.parallelism_of(trial) >= 1


def test_each_searched_market_runs_at_exactly_one_parallelism_number():
    numbered = multi_head_gcgnet_configs.EXOGENOUS_PARALLELISM
    carried: dict[str, set[int]] = {}
    for trial in multi_head_gcgnet_configs.exogenous_search_space():
        carried.setdefault(trial["dataset_name"], set()).add(
            scheduling.parallelism_of(trial)
        )

    assert carried == {name: {number} for name, number in numbered.items()}


def test_the_batch_size_is_shared_rather_than_searched():
    sizes = {
        trial["batch_size"]
        for trial in multi_head_gcgnet_configs.exogenous_search_space()
    }

    assert sizes == {32}


@pytest.mark.parametrize(
    ("point", "label"),
    [
        (SearchPoint(d_model=512, d_ff=256, learning_rate=1e-4), "D512-F256-R0.0001"),
        (SearchPoint(d_model=64, d_ff=64, learning_rate=1e-3), "D64-F64-R0.001"),
    ],
)
def test_a_points_label_is_its_widths_and_its_rate_in_plain_decimal(point, label):
    assert point.label == label
    assert point.as_config()[search.GRID_POINT_KEY] == label


def test_a_points_config_round_trips_through_its_own_fields():
    for point in multi_head_gcgnet_configs.EXOGENOUS_GRID:
        carried = point.as_config()

        assert carried["d_model"] == point.d_model
        assert carried["d_ff"] == point.d_ff
        assert carried["learning_rate"] == point.learning_rate
        assert carried[search.GRID_POINT_KEY] == point.label


def test_a_shapes_two_learning_rates_are_two_labels_and_not_one():
    slow, fast = multi_head_gcgnet_configs.LEARNING_RATES
    shape = SearchPoint(d_model=256, d_ff=128, learning_rate=slow)

    assert shape.label != replace(shape, learning_rate=fast).label
    assert len(
        {point.label for point in multi_head_gcgnet_configs.EXOGENOUS_GRID}
    ) == len(multi_head_gcgnet_configs.EXOGENOUS_GRID)


def test_a_grid_whose_rates_render_the_same_way_is_refused():
    collided = (
        SearchPoint(d_model=64, d_ff=64, learning_rate=1e-4),
        SearchPoint(d_model=64, d_ff=64, learning_rate=1e-4 + 1e-30),
    )

    assert collided[0].label == collided[1].label
    with pytest.raises(ValueError, match="more than one point"):
        multi_head_gcgnet_configs._refuse_colliding_labels(collided)

    multi_head_gcgnet_configs._refuse_colliding_labels(
        multi_head_gcgnet_configs.EXOGENOUS_GRID
    )


def test_a_market_without_a_parallelism_number_is_refused():
    numbered = multi_head_gcgnet_configs.EXOGENOUS_PARALLELISM

    with pytest.raises(ValueError, match="needs a parallelism"):
        multi_head_gcgnet_configs._refuse_unnumbered_markets(
            {name: number for name, number in numbered.items() if name != "DE"},
            SHORT_TERM.datasets,
        )

    with pytest.raises(ValueError, match="needs a parallelism"):
        multi_head_gcgnet_configs._refuse_unnumbered_markets(
            {**numbered, "Atlantis": 1}, SHORT_TERM.datasets
        )

    multi_head_gcgnet_configs._refuse_unnumbered_markets(numbered, SHORT_TERM.datasets)


def test_the_model_is_settled_and_keeps_the_space_it_was_settled_from():
    entry = registry.entry(SHORT_TERM.name, MODEL)

    assert not entry.is_tunable
    assert entry.has_search_space
    assert entry.trials() == multi_head_gcgnet_configs.exogenous_search_space()
    assert entry.configs() == multi_head_gcgnet_configs.exogenous_configs()
    assert entry.known_configs() == entry.configs()


def test_the_table_names_the_winners_its_own_search_chose():
    chosen = {
        c["dataset_name"]: (c["d_model"], c["d_ff"], c["learning_rate"])
        for c in multi_head_gcgnet_configs.exogenous_configs()
    }
    labelled = {
        name: point.label
        for name, point in multi_head_gcgnet_configs.EXOGENOUS_TABLE.items()
    }

    assert chosen == {
        "NP": (64, 512, 1e-3),
        "PJM": (64, 256, 1e-3),
        "BE": (256, 256, 1e-3),
        "FR": (64, 512, 1e-3),
        "DE": (512, 512, 1e-4),
    }
    assert labelled == {
        "NP": "D64-F512-R0.001",
        "PJM": "D64-F256-R0.001",
        "BE": "D256-F256-R0.001",
        "FR": "D64-F512-R0.001",
        "DE": "D512-F512-R0.0001",
    }


def test_the_table_chooses_one_configuration_for_every_market():
    configs = multi_head_gcgnet_configs.exogenous_configs()

    assert [c["dataset_name"] for c in configs] == list(SHORT_TERM.datasets)
    assert set(multi_head_gcgnet_configs.EXOGENOUS_TABLE) == set(SHORT_TERM.datasets)
    assert {c["pred_len"] for c in configs} == {24}


def test_every_winner_is_a_point_of_the_grid_it_was_chosen_from():
    grid = multi_head_gcgnet_configs.EXOGENOUS_GRID

    for point in multi_head_gcgnet_configs.EXOGENOUS_TABLE.values():
        assert point in grid


def test_a_winner_the_grid_does_not_contain_is_refused_at_import():
    table = multi_head_gcgnet_configs.EXOGENOUS_TABLE
    strayed = {
        **table,
        "BE": replace(table["BE"], d_model=table["BE"].d_model * 4),
    }

    assert strayed["BE"].label == "D1024-F256-R0.001"
    with pytest.raises(ValueError, match=re.escape("D1024-F256-R0.001")):
        multi_head_gcgnet_configs._refuse_winners_the_grid_does_not_contain(
            strayed, multi_head_gcgnet_configs.EXOGENOUS_GRID
        )


def test_a_market_left_without_a_settled_configuration_is_refused_at_import():
    settled = multi_head_gcgnet_configs.EXOGENOUS_TABLE

    with pytest.raises(ValueError, match="carry configurations"):
        multi_head_gcgnet_configs._refuse_unsettled_markets(
            {name: p for name, p in settled.items() if name != "FR"},
            SHORT_TERM.datasets,
        )


def test_a_settled_configuration_at_a_misspelt_market_is_refused_at_import():
    settled = multi_head_gcgnet_configs.EXOGENOUS_TABLE
    misspelt = {
        **{name: p for name, p in settled.items() if name != "FR"},
        "FRA": settled["FR"],
    }

    with pytest.raises(ValueError, match="carry configurations"):
        multi_head_gcgnet_configs._refuse_unsettled_markets(
            misspelt, SHORT_TERM.datasets
        )


def test_a_winner_keeps_the_number_its_market_was_swept_at():
    swept = multi_head_gcgnet_configs.EXOGENOUS_PARALLELISM

    for config in multi_head_gcgnet_configs.exogenous_configs():
        assert scheduling.parallelism_of(config) == swept[config["dataset_name"]]


def test_every_chosen_configuration_is_byte_for_byte_a_trial_it_swept():
    won = multi_head_gcgnet_configs.EXOGENOUS_TABLE
    swept = {
        (t["dataset_name"], search.grid_point_of(t)): t
        for t in multi_head_gcgnet_configs.exogenous_search_space()
    }

    for config in multi_head_gcgnet_configs.exogenous_configs():
        name = config["dataset_name"]
        key = (name, search.grid_point_of(config))

        assert search.grid_point_of(config) == won[name].label
        assert key in swept
        assert config == swept[key]


def test_the_setting_knows_every_model_the_registry_carries_for_it():
    assert registry.models_for(SHORT_TERM.name) == (
        "CrossLinear",
        "CrossLinear_sbs",
        "GCGNet",
        "MultiHeadGCGNet",
        "MultiHeadLinear",
        "MultiHeadTimeXer",
        "MultiHeadTimeXer_sweep",
        "TimeXer",
        "TimeXer_sbs",
    )
