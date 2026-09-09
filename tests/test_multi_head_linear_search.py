from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, replace

import pytest

from halo import (
    crosslinear_configs,
    multi_head_linear_configs,
    registry,
    scheduling,
    search,
    seeding,
    settings,
)
from halo.multi_head_linear_configs import SearchPoint

SHORT_TERM = settings.SHORT_TERM_EXOGENOUS
MODEL = multi_head_linear_configs.MODEL_NAME

SEARCHED_FIELDS = (
    "patch_len",
    "d_model",
    "d_ff",
    "alpha",
)

FIXED_FIELDS = (
    "batch_size",
    "learning_rate",
)

GRID_SIZE = 16
SWEEP_SIZE = 80


def _without_the_parallelism_number(config: dict) -> dict:
    return {
        key: value for key, value in config.items() if key != scheduling.PARALLELISM_KEY
    }


def _trials_by_market() -> dict[str, list[dict]]:
    trials: dict[str, list[dict]] = {}
    for trial in multi_head_linear_configs.exogenous_search_space():
        trials.setdefault(trial["dataset_name"], []).append(trial)
    return trials


def _upstream_point(row: tuple) -> SearchPoint:
    patch_len, d_model, d_ff, alpha, _, _, _ = row
    return SearchPoint(
        patch_len=patch_len,
        d_model=d_model,
        d_ff=d_ff,
        alpha=alpha,
    )


def _markets_whose_upstream_optimizer_settings_are_the_fixed_ones() -> set[str]:
    return {
        name
        for name, row in crosslinear_configs.EXOGENOUS_TABLE.items()
        if row[5] == multi_head_linear_configs.BATCH_SIZE
        and row[6] == multi_head_linear_configs.LEARNING_RATE
    }


def test_the_searched_axes_are_the_distinct_values_crosslinears_own_table_takes():
    assert multi_head_linear_configs.PATCH_LENGTHS == (16, 24)
    assert multi_head_linear_configs.LATENT_WIDTHS == (768, 1024)
    assert multi_head_linear_configs.FEED_FORWARD_WIDTHS == (2048, 4096)
    assert multi_head_linear_configs.ALPHAS == (1.0, 2.0)

    table = crosslinear_configs.EXOGENOUS_TABLE.values()
    assert set(multi_head_linear_configs.PATCH_LENGTHS) == {row[0] for row in table}
    assert set(multi_head_linear_configs.LATENT_WIDTHS) == {row[1] for row in table}
    assert set(multi_head_linear_configs.FEED_FORWARD_WIDTHS) == {
        row[2] for row in table
    }
    assert set(multi_head_linear_configs.ALPHAS) == {row[3] for row in table}


def test_the_batch_size_is_the_largest_one_crosslinears_table_uses():
    table = crosslinear_configs.EXOGENOUS_TABLE.values()

    assert multi_head_linear_configs.BATCH_SIZE == 16
    assert max(row[5] for row in table) == multi_head_linear_configs.BATCH_SIZE


def test_the_learning_rate_is_the_one_most_of_crosslinears_markets_use():
    table = crosslinear_configs.EXOGENOUS_TABLE.values()
    counted = Counter(row[6] for row in table)

    assert multi_head_linear_configs.LEARNING_RATE == 1e-3
    assert counted[multi_head_linear_configs.LEARNING_RATE] == max(counted.values())
    assert counted[1e-3] == 4


def test_neither_the_batch_size_nor_the_learning_rate_is_an_axis():
    assert not hasattr(multi_head_linear_configs, "BATCH_SIZES")
    assert not hasattr(multi_head_linear_configs, "LEARNING_RATES")
    assert {field.name for field in SearchPoint.__dataclass_fields__.values()} == set(
        SEARCHED_FIELDS
    )


def test_the_grid_is_the_sixteen_points_of_the_stated_cross_product():
    grid = multi_head_linear_configs.EXOGENOUS_GRID

    assert len(grid) == GRID_SIZE
    assert set(grid) == {
        SearchPoint(
            patch_len=patch_len,
            d_model=d_model,
            d_ff=d_ff,
            alpha=alpha,
        )
        for patch_len in multi_head_linear_configs.PATCH_LENGTHS
        for d_model in multi_head_linear_configs.LATENT_WIDTHS
        for d_ff in multi_head_linear_configs.FEED_FORWARD_WIDTHS
        for alpha in multi_head_linear_configs.ALPHAS
    }


def test_every_market_is_searched_over_the_same_sixteen_points():
    trials = _trials_by_market()

    assert trials.keys() == set(SHORT_TERM.datasets)
    for market_trials in trials.values():
        assert len(market_trials) == GRID_SIZE
        assert {search.grid_point_of(trial) for trial in market_trials} == {
            point.label for point in multi_head_linear_configs.EXOGENOUS_GRID
        }


def test_the_whole_sweep_is_eighty_trials():
    assert len(multi_head_linear_configs.exogenous_search_space()) == SWEEP_SIZE
    assert GRID_SIZE * len(SHORT_TERM.datasets) == SWEEP_SIZE


def test_every_trial_runs_at_the_fixed_batch_size_and_learning_rate():
    for trial in multi_head_linear_configs.exogenous_search_space():
        assert trial["batch_size"] == multi_head_linear_configs.BATCH_SIZE
        assert trial["learning_rate"] == multi_head_linear_configs.LEARNING_RATE


def test_de_is_the_only_market_whose_whole_upstream_configuration_is_a_trial():
    assert _markets_whose_upstream_optimizer_settings_are_the_fixed_ones() == {"DE"}


def test_every_markets_upstream_shape_is_still_one_the_grid_visits():
    for row in crosslinear_configs.EXOGENOUS_TABLE.values():
        assert _upstream_point(row) in multi_head_linear_configs.EXOGENOUS_GRID


def test_a_trial_at_the_one_market_upstream_matches_is_crosslinears_run_but_for_keys():
    published = {c["dataset_name"]: c for c in crosslinear_configs.exogenous_configs()}
    swept = _trials_by_market()

    for name in _markets_whose_upstream_optimizer_settings_are_the_fixed_ones():
        upstream = _upstream_point(crosslinear_configs.EXOGENOUS_TABLE[name])
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


def test_every_trial_carries_crosslinears_constants_and_departs_only_where_it_sweeps():
    published = {c["dataset_name"]: c for c in crosslinear_configs.exogenous_configs()}

    for trial in multi_head_linear_configs.exogenous_search_space():
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
            *FIXED_FIELDS,
        }


def test_every_trial_derives_the_seed_it_shares_with_crosslinear():
    for trial in multi_head_linear_configs.exogenous_search_space():
        assert seeding.seed_identity_of(trial) == crosslinear_configs.MODEL_NAME
        assert trial["model"] == MODEL


def test_every_trial_is_schedulable_and_can_be_read_back():
    for trial in multi_head_linear_configs.exogenous_search_space():
        assert search.grid_point_of(trial)
        assert scheduling.parallelism_of(trial) >= 1


def test_each_searched_market_runs_at_exactly_one_parallelism_number():
    numbered = multi_head_linear_configs.EXOGENOUS_PARALLELISM
    carried: dict[str, set[int]] = {}
    for trial in multi_head_linear_configs.exogenous_search_space():
        carried.setdefault(trial["dataset_name"], set()).add(
            scheduling.parallelism_of(trial)
        )

    assert carried == {name: {number} for name, number in numbered.items()}


def test_the_position_mixing_coefficient_is_shared_rather_than_searched():
    betas = {
        trial["beta"] for trial in multi_head_linear_configs.exogenous_search_space()
    }

    assert betas == {1.0}
    assert {row[4] for row in crosslinear_configs.EXOGENOUS_TABLE.values()} == {1.0}


@pytest.mark.parametrize(
    ("point", "label"),
    [
        (
            SearchPoint(
                patch_len=16,
                d_model=1024,
                d_ff=4096,
                alpha=1.0,
            ),
            "P16-D1024-F4096-A1",
        ),
        (
            SearchPoint(
                patch_len=24,
                d_model=768,
                d_ff=2048,
                alpha=2.0,
            ),
            "P24-D768-F2048-A2",
        ),
    ],
)
def test_a_points_label_is_its_four_searched_values_in_plain_decimal(point, label):
    assert point.label == label
    assert point.as_config()[search.GRID_POINT_KEY] == label


def test_a_label_names_no_field_the_search_holds_fixed():
    for point in multi_head_linear_configs.EXOGENOUS_GRID:
        assert "-B" not in point.label
        assert "-R" not in point.label


def test_a_points_config_round_trips_through_its_own_fields():
    for point in multi_head_linear_configs.EXOGENOUS_GRID:
        carried = point.as_config()

        for field in SEARCHED_FIELDS:
            assert carried[field] == getattr(point, field)
        assert carried["batch_size"] == multi_head_linear_configs.BATCH_SIZE
        assert carried["learning_rate"] == multi_head_linear_configs.LEARNING_RATE
        assert carried[search.GRID_POINT_KEY] == point.label


def test_a_points_label_is_the_same_string_every_time_it_is_asked_for():
    for point in multi_head_linear_configs.EXOGENOUS_GRID:
        assert point.label == point.label
        assert point.label == replace(point).label


@pytest.mark.parametrize("field", SEARCHED_FIELDS)
def test_moving_any_searched_field_moves_the_label(field):
    point = SearchPoint(
        patch_len=16,
        d_model=768,
        d_ff=2048,
        alpha=1.0,
    )
    axis = {
        "patch_len": multi_head_linear_configs.PATCH_LENGTHS,
        "d_model": multi_head_linear_configs.LATENT_WIDTHS,
        "d_ff": multi_head_linear_configs.FEED_FORWARD_WIDTHS,
        "alpha": multi_head_linear_configs.ALPHAS,
    }[field]

    for value in axis:
        moved = SearchPoint(**{**asdict(point), field: value})
        assert (moved.label == point.label) == (value == getattr(point, field))


def test_the_sixteen_labels_are_sixteen_distinct_strings():
    grid = multi_head_linear_configs.EXOGENOUS_GRID

    assert len({point.label for point in grid}) == len(grid)


def test_a_grid_whose_points_render_the_same_way_is_refused():
    collided = (
        SearchPoint(
            patch_len=16,
            d_model=768,
            d_ff=2048,
            alpha=1.0,
        ),
        SearchPoint(
            patch_len=16,
            d_model=768,
            d_ff=2048,
            alpha=1.0 + 1e-30,
        ),
    )

    assert collided[0].label == collided[1].label
    with pytest.raises(ValueError, match="more than one point"):
        multi_head_linear_configs._refuse_colliding_labels(collided)

    multi_head_linear_configs._refuse_colliding_labels(
        multi_head_linear_configs.EXOGENOUS_GRID
    )


def test_a_market_without_a_parallelism_number_is_refused():
    numbered = multi_head_linear_configs.EXOGENOUS_PARALLELISM

    with pytest.raises(ValueError, match="needs a parallelism"):
        multi_head_linear_configs._refuse_unnumbered_markets(
            {name: number for name, number in numbered.items() if name != "DE"},
            SHORT_TERM.datasets,
        )

    with pytest.raises(ValueError, match="needs a parallelism"):
        multi_head_linear_configs._refuse_unnumbered_markets(
            {**numbered, "Atlantis": 1}, SHORT_TERM.datasets
        )

    multi_head_linear_configs._refuse_unnumbered_markets(numbered, SHORT_TERM.datasets)


def test_the_model_is_settled_and_keeps_the_space_it_was_settled_from():
    entry = registry.entry(SHORT_TERM.name, MODEL)

    assert not entry.is_tunable
    assert entry.has_search_space
    assert entry.trials() == multi_head_linear_configs.exogenous_search_space()
    assert entry.configs() == multi_head_linear_configs.exogenous_configs()
    assert entry.known_configs() == entry.configs()


def test_every_trial_forecasts_the_one_horizon_this_setting_scores():
    trials = multi_head_linear_configs.exogenous_search_space()

    assert {trial["pred_len"] for trial in trials} == {24}
    assert [trial["dataset_name"] for trial in trials[::GRID_SIZE]] == list(
        SHORT_TERM.datasets
    )


def test_the_table_names_the_winners_its_own_search_chose():
    chosen = {
        c["dataset_name"]: (c["patch_len"], c["d_model"], c["d_ff"], c["alpha"])
        for c in multi_head_linear_configs.exogenous_configs()
    }
    labelled = {
        name: point.label
        for name, point in multi_head_linear_configs.EXOGENOUS_TABLE.items()
    }

    assert chosen == {
        "NP": (16, 1024, 2048, 2.0),
        "PJM": (16, 1024, 2048, 2.0),
        "BE": (16, 768, 4096, 2.0),
        "FR": (16, 768, 2048, 2.0),
        "DE": (16, 1024, 4096, 1.0),
    }
    assert labelled == {
        "NP": "P16-D1024-F2048-A2",
        "PJM": "P16-D1024-F2048-A2",
        "BE": "P16-D768-F4096-A2",
        "FR": "P16-D768-F2048-A2",
        "DE": "P16-D1024-F4096-A1",
    }


def test_every_winner_took_the_patch_length_that_pads_the_lookback():
    for point in multi_head_linear_configs.EXOGENOUS_TABLE.values():
        assert point.patch_len == 16


def test_the_table_chooses_one_configuration_for_every_market():
    configs = multi_head_linear_configs.exogenous_configs()

    assert [c["dataset_name"] for c in configs] == list(SHORT_TERM.datasets)
    assert set(multi_head_linear_configs.EXOGENOUS_TABLE) == set(SHORT_TERM.datasets)
    assert {c["pred_len"] for c in configs} == {24}


def test_every_winner_is_a_point_of_the_grid_it_was_chosen_from():
    grid = multi_head_linear_configs.EXOGENOUS_GRID

    for point in multi_head_linear_configs.EXOGENOUS_TABLE.values():
        assert point in grid


def test_every_settled_configuration_runs_at_the_fixed_batch_size_and_rate():
    for config in multi_head_linear_configs.exogenous_configs():
        assert config["batch_size"] == multi_head_linear_configs.BATCH_SIZE
        assert config["learning_rate"] == multi_head_linear_configs.LEARNING_RATE


def test_a_winner_the_grid_does_not_contain_is_refused_at_import():
    table = multi_head_linear_configs.EXOGENOUS_TABLE
    strayed = {
        **table,
        "BE": replace(table["BE"], d_model=table["BE"].d_model * 4),
    }

    assert strayed["BE"].label == "P16-D3072-F4096-A2"
    with pytest.raises(ValueError, match=re.escape("P16-D3072-F4096-A2")):
        multi_head_linear_configs._refuse_winners_the_grid_does_not_contain(
            strayed, multi_head_linear_configs.EXOGENOUS_GRID
        )


def test_a_market_left_without_a_settled_configuration_is_refused_at_import():
    settled = multi_head_linear_configs.EXOGENOUS_TABLE

    with pytest.raises(ValueError, match="carry configurations"):
        multi_head_linear_configs._refuse_unsettled_markets(
            {name: p for name, p in settled.items() if name != "FR"},
            SHORT_TERM.datasets,
        )


def test_a_settled_configuration_at_a_misspelt_market_is_refused_at_import():
    settled = multi_head_linear_configs.EXOGENOUS_TABLE
    misspelt = {
        **{name: p for name, p in settled.items() if name != "FR"},
        "FRA": settled["FR"],
    }

    with pytest.raises(ValueError, match="carry configurations"):
        multi_head_linear_configs._refuse_unsettled_markets(
            misspelt, SHORT_TERM.datasets
        )


def test_a_winner_keeps_the_number_its_market_was_swept_at():
    swept = multi_head_linear_configs.EXOGENOUS_PARALLELISM

    for config in multi_head_linear_configs.exogenous_configs():
        assert scheduling.parallelism_of(config) == swept[config["dataset_name"]]


def test_every_chosen_configuration_is_byte_for_byte_a_trial_it_swept():
    won = multi_head_linear_configs.EXOGENOUS_TABLE
    swept = {
        (t["dataset_name"], search.grid_point_of(t)): t
        for t in multi_head_linear_configs.exogenous_search_space()
    }

    for config in multi_head_linear_configs.exogenous_configs():
        name = config["dataset_name"]
        key = (name, search.grid_point_of(config))

        assert search.grid_point_of(config) == won[name].label
        assert key in swept
        assert config == swept[key]


def test_every_settled_configuration_derives_the_seed_it_shares_with_crosslinear():
    for config in multi_head_linear_configs.exogenous_configs():
        assert seeding.seed_identity_of(config) == crosslinear_configs.MODEL_NAME
