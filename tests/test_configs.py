from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from halo import (
    cli,
    data,
    errors,
    gcgnet_configs,
    multi_head_gcgnet_configs,
    multi_head_timexer_configs,
    multi_head_timexer_sweep_configs,
    registry,
    scheduling,
    search,
    seeding,
    settings,
    timexer_configs,
)

SHORT_TERM = settings.SHORT_TERM_EXOGENOUS

SWEPT = multi_head_timexer_sweep_configs.MODEL_NAME


def test_a_dataset_the_setting_does_not_cover_is_refused_by_name():
    with pytest.raises(errors.ConfigurationError, match="short_term_exogenous"):
        cli._datasets_of(SHORT_TERM, ["Atlantis"])


def test_a_horizon_the_setting_does_not_use_is_refused_by_name():
    with pytest.raises(errors.ConfigurationError, match="short_term_exogenous"):
        cli._horizons_of(SHORT_TERM, [96])


def test_omitting_the_selection_takes_the_whole_setting():
    assert cli._datasets_of(SHORT_TERM, None) == list(SHORT_TERM.datasets)
    assert cli._horizons_of(SHORT_TERM, None) == [24]


def test_the_cli_refuses_a_foreign_dataset_without_a_traceback():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "run",
                "--model",
                "TimeXer",
                "--setting",
                "short_term_exogenous",
                "--datasets",
                "Atlantis",
            ]
        )

    assert "short_term_exogenous" in str(exit_info.value)


def test_every_short_term_config_covers_one_market_at_one_horizon():
    configs = timexer_configs.exogenous_configs()

    assert [c["dataset_name"] for c in configs] == list(SHORT_TERM.datasets)
    assert {c["pred_len"] for c in configs} == {24}
    assert {c["setting"] for c in configs} == {SHORT_TERM.name}


def test_every_short_term_config_carries_a_parallelism_number():
    for config in timexer_configs.exogenous_configs():
        assert scheduling.parallelism_of(config) >= 1


def test_every_short_term_config_declares_three_inputs_against_one_output():
    for config in timexer_configs.exogenous_configs():
        assert (config["enc_in"], config["dec_in"], config["c_out"]) == (3, 3, 1)
        assert config["features"] == "MS"


def test_the_short_term_table_looks_back_a_week_in_day_long_patches():
    short_term = timexer_configs.exogenous_configs()[0]

    assert (short_term["seq_len"], short_term["patch_len"]) == (168, 24)
    assert short_term["factor"] == 1


def test_a_model_without_a_table_for_this_setting_is_refused_naming_both():
    with pytest.raises(errors.ConfigurationError, match="iTransformer"):
        registry.entry(SHORT_TERM.name, "iTransformer")

    with pytest.raises(errors.ConfigurationError, match="short_term_exogenous"):
        registry.entry(SHORT_TERM.name, "iTransformer")


def test_every_model_resolves_to_a_config_table_of_one_row_per_market():
    assert len(registry.entry(SHORT_TERM.name, "TimeXer").configs()) == 5
    assert len(registry.entry(SHORT_TERM.name, "GCGNet").configs()) == 5
    assert len(registry.entry(SHORT_TERM.name, "MultiHeadTimeXer").configs()) == 5
    assert len(registry.entry(SHORT_TERM.name, "MultiHeadTimeXer_sweep").configs()) == 5
    assert len(registry.entry(SHORT_TERM.name, "MultiHeadGCGNet").configs()) == 5
    assert len(registry.entry(SHORT_TERM.name, "CrossLinear").configs()) == 5
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


def test_gcgnet_is_settled_and_has_nothing_to_search():
    entry = registry.entry(SHORT_TERM.name, "GCGNet")

    assert not entry.is_tunable
    assert not entry.has_search_space
    with pytest.raises(errors.ConfigurationError, match="nothing to search"):
        entry.trials()


def test_every_gcgnet_config_covers_one_market_and_declines_the_future():
    configs = gcgnet_configs.exogenous_configs()

    assert [c["dataset_name"] for c in configs] == list(SHORT_TERM.datasets)
    assert {c["pred_len"] for c in configs} == {24}
    assert {c[gcgnet_configs.USE_FUTURE_EXOG_KEY] for c in configs} == {False}
    for config in configs:
        assert scheduling.parallelism_of(config) >= 1


def test_the_gcgnet_table_varies_the_widths_and_holds_the_batch_size_at_thirty_two():
    carried = {
        c["dataset_name"]: (
            c["d_model"],
            c["d_ff"],
            c["batch_size"],
            c["learning_rate"],
        )
        for c in gcgnet_configs.exogenous_configs()
    }

    assert carried == {
        "NP": (512, 256, 32, 1e-4),
        "PJM": (512, 512, 32, 1e-3),
        "BE": (256, 512, 32, 1e-3),
        "FR": (64, 128, 32, 1e-3),
        "DE": (64, 64, 32, 1e-3),
    }


def test_gcgnet_trains_for_fifty_epochs_on_the_gentler_decay():
    for config in gcgnet_configs.exogenous_configs():
        assert (config["train_epochs"], config["patience"]) == (50, 5)
        assert config["lradj"] == "type3"


def test_the_gcgnet_parallelism_mapping_names_every_market_and_nothing_else():
    assert set(gcgnet_configs.EXOGENOUS_PARALLELISM) == set(SHORT_TERM.datasets)


def test_a_tunable_model_refuses_to_be_run_as_if_it_were_settled():
    entry = registry.ModelEntry(
        build=lambda config: torch.nn.Module(),
        paper_name="Untuned",
        search_space=list,
    )

    assert entry.is_tunable
    with pytest.raises(errors.ConfigurationError, match="Tune it first"):
        entry.configs()


def test_a_model_without_a_search_space_has_no_trials_to_run():
    entry = registry.entry(SHORT_TERM.name, "TimeXer")

    assert not entry.is_tunable
    with pytest.raises(errors.ConfigurationError, match="no search space"):
        entry.trials()


def test_an_entry_offering_no_configurations_at_all_is_refused():
    with pytest.raises(ValueError, match="settled_table, a search_space, or both"):
        registry.ModelEntry(
            build=lambda config: torch.nn.Module(), paper_name="Nothing"
        )


def test_calibrating_a_tunable_model_refuses_until_a_grid_point_is_named(monkeypatch):
    monkeypatch.setitem(
        registry.MODEL_REGISTRY,
        (SHORT_TERM.name, SWEPT),
        registry.ModelEntry(
            build=multi_head_timexer_configs.build,
            paper_name="TimeXer + dual-head Halo",
            search_space=multi_head_timexer_sweep_configs.exogenous_search_space,
        ),
    )

    with pytest.raises(errors.ConfigurationError, match="L3-F512-B4"):
        cli._config_to_measure(SHORT_TERM, SWEPT, "NP", 24, None)


def test_naming_a_grid_point_for_a_never_searched_model_blames_the_flag():
    with pytest.raises(errors.ConfigurationError, match="never searched"):
        cli._config_to_measure(SHORT_TERM, "TimeXer", "NP", 24, "L1-F512-B4")


def test_naming_the_chosen_grid_point_gives_back_the_configuration_it_runs():
    named = cli._config_to_measure(SHORT_TERM, SWEPT, "NP", 24, "L3-F512-B2")
    unnamed = cli._config_to_measure(SHORT_TERM, SWEPT, "NP", 24, None)

    assert named == unnamed


def test_a_grid_point_outside_what_a_model_runs_is_refused_with_the_choices():
    with pytest.raises(errors.ConfigurationError, match="L3-F512-B4"):
        cli._config_to_measure(SHORT_TERM, SWEPT, "NP", 24, "L9-F512-B4")


def test_a_settled_model_needs_no_grid_point_to_be_calibrated():
    timexer = cli._config_to_measure(SHORT_TERM, "TimeXer", "NP", 24, None)
    gcgnet = cli._config_to_measure(
        SHORT_TERM, multi_head_gcgnet_configs.MODEL_NAME, "NP", 24, None
    )

    assert (timexer["e_layers"], timexer["d_ff"], timexer["batch_size"]) == (3, 512, 4)
    assert (
        gcgnet["e_layers"],
        gcgnet["d_ff"],
        gcgnet["batch_size"],
    ) == (1, 512, 32)


def test_every_epf_market_reads_its_csv_from_the_vendored_timexer_tree():
    assert set(data.DATASETS) == set(SHORT_TERM.datasets)
    for name in SHORT_TERM.datasets:
        spec = data.DATASETS[name]
        assert spec.subdir == "dataset/EPF"
        assert spec.data_path == f"{name}.csv"
        assert spec.n_vars == 3


def test_a_missing_vendored_tree_is_refused_before_a_sweep_starts(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HALO_VENDOR_DIR", str(tmp_path / "absent"))

    with pytest.raises(errors.ConfigurationError, match="get_deps"):
        data.ensure_sources(["NP"])


def test_multi_head_timexer_inherits_timexers_table_field_for_field():
    timexer = {c["dataset_name"]: c for c in timexer_configs.exogenous_configs()}

    for config in multi_head_timexer_configs.exogenous_configs():
        inherited = timexer[config["dataset_name"]]
        differing = sorted(
            key
            for key in config.keys() | inherited.keys()
            if config.get(key) != inherited.get(key)
        )

        assert differing == ["model", seeding.SEED_IDENTITY_KEY]


def test_multi_head_timexer_derives_the_same_seed_per_market_as_timexer():
    timexer = {c["dataset_name"]: c for c in timexer_configs.exogenous_configs()}

    for config in multi_head_timexer_configs.exogenous_configs():
        inherited = timexer[config["dataset_name"]]

        assert config["model"] == multi_head_timexer_configs.MODEL_NAME
        assert seeding.seed_identity_of(config) == seeding.seed_identity_of(inherited)
        assert seeding.seed_identity_of(config) == timexer_configs.MODEL_NAME


def test_multi_head_timexer_is_settled_and_has_nothing_to_search():
    entry = registry.entry(SHORT_TERM.name, multi_head_timexer_configs.MODEL_NAME)

    assert not entry.is_tunable
    assert not entry.has_search_space
    with pytest.raises(errors.ConfigurationError, match="nothing to search"):
        entry.trials()


def _without_the_parallelism_number(config: dict) -> dict:
    return {
        key: value for key, value in config.items() if key != scheduling.PARALLELISM_KEY
    }


def _sweep_trials_by_market() -> dict[str, list[dict]]:
    trials: dict[str, list[dict]] = {}
    for trial in multi_head_timexer_sweep_configs.exogenous_search_space():
        trials.setdefault(trial["dataset_name"], []).append(trial)
    return trials


def test_the_whole_sweep_is_thirty_four_points_seven_a_market_but_six_at_de():
    trials = multi_head_timexer_sweep_configs.exogenous_search_space()
    counts = {
        name: len([t for t in trials if t["dataset_name"] == name])
        for name in SHORT_TERM.datasets
    }

    assert len(trials) == 34
    assert counts == {"NP": 7, "PJM": 7, "BE": 7, "FR": 7, "DE": 6}


def test_each_swept_market_anchors_on_the_row_timexers_table_names():
    anchors = multi_head_timexer_sweep_configs.EXOGENOUS_ANCHORS

    assert anchors.keys() == set(SHORT_TERM.datasets)
    for name, (
        e_layers,
        _,
        d_ff,
        batch_size,
    ) in timexer_configs.EXOGENOUS_TABLE.items():
        assert anchors[name] == multi_head_timexer_configs.GridPoint(
            e_layers=e_layers, d_ff=d_ff, batch_size=batch_size
        )


def test_each_swept_market_stars_its_anchor_and_one_step_along_each_axis():
    for name, anchor in multi_head_timexer_sweep_configs.EXOGENOUS_ANCHORS.items():
        if name == "DE":
            continue
        points = multi_head_timexer_sweep_configs.EXOGENOUS_GRIDS[name]
        expected = {anchor} | {
            replace(anchor, **{axis: value})
            for axis, value in (
                ("e_layers", anchor.e_layers - 1),
                ("e_layers", anchor.e_layers + 1),
                ("d_ff", anchor.d_ff // 2),
                ("d_ff", anchor.d_ff * 2),
                ("batch_size", anchor.batch_size // 2),
                ("batch_size", anchor.batch_size * 2),
            )
        }

        assert len(points) == 7
        assert set(points) == expected


def test_des_star_is_the_six_points_the_one_layer_floor_allows():
    anchor = multi_head_timexer_sweep_configs.EXOGENOUS_ANCHORS["DE"]
    grid = multi_head_timexer_sweep_configs.EXOGENOUS_GRIDS["DE"]

    assert anchor.e_layers == 1
    assert [point.label for point in grid] == [
        "L1-F2048-B4",
        "L2-F2048-B4",
        "L1-F1024-B4",
        "L1-F4096-B4",
        "L1-F2048-B2",
        "L1-F2048-B8",
    ]
    multi_head_timexer_sweep_configs._refuse_stars_missing_a_step_no_axis_floor_forbids(
        {"DE": anchor}, {"DE": grid}
    )


def test_no_swept_grid_point_asks_for_a_zero_layer_encoder():
    for grid in multi_head_timexer_sweep_configs.EXOGENOUS_GRIDS.values():
        for point in grid:
            assert min(point.e_layers, point.d_ff, point.batch_size) >= 1


def test_a_sweep_trial_at_the_anchor_is_the_published_run_but_for_name_and_point():
    published = {
        c["dataset_name"]: c for c in multi_head_timexer_configs.exogenous_configs()
    }
    swept = _sweep_trials_by_market()

    for name, anchor in multi_head_timexer_sweep_configs.EXOGENOUS_ANCHORS.items():
        at_anchor = [t for t in swept[name] if search.grid_point_of(t) == anchor.label]
        assert len(at_anchor) == 1

        ours = _without_the_parallelism_number(at_anchor[0])
        theirs = _without_the_parallelism_number(published[name])
        differing = sorted(
            key
            for key in ours.keys() | theirs.keys()
            if ours.get(key) != theirs.get(key)
        )

        assert differing == sorted([search.GRID_POINT_KEY, "model"])


def test_every_sweep_trial_derives_the_seed_it_shares_with_timexer():
    for trial in multi_head_timexer_sweep_configs.exogenous_search_space():
        assert seeding.seed_identity_of(trial) == timexer_configs.MODEL_NAME
        assert seeding.seed_identity_of(trial) == (
            multi_head_timexer_configs.SEED_IDENTITY
        )
        assert trial["model"] == multi_head_timexer_sweep_configs.MODEL_NAME


def test_every_sweep_trial_is_schedulable_and_can_be_read_back():
    for trial in multi_head_timexer_sweep_configs.exogenous_search_space():
        assert search.grid_point_of(trial)
        assert scheduling.parallelism_of(trial) >= 1


def test_each_swept_market_runs_at_exactly_one_parallelism_number():
    numbered = multi_head_timexer_sweep_configs.EXOGENOUS_PARALLELISM
    carried: dict[str, set[int]] = {}
    for trial in multi_head_timexer_sweep_configs.exogenous_search_space():
        carried.setdefault(trial["dataset_name"], set()).add(
            scheduling.parallelism_of(trial)
        )

    assert carried == {name: {number} for name, number in numbered.items()}


def test_no_swept_market_sweeps_one_grid_point_twice():
    for name, trials in _sweep_trials_by_market().items():
        labels = [search.grid_point_of(t) for t in trials]

        assert len(labels) == len(set(labels)), name


def test_each_swept_markets_trials_stay_contiguous_and_in_the_settings_order():
    markets = [
        t["dataset_name"]
        for t in multi_head_timexer_sweep_configs.exogenous_search_space()
    ]
    first_seen = list(dict.fromkeys(markets))

    assert first_seen == list(SHORT_TERM.datasets)
    assert markets == sorted(markets, key=first_seen.index)


def test_the_swept_model_is_settled_and_keeps_the_space_it_was_settled_from():
    entry = registry.entry(SHORT_TERM.name, multi_head_timexer_sweep_configs.MODEL_NAME)

    assert not entry.is_tunable
    assert entry.has_search_space
    assert entry.trials() == (multi_head_timexer_sweep_configs.exogenous_search_space())
    assert entry.configs() == multi_head_timexer_sweep_configs.exogenous_configs()
    assert entry.known_configs() == entry.configs()


def test_the_swept_table_names_the_winners_its_own_search_chose():
    chosen = {
        c["dataset_name"]: (c["e_layers"], c["d_ff"], c["batch_size"])
        for c in multi_head_timexer_sweep_configs.exogenous_configs()
    }
    labelled = {
        name: point.label
        for name, point in multi_head_timexer_sweep_configs.EXOGENOUS_TABLE.items()
    }

    assert chosen == {
        "NP": (3, 512, 2),
        "PJM": (2, 2048, 16),
        "BE": (2, 256, 16),
        "FR": (2, 2048, 32),
        "DE": (2, 2048, 4),
    }
    assert labelled == {
        "NP": "L3-F512-B2",
        "PJM": "L2-F2048-B16",
        "BE": "L2-F256-B16",
        "FR": "L2-F2048-B32",
        "DE": "L2-F2048-B4",
    }


def test_the_swept_table_chooses_one_configuration_for_every_market():
    configs = multi_head_timexer_sweep_configs.exogenous_configs()

    assert [c["dataset_name"] for c in configs] == list(SHORT_TERM.datasets)
    assert set(multi_head_timexer_sweep_configs.EXOGENOUS_TABLE) == set(
        SHORT_TERM.datasets
    )
    assert {c["pred_len"] for c in configs} == {24}


def test_every_swept_winner_is_a_point_of_the_star_it_was_chosen_from():
    grids = multi_head_timexer_sweep_configs.EXOGENOUS_GRIDS

    for name, point in multi_head_timexer_sweep_configs.EXOGENOUS_TABLE.items():
        assert point in grids[name]


def test_a_winner_no_star_contains_is_refused_at_import():
    table = multi_head_timexer_sweep_configs.EXOGENOUS_TABLE
    grids = multi_head_timexer_sweep_configs.EXOGENOUS_GRIDS
    strayed = {
        **table,
        "BE": replace(table["BE"], d_ff=table["BE"].d_ff * 8),
    }

    assert strayed["BE"].label == "L2-F2048-B16"
    with pytest.raises(ValueError, match="L2-F2048-B16"):
        multi_head_timexer_sweep_configs._refuse_winners_no_star_contains(
            strayed, grids
        )


def test_a_swept_winner_keeps_the_number_its_market_was_swept_at():
    swept = multi_head_timexer_sweep_configs.EXOGENOUS_PARALLELISM

    for config in multi_head_timexer_sweep_configs.exogenous_configs():
        assert scheduling.parallelism_of(config) == swept[config["dataset_name"]]


def test_every_swept_chosen_configuration_is_byte_for_byte_a_trial_it_swept():
    won = multi_head_timexer_sweep_configs.EXOGENOUS_TABLE
    swept = {
        (t["dataset_name"], search.grid_point_of(t)): t
        for t in multi_head_timexer_sweep_configs.exogenous_search_space()
    }

    for config in multi_head_timexer_sweep_configs.exogenous_configs():
        name = config["dataset_name"]
        key = (name, search.grid_point_of(config))

        assert search.grid_point_of(config) == won[name].label
        assert key in swept
        assert config == swept[key]


def test_the_swept_model_builds_the_multi_head_timexer_the_inherited_row_builds():
    entry = registry.entry(SHORT_TERM.name, multi_head_timexer_sweep_configs.MODEL_NAME)
    inherited = registry.entry(SHORT_TERM.name, multi_head_timexer_configs.MODEL_NAME)

    assert entry.build is inherited.build
    assert entry.objective is inherited.objective


def test_a_market_left_without_a_parallelism_number_is_refused_at_import():
    numbered = multi_head_timexer_sweep_configs.EXOGENOUS_PARALLELISM

    with pytest.raises(ValueError, match="carry numbers"):
        multi_head_timexer_sweep_configs._refuse_unnumbered_markets(
            {name: n for name, n in numbered.items() if name != "DE"},
            SHORT_TERM.datasets,
        )

    with pytest.raises(ValueError, match="carry numbers"):
        multi_head_timexer_sweep_configs._refuse_unnumbered_markets(
            {**numbered, "Atlantis": 8}, SHORT_TERM.datasets
        )


def test_a_market_left_without_a_settled_configuration_is_refused_at_import():
    settled = multi_head_timexer_sweep_configs.EXOGENOUS_TABLE

    with pytest.raises(ValueError, match="carry configurations"):
        multi_head_timexer_sweep_configs._refuse_unsettled_markets(
            {name: p for name, p in settled.items() if name != "FR"},
            SHORT_TERM.datasets,
        )


def test_a_settled_configuration_at_a_misspelt_market_is_refused_at_import():
    settled = multi_head_timexer_sweep_configs.EXOGENOUS_TABLE
    misspelt = {
        **{name: p for name, p in settled.items() if name != "FR"},
        "FRA": settled["FR"],
    }

    with pytest.raises(ValueError, match="carry configurations"):
        multi_head_timexer_sweep_configs._refuse_unsettled_markets(
            misspelt, SHORT_TERM.datasets
        )


def test_a_star_thinned_below_an_anchor_clear_of_the_floor_is_refused_at_import():
    anchor = multi_head_timexer_sweep_configs.EXOGENOUS_ANCHORS["NP"]
    thinned = tuple(
        point
        for point in multi_head_timexer_sweep_configs.neighbourhood(anchor)
        if point.batch_size != anchor.batch_size // 2
    )

    with pytest.raises(ValueError, match="L3-F512-B2"):
        multi_head_timexer_sweep_configs._refuse_stars_missing_a_step_no_axis_floor_forbids(
            {"NP": anchor}, {"NP": thinned}
        )


@pytest.mark.parametrize(
    ("dropped", "label"),
    [("d_ff", "L1-F1024-B4"), ("batch_size", "L1-F2048-B2")],
)
def test_de_is_excused_only_on_the_axis_that_sits_on_its_floor(dropped, label):
    anchor = multi_head_timexer_sweep_configs.EXOGENOUS_ANCHORS["DE"]
    thinned = tuple(
        point
        for point in multi_head_timexer_sweep_configs.neighbourhood(anchor)
        if point.label != label
    )

    assert anchor.e_layers == 1
    assert getattr(anchor, dropped) > 1
    with pytest.raises(ValueError, match=label):
        multi_head_timexer_sweep_configs._refuse_stars_missing_a_step_no_axis_floor_forbids(
            {"DE": anchor}, {"DE": thinned}
        )


def test_the_swept_model_can_be_calibrated_anywhere_in_the_space_it_sweeps():
    config = cli._config_to_measure(
        SHORT_TERM,
        multi_head_timexer_sweep_configs.MODEL_NAME,
        "DE",
        24,
        "L1-F4096-B4",
    )

    assert search.grid_point_of(config) == "L1-F4096-B4"
    assert (config["e_layers"], config["d_ff"], config["batch_size"]) == (1, 4096, 4)
