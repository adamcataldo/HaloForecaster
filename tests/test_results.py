from __future__ import annotations

import csv
import sqlite3

import pytest

from halo import errors, paths, results, settings, splits

SHORT_TERM = settings.SHORT_TERM_EXOGENOUS
SETTING = settings.Setting(
    name="many_horizons",
    datasets=SHORT_TERM.datasets,
    horizons=(96, 192, 336, 720),
    seq_len=96,
    label_len=48,
)
HORIZONS = SETTING.horizons
MARKETS = SHORT_TERM.datasets
SEED = 2021


@pytest.fixture(name="db")
def _db(tmp_path):
    return tmp_path / "results.db"


def make_run(dataset="NP", horizon=96, mse=0.4, mae=0.3, model="TimeXer", seed=SEED):
    return results.Run(
        model_name=model,
        dataset_name=dataset,
        pred_len=horizon,
        mse=mse,
        mae=mae,
        best_epoch=3,
        seed=seed,
    )


def record(db, run, setting=SETTING):
    return results.record_run(run, setting=setting, split=splits.VALIDATION, db_path=db)


def record_all(db, mses, *, dataset="NP", model="TimeXer"):
    rolled = None
    for horizon, mse in zip(HORIZONS, mses, strict=True):
        rolled = record(
            db,
            make_run(
                dataset=dataset, horizon=horizon, mse=mse, mae=mse / 2, model=model
            ),
        )
    return rolled


def record_market(db, market, mse, *, mae=None, seed=SEED, model="TimeXer"):
    return record(
        db,
        make_run(
            dataset=market,
            horizon=SHORT_TERM.horizons[0],
            mse=mse,
            mae=mse / 2 if mae is None else mae,
            model=model,
            seed=seed,
        ),
        setting=SHORT_TERM,
    )


def read_csv(directory, setting=SETTING):
    with results.csv_path_for(setting, splits.VALIDATION, directory).open(
        newline=""
    ) as handle:
        return list(csv.reader(handle))


def models_of(rows):
    return [name for name in rows[0] if name]


def datasets_of(rows):
    return [row[0] for row in rows[2:]]


def cell(rows, model, dataset):
    column = 1 + 2 * models_of(rows).index(model)
    row = next(row for row in rows[2:] if row[0] == dataset)
    return (row[column], row[column + 1])


def published(rows):
    return {
        (model, dataset): cell(rows, model, dataset)
        for model in models_of(rows)
        for dataset in datasets_of(rows)
        if any(cell(rows, model, dataset))
    }


def test_a_run_is_stored_the_moment_it_finishes(db):
    record(db, make_run())

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.dataset_name, r.pred_len, r.mse) for r in stored] == [("NP", 96, 0.4)]


def test_the_database_stores_only_measurements_never_an_aggregate(db):
    record_all(db, [0.1, 0.2, 0.3, 0.4])

    with results._connect_for_write(db, splits.VALIDATION) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {results.table_for(SETTING)}


def test_no_roll_up_until_every_horizon_is_in(db, tmp_path):
    for horizon in HORIZONS[:3]:
        assert record(db, make_run(horizon=horizon)) is None
    assert published(read_csv(tmp_path)) == {}


def test_roll_up_is_the_unweighted_mean_of_four_horizons(db, tmp_path):
    rolled = record_all(db, [0.1, 0.2, 0.3, 0.4])

    assert rolled == results.Result("TimeXer", "NP", 0.25, 0.125)
    assert published(read_csv(tmp_path)) == {("TimeXer", "NP"): ("0.25", "0.125")}


def test_rerunning_a_horizon_overwrites_and_refreshes_the_csv(db, tmp_path):
    record_all(db, [0.1, 0.2, 0.3, 0.4])
    rolled = record(db, make_run(horizon=96, mse=0.5, mae=0.25))

    assert len(
        results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    ) == len(HORIZONS)
    assert rolled.mse == pytest.approx((0.5 + 0.2 + 0.3 + 0.4) / 4)
    assert cell(read_csv(tmp_path), "TimeXer", "NP") == (
        str(rolled.mse),
        str(rolled.mae),
    )


def test_resume_query_reports_what_is_already_done(db):
    record(db, make_run(horizon=96))
    record(db, make_run(horizon=336))

    done = {
        r.pred_len
        for r in results.read_runs(
            model_name="TimeXer", setting=SETTING, split=splits.VALIDATION, db_path=db
        )
    }
    assert done == {96, 336}
    assert (
        results.read_runs(
            model_name="Absent", setting=SETTING, split=splits.VALIDATION, db_path=db
        )
        == []
    )


def test_distinct_pairs_coexist(db, tmp_path):
    record_all(db, [0.1, 0.2, 0.3, 0.4], dataset="NP")
    record_all(db, [0.5, 0.5, 0.5, 0.5], dataset="PJM")
    record_all(db, [0.9, 0.9, 0.9, 0.9], dataset="NP", model="iTransformer")

    rows = read_csv(tmp_path)
    assert set(published(rows)) == {
        ("TimeXer", "NP"),
        ("TimeXer", "PJM"),
        ("iTransformer", "NP"),
    }


def test_an_incomplete_dataset_is_an_empty_pair_of_cells_rather_than_a_number(
    db, tmp_path
):
    record_all(db, [0.1, 0.2, 0.3, 0.4], dataset="NP")
    for horizon in HORIZONS[:3]:
        record(db, make_run(dataset="BE", horizon=horizon, mse=0.1))

    rows = read_csv(tmp_path)
    assert "BE" in datasets_of(rows)
    assert cell(rows, "TimeXer", "BE") == ("", "")
    assert set(published(rows)) == {("TimeXer", "NP")}


def test_the_header_is_two_lines_with_each_models_name_over_its_first_column(
    db, tmp_path
):
    record_all(db, [0.4321] * 4, dataset="PJM")
    record_all(db, [0.9] * 4, dataset="PJM", model="iTransformer")

    rows = read_csv(tmp_path)
    assert rows[0] == ["", "TimeXer", "", "iTransformer", ""]
    assert rows[1] == ["", "MSE", "MAE", "MSE", "MAE"]


def test_the_rows_are_every_dataset_the_setting_names_in_its_own_order(db, tmp_path):
    record_all(db, [0.241] * 4, dataset="NP")

    assert datasets_of(read_csv(tmp_path)) == list(SETTING.datasets)


def test_models_are_columns_sorted_by_name(db, tmp_path):
    for model in ("iTransformer", "TimeXer", "PatchTST"):
        record_all(db, [0.4] * 4, dataset="NP", model=model)

    assert models_of(read_csv(tmp_path)) == ["PatchTST", "TimeXer", "iTransformer"]


def test_a_model_with_nothing_published_gets_no_columns(db, tmp_path):
    record_all(db, [0.4] * 4, dataset="NP")
    for horizon in HORIZONS[:3]:
        record(db, make_run(horizon=horizon, model="iTransformer"))

    assert models_of(read_csv(tmp_path)) == ["TimeXer"]


def test_a_cell_keeps_the_full_precision_of_the_number_behind_it(db, tmp_path):
    rolled = record_all(db, [0.1, 0.2, 0.30000000000000004, 0.4])

    assert cell(read_csv(tmp_path), "TimeXer", "NP") == (
        repr(rolled.mse),
        repr(rolled.mae),
    )


def test_the_two_settings_are_two_tables_in_one_file(db, tmp_path):
    record_all(db, [0.4] * 4, dataset="NP")
    record_market(db, "NP", 0.238)

    assert results.table_for(SHORT_TERM) == "short_term_exogenous_runs"
    assert (
        results.csv_path_for(SHORT_TERM, splits.VALIDATION, tmp_path).name
        == "short_term_exogenous_results.csv"
    )
    assert [
        r.mse
        for r in results.read_runs(
            setting=SHORT_TERM, split=splits.VALIDATION, db_path=db
        )
    ] == [0.238]
    assert [
        r.mse
        for r in results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    ] == [0.4] * 4


def test_a_single_horizon_setting_publishes_a_row_from_one_finished_run(db, tmp_path):
    rolled = record_market(db, "NP", 0.238, mae=0.244)

    assert rolled == results.Result("TimeXer", "NP", 0.238, 0.244)
    assert published(read_csv(tmp_path, SHORT_TERM)) == {
        ("TimeXer", "NP"): ("0.238", "0.244")
    }


def test_the_two_settings_roll_up_independently(db, tmp_path):
    for horizon in HORIZONS[:3]:
        record(db, make_run(dataset="NP", horizon=horizon))
    record_market(db, "NP", 0.238)

    assert published(read_csv(tmp_path, SETTING)) == {}
    assert set(published(read_csv(tmp_path, SHORT_TERM))) == {("TimeXer", "NP")}


def test_the_average_is_the_unweighted_mean_of_the_five_markets(db, tmp_path):
    mses = [0.10, 0.20, 0.30, 0.40, 0.55]
    for market, mse in zip(MARKETS, mses, strict=True):
        record_market(db, market, mse)

    assert cell(read_csv(tmp_path, SHORT_TERM), "TimeXer", "Avg") == (
        str(sum(mses) / 5),
        str(sum(mses) / 10),
    )


def test_the_average_is_an_empty_pair_of_cells_until_every_market_is_in(db, tmp_path):
    for market in MARKETS[:-1]:
        record_market(db, market, 0.3)

    rows = read_csv(tmp_path, SHORT_TERM)
    assert cell(rows, "TimeXer", "Avg") == ("", "")
    assert set(published(rows)) == {("TimeXer", market) for market in MARKETS[:-1]}


def test_the_average_stays_empty_while_the_markets_disagree_on_seed(db, tmp_path):
    for market in MARKETS[:-1]:
        record_market(db, market, 0.3, seed=SEED)
    record_market(db, MARKETS[-1], 0.3, seed=SEED + 1)

    rows = read_csv(tmp_path, SHORT_TERM)
    assert cell(rows, "TimeXer", "Avg") == ("", "")
    assert set(published(rows)) == {("TimeXer", market) for market in MARKETS}


def test_the_average_returns_once_the_markets_agree_on_a_new_seed(db, tmp_path):
    for market in MARKETS:
        record_market(db, market, 0.3, seed=SEED)
    for market in MARKETS:
        record_market(db, market, 0.5, seed=SEED + 1)

    assert cell(read_csv(tmp_path, SHORT_TERM), "TimeXer", "Avg") == ("0.5", "0.25")


def test_the_average_is_the_last_row_beneath_every_models_columns(db, tmp_path):
    for market in MARKETS:
        record_market(db, market, 0.3)
        record_market(db, market, 0.7, model="iTransformer")

    rows = read_csv(tmp_path, SHORT_TERM)
    assert datasets_of(rows) == [*MARKETS, "Avg"]
    assert cell(rows, "TimeXer", "Avg") == ("0.3", "0.15")
    assert cell(rows, "iTransformer", "Avg") == ("0.7", "0.35")


def test_a_setting_without_an_average_has_no_row_for_one(db, tmp_path):
    record_all(db, [0.1, 0.2, 0.3, 0.4], dataset="NP")
    record_all(db, [0.5] * 4, dataset="PJM")

    rows = read_csv(tmp_path)
    assert datasets_of(rows) == list(SETTING.datasets)
    assert SETTING.average_name is None


def test_recording_a_market_reports_that_market_rather_than_the_average(db):
    for market in MARKETS[:-1]:
        record_market(db, market, 0.3)
    rolled = record_market(db, MARKETS[-1], 0.8)

    assert rolled == results.Result("TimeXer", MARKETS[-1], 0.8, 0.4)


def test_results_directory_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.RESULTS_DIR_ENV_VAR, str(tmp_path / "elsewhere"))

    assert paths.results_dir() == tmp_path / "elsewhere"
    assert paths.db_path(splits.VALIDATION).parent == tmp_path / "elsewhere"


def test_the_database_is_named_for_the_results_not_for_one_setting(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(paths.RESULTS_DIR_ENV_VAR, str(tmp_path))

    assert paths.db_path(splits.VALIDATION) == tmp_path / "results.db"


def test_a_results_directory_still_on_the_old_name_stops_rather_than_starting_over(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(paths.RESULTS_DIR_ENV_VAR, str(tmp_path))
    (tmp_path / paths.LEGACY_DB_NAME).touch()

    with pytest.raises(errors.ConfigurationError, match=paths.LEGACY_DB_NAME):
        paths.db_path(splits.VALIDATION)


def test_the_old_database_is_ignored_once_the_renamed_one_is_beside_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(paths.RESULTS_DIR_ENV_VAR, str(tmp_path))
    (tmp_path / paths.LEGACY_DB_NAME).touch()
    (tmp_path / paths.DB_NAME).touch()

    assert paths.db_path(splits.VALIDATION) == tmp_path / paths.DB_NAME


@pytest.mark.parametrize("value", [None, "", "   "])
def test_results_directory_raises_rather_than_guessing_when_unset(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(paths.RESULTS_DIR_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(paths.RESULTS_DIR_ENV_VAR, value)

    with pytest.raises(RuntimeError, match=paths.RESULTS_DIR_ENV_VAR):
        paths.results_dir()


def hostile(name):
    return settings.Setting(
        name=name, datasets=("NP",), horizons=(96,), seq_len=96, label_len=48
    )


def test_a_setting_name_cannot_smuggle_sql(db):
    with pytest.raises(errors.ConfigurationError):
        results.read_runs(
            setting=hostile("runs; DROP TABLE x"), split=splits.VALIDATION, db_path=db
        )


def test_a_setting_name_cannot_smuggle_sql_even_when_the_database_is_absent(db):
    assert not db.exists()

    with pytest.raises(errors.ConfigurationError):
        results.read_runs(
            setting=hostile("runs; DROP TABLE x"), split=splits.VALIDATION, db_path=db
        )


def test_a_setting_name_is_rejected_before_any_directory_is_created(db, tmp_path):
    record_all(db, [0.1, 0.2, 0.3, 0.4])
    escape = tmp_path / "outside"

    with pytest.raises(errors.ConfigurationError):
        results.export_tables(
            setting=hostile(f"../{escape.name}/evil"),
            split=splits.VALIDATION,
            db_path=db,
        )

    assert not escape.exists()


def test_a_setting_cannot_name_a_dataset_that_does_not_exist():
    with pytest.raises(ValueError, match="Atlantis"):
        settings.Setting(
            name="imaginary",
            datasets=("NP", "Atlantis"),
            horizons=(96,),
            seq_len=96,
            label_len=48,
        )


def test_a_setting_cannot_label_its_average_after_one_of_its_datasets():
    with pytest.raises(ValueError, match="NP"):
        settings.Setting(
            name="colliding",
            datasets=("NP", "PJM"),
            horizons=(96,),
            seq_len=96,
            label_len=48,
            average_name="NP",
        )


def test_an_out_of_date_table_says_to_rebuild_rather_than_failing_on_a_column(db):
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"""
            CREATE TABLE {results.table_for(SETTING)} (
                model_name TEXT, dataset_name TEXT, pred_len INTEGER,
                MSE REAL, MAE REAL, best_epoch INTEGER,
                PRIMARY KEY (model_name, dataset_name, pred_len)
            )
            """
        )

    with pytest.raises(errors.ConfigurationError, match="rebuild"):
        results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)


def test_reading_a_missing_database_does_not_create_one(tmp_path):
    absent = tmp_path / "typo" / "results.db"

    assert (
        results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=absent)
        == []
    )
    assert not absent.exists()
    assert not absent.parent.exists()


def test_the_stored_seed_comes_back_with_the_run(db):
    record(db, make_run(seed=7))

    assert [
        r.seed
        for r in results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    ] == [7]


def test_recording_at_a_new_seed_overwrites_rather_than_appending(db):
    record(db, make_run(mse=0.4, seed=2021))
    record(db, make_run(mse=0.9, seed=7))

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.mse, r.seed) for r in stored] == [(0.9, 7)]


@pytest.mark.parametrize("mse,mae", [(float("inf"), 0.3), (0.4, float("nan"))])
def test_a_non_finite_result_is_refused_rather_than_stored(db, mse, mae):
    with pytest.raises(ValueError, match="non-finite"):
        record(db, make_run(mse=mse, mae=mae))

    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def test_horizons_measured_at_different_seeds_are_not_averaged_together(db, tmp_path):
    record_all(db, [0.1, 0.2, 0.3, 0.4])
    record(db, make_run(horizon=96, mse=0.9, mae=0.45, seed=7))

    assert published(read_csv(tmp_path)) == {}


def test_a_dataset_reappears_once_every_horizon_shares_the_new_seed(db, tmp_path):
    record_all(db, [0.4, 0.4, 0.4, 0.4])
    for horizon in HORIZONS:
        record(db, make_run(horizon=horizon, mse=0.8, mae=0.4, seed=7))

    assert published(read_csv(tmp_path)) == {("TimeXer", "NP"): ("0.8", "0.4")}


def test_exporting_against_a_missing_database_refuses(tmp_path):
    absent = tmp_path / "results.db"

    with pytest.raises(errors.ConfigurationError, match="nothing to roll"):
        results.export_tables(setting=SETTING, split=splits.VALIDATION, db_path=absent)


def test_a_missing_database_leaves_an_existing_csv_untouched(db, tmp_path):
    record_all(db, [0.1, 0.2, 0.3, 0.4])
    before = results.csv_path_for(SETTING, splits.VALIDATION, tmp_path).read_text()
    db.unlink()

    with pytest.raises(errors.ConfigurationError):
        results.export_tables(setting=SETTING, split=splits.VALIDATION, db_path=db)

    assert (
        results.csv_path_for(SETTING, splits.VALIDATION, tmp_path).read_text() == before
    )


def test_an_empty_database_still_exports_the_skeleton_with_no_model_columns(
    db, tmp_path
):
    record(db, make_run())
    results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute(f"DELETE FROM {results.table_for(SETTING)}")

    results.export_tables(setting=SETTING, split=splits.VALIDATION, db_path=db)

    rows = read_csv(tmp_path)
    assert models_of(rows) == []
    assert datasets_of(rows) == list(SETTING.datasets)


def test_check_schema_rejects_an_out_of_date_table_without_reading_runs(db):
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"""
            CREATE TABLE {results.table_for(SETTING)} (
                model_name TEXT, dataset_name TEXT, pred_len INTEGER,
                MSE REAL, MAE REAL, best_epoch INTEGER,
                PRIMARY KEY (model_name, dataset_name, pred_len)
            )
            """
        )

    with pytest.raises(errors.ConfigurationError, match="rebuild"):
        results.check_schema(setting=SETTING, split=splits.VALIDATION, db_path=db)


def test_check_schema_accepts_a_database_that_does_not_exist_yet(tmp_path):
    results.check_schema(
        setting=SETTING, split=splits.VALIDATION, db_path=tmp_path / "absent.db"
    )


def make_trial(label="L1-F512-B4", mse=0.4, dataset="NP", seed=SEED, config=None):
    return results.Trial(
        model_name="MultiHeadTimeXer",
        dataset_name=dataset,
        pred_len=SHORT_TERM.horizons[0],
        grid_point=label,
        mse=mse,
        mae=mse / 2,
        best_epoch=3,
        seed=seed,
        config={"e_layers": 1, "d_ff": 512, "batch_size": 4}
        if config is None
        else config,
    )


def record_trial(db, trial):
    results.record_trial(trial, setting=SHORT_TERM, split=splits.VALIDATION, db_path=db)


def test_a_trial_comes_back_with_the_configuration_that_produced_it(db):
    record_trial(db, make_trial())

    stored = results.read_trials(
        setting=SHORT_TERM, split=splits.VALIDATION, db_path=db
    )
    assert len(stored) == 1
    assert stored[0] == make_trial()


def test_a_trial_is_keyed_by_its_grid_point_as_well_as_its_market(db):
    record_trial(db, make_trial(label="L1-F512-B4", mse=0.4))
    record_trial(db, make_trial(label="L2-F512-B4", mse=0.2))

    stored = results.read_trials(
        setting=SHORT_TERM, split=splits.VALIDATION, db_path=db
    )
    assert [(t.grid_point, t.mse) for t in stored] == [
        ("L1-F512-B4", 0.4),
        ("L2-F512-B4", 0.2),
    ]


def test_re_running_one_grid_point_overwrites_rather_than_appending(db):
    record_trial(db, make_trial(mse=0.4, seed=SEED))
    record_trial(db, make_trial(mse=0.9, seed=7))

    stored = results.read_trials(
        setting=SHORT_TERM, split=splits.VALIDATION, db_path=db
    )
    assert [(t.mse, t.seed) for t in stored] == [(0.9, 7)]


def test_trials_can_be_narrowed_to_one_market_and_horizon(db):
    record_trial(db, make_trial(dataset="NP"))
    record_trial(db, make_trial(dataset="PJM"))

    stored = results.read_trials(
        setting=SHORT_TERM,
        dataset_name="PJM",
        pred_len=SHORT_TERM.horizons[0],
        split=splits.VALIDATION,
        db_path=db,
    )
    assert [t.dataset_name for t in stored] == ["PJM"]


def test_a_non_finite_trial_is_refused_rather_than_stored(db):
    with pytest.raises(ValueError, match="non-finite"):
        record_trial(db, make_trial(mse=float("nan")))

    assert (
        results.read_trials(setting=SHORT_TERM, split=splits.VALIDATION, db_path=db)
        == []
    )


def test_a_trial_is_not_published_and_writes_no_csv(db, tmp_path):
    record_trial(db, make_trial())

    assert (
        results.read_runs(setting=SHORT_TERM, split=splits.VALIDATION, db_path=db) == []
    )
    assert not results.csv_path_for(SHORT_TERM, splits.VALIDATION, tmp_path).exists()


def test_reading_trials_from_a_database_with_no_search_yet_is_empty(db):
    record(db, make_run(), setting=SHORT_TERM)

    assert (
        results.read_trials(setting=SHORT_TERM, split=splits.VALIDATION, db_path=db)
        == []
    )


def test_check_schema_rejects_an_out_of_date_trials_table(db):
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"""
            CREATE TABLE {results.trials_table_for(SHORT_TERM)} (
                model_name TEXT, dataset_name TEXT, pred_len INTEGER,
                grid_point TEXT, MSE REAL, MAE REAL, best_epoch INTEGER,
                PRIMARY KEY (model_name, dataset_name, pred_len, grid_point)
            )
            """
        )

    with pytest.raises(errors.ConfigurationError, match="rebuild"):
        results.check_schema(setting=SHORT_TERM, split=splits.VALIDATION, db_path=db)


def test_a_failed_export_leaves_the_previous_csv_intact(db, tmp_path, monkeypatch):
    record_all(db, [0.1, 0.2, 0.3, 0.4])
    before = results.csv_path_for(SETTING, splits.VALIDATION, tmp_path).read_text()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(results.csv, "writer", explode)
    with pytest.raises(OSError):
        results.export_tables(setting=SETTING, split=splits.VALIDATION, db_path=db)

    assert (
        results.csv_path_for(SETTING, splits.VALIDATION, tmp_path).read_text() == before
    )
    assert list(tmp_path.glob(".*tmp")) == []
