from __future__ import annotations

import pytest

from halo import errors, paths, registry, results, settings, splits

SETTING = settings.SHORT_TERM_EXOGENOUS

PAPER_NAMES = {
    "TimeXer": "TimeXer",
    "MultiHeadTimeXer_sweep": "TimeXer + dual-head Halo",
    "MultiHeadTimeXer": "TimeXer + dual-head Halo, untuned",
    "TimeXer_sbs": "TimeXer + parallel Halo",
    "GCGNet": "GCGNet",
    "MultiHeadGCGNet": "GCGNet + dual-head Halo",
    "CrossLinear": "CrossLinear",
    "MultiHeadLinear": "CrossLinear + dual-head Halo",
    "CrossLinear_sbs": "CrossLinear + parallel Halo",
}


@pytest.fixture(name="results_dir")
def _results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.RESULTS_DIR_ENV_VAR, str(tmp_path))
    return tmp_path


def make_run(model="TimeXer", dataset="NP", mse=0.4, mae=0.2):
    return results.Run(
        model_name=model,
        dataset_name=dataset,
        pred_len=SETTING.horizons[0],
        mse=mse,
        mae=mae,
        best_epoch=3,
        seed=2021,
    )


def test_validation_keeps_the_filenames_in_use_today(results_dir):
    assert paths.db_path(splits.VALIDATION) == results_dir / "results.db"
    assert (
        results.csv_path_for(SETTING, splits.VALIDATION)
        == results_dir / "short_term_exogenous_results.csv"
    )
    assert (
        results.xlsx_path_for(SETTING, splits.VALIDATION)
        == results_dir / "short_term_exogenous_results.xlsx"
    )


def test_test_takes_its_own_trio_in_the_same_directory(results_dir):
    assert paths.db_path(splits.TEST) == results_dir / "test_results.db"
    assert (
        results.csv_path_for(SETTING, splits.TEST)
        == results_dir / "short_term_exogenous_test_results.csv"
    )
    assert (
        results.xlsx_path_for(SETTING, splits.TEST)
        == results_dir / "short_term_exogenous_test_results.xlsx"
    )


def test_the_legacy_database_name_is_only_the_validation_split_s_problem(results_dir):
    (results_dir / paths.LEGACY_DB_NAME).write_bytes(b"")

    assert paths.db_path(splits.TEST) == results_dir / "test_results.db"
    with pytest.raises(errors.ConfigurationError, match=paths.LEGACY_DB_NAME):
        paths.db_path(splits.VALIDATION)


def test_every_model_in_the_setting_carries_the_paper_name_of_its_column():
    assert {
        model: registry.entry(SETTING.name, model).paper_name
        for model in registry.models_for(SETTING.name)
    } == PAPER_NAMES


def test_the_nine_paper_names_are_distinct():
    assert len(set(PAPER_NAMES.values())) == len(PAPER_NAMES)


def test_two_models_claiming_one_paper_name_are_refused():
    entry = registry.entry(SETTING.name, "TimeXer")
    with pytest.raises(errors.ConfigurationError, match="TimeXer"):
        registry.refuse_colliding_paper_names(
            {
                (SETTING.name, "TimeXer"): entry,
                (SETTING.name, "TimeXer_sbs"): entry,
            }
        )


def test_the_same_paper_name_in_two_settings_is_not_a_collision():
    entry = registry.entry(SETTING.name, "TimeXer")
    registry.refuse_colliding_paper_names(
        {
            (SETTING.name, "TimeXer"): entry,
            ("elsewhere", "TimeXer"): entry,
        }
    )


def test_a_test_run_lands_under_its_paper_name_in_the_test_database(results_dir):
    results.record_run(
        make_run(
            model=registry.recorded_name(SETTING.name, "TimeXer", splits.TEST),
        ),
        setting=SETTING,
        split=splits.TEST,
    )

    stored = results.read_runs(setting=SETTING, split=splits.TEST)
    assert [run.model_name for run in stored] == ["TimeXer"]
    assert paths.db_path(splits.TEST).exists()
    assert not paths.db_path(splits.VALIDATION).exists()


def test_a_multi_head_test_run_lands_under_the_name_the_paper_prints(results_dir):
    results.record_run(
        make_run(
            model=registry.recorded_name(
                SETTING.name, "MultiHeadTimeXer_sweep", splits.TEST
            ),
        ),
        setting=SETTING,
        split=splits.TEST,
    )

    stored = results.read_runs(setting=SETTING, split=splits.TEST)
    assert [run.model_name for run in stored] == ["TimeXer + dual-head Halo"]


def test_a_validation_run_lands_under_its_code_name_in_the_validation_database(
    results_dir,
):
    results.record_run(
        make_run(
            model=registry.recorded_name(
                SETTING.name, "MultiHeadTimeXer_sweep", splits.VALIDATION
            ),
        ),
        setting=SETTING,
        split=splits.VALIDATION,
    )

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION)
    assert [run.model_name for run in stored] == ["MultiHeadTimeXer_sweep"]
    assert paths.db_path(splits.VALIDATION).exists()
    assert not paths.db_path(splits.TEST).exists()


def test_recording_one_split_leaves_the_other_s_rows_alone(results_dir):
    results.record_run(
        make_run(model="TimeXer", mse=0.1), setting=SETTING, split=splits.VALIDATION
    )
    results.record_run(
        make_run(model="TimeXer", mse=0.9), setting=SETTING, split=splits.TEST
    )

    assert [
        run.mse for run in results.read_runs(setting=SETTING, split=splits.VALIDATION)
    ] == [0.1]
    assert [
        run.mse for run in results.read_runs(setting=SETTING, split=splits.TEST)
    ] == [0.9]
