from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest
import torch

from halo import (
    calibrate,
    cli,
    errors,
    paths,
    results,
    settings,
    splits,
    validate,
    vendor,
)

SETTING = settings.Setting(
    name="many_horizons",
    datasets=settings.SHORT_TERM_EXOGENOUS.datasets,
    horizons=(96, 192, 336, 720),
    seq_len=96,
    label_len=48,
)
HORIZONS = SETTING.horizons


@pytest.fixture(name="db")
def _db(tmp_path):
    return tmp_path / "results.db"


def make_config(dataset="NP", horizon=96):
    return {"dataset_name": dataset, "pred_len": horizon, "parallelism": 4}


def record(db, dataset, horizon, seed):
    results.record_run(
        results.Run(
            model_name="TimeXer",
            dataset_name=dataset,
            pred_len=horizon,
            mse=0.4,
            mae=0.3,
            best_epoch=3,
            seed=seed,
        ),
        setting=SETTING,
        split=splits.VALIDATION,
        db_path=db,
    )


def test_ray_uv_runtime_env_is_actually_disabled():
    probe = (
        "import halo.validate;"
        "from ray._private import ray_constants;"
        "print(ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV)"
    )
    environment = dict(os.environ)
    environment.pop("RAY_ENABLE_UV_RUN_RUNTIME_ENV", None)
    finished = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )

    assert finished.stdout.strip().splitlines()[-1] == "False"


def test_a_recorded_config_at_the_same_seed_is_not_pending(db):
    record(db, "NP", 96, seed=2021)

    todo = validate.pending(
        [make_config()],
        model_name="TimeXer",
        seed=2021,
        setting=SETTING,
        split=splits.VALIDATION,
        db_path=db,
    )

    assert todo == []


def test_a_recorded_config_at_another_seed_is_still_pending(db):
    record(db, "NP", 96, seed=2021)

    todo = validate.pending(
        [make_config()],
        model_name="TimeXer",
        seed=7,
        setting=SETTING,
        split=splits.VALIDATION,
        db_path=db,
    )

    assert [c["pred_len"] for c in todo] == [96]


def test_force_keeps_every_config_regardless_of_seed(db):
    record(db, "NP", 96, seed=2021)

    todo = validate.pending(
        [make_config()],
        model_name="TimeXer",
        seed=2021,
        setting=SETTING,
        split=splits.VALIDATION,
        db_path=db,
        force=True,
    )

    assert [c["pred_len"] for c in todo] == [96]


def test_sweep_failed_names_every_lost_trial():
    error = errors.SweepFailed([("BE", 720), ("FR", 96)])

    assert "BE/S=720" in str(error)
    assert "FR/S=96" in str(error)
    assert error.failures == [("BE", 720), ("FR", 96)]


def test_the_cli_exits_non_zero_when_a_sweep_loses_trials(monkeypatch):
    def explode(**kwargs):
        raise errors.SweepFailed([("BE", 720)])

    monkeypatch.setattr(validate, "run", explode)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--model", "TimeXer"])

    assert exit_info.value.code != 0
    assert "BE/S=720" in str(exit_info.value)


def test_the_cli_reports_a_configuration_error_without_a_traceback(monkeypatch):
    monkeypatch.delenv(paths.RESULTS_DIR_ENV_VAR, raising=False)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["export"])

    assert paths.RESULTS_DIR_ENV_VAR in str(exit_info.value)


def test_an_unexpected_runtime_error_keeps_its_traceback(monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(validate, "run", explode)
    with pytest.raises(RuntimeError, match="MPS backend"):
        cli.main(["run", "--model", "TimeXer"])


def test_a_configuration_error_is_a_runtime_error():
    assert issubclass(errors.ConfigurationError, RuntimeError)
    assert issubclass(errors.SweepFailed, RuntimeError)


def test_a_stale_schema_is_caught_before_training_even_under_force(db, monkeypatch):
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

    def never(*args, **kwargs):
        raise AssertionError("training must not start")

    monkeypatch.setattr(validate, "sweep", never)
    monkeypatch.setattr(validate.data, "fetch", never)

    with pytest.raises(errors.ConfigurationError, match="rebuild"):
        validate.run(
            model_name="TimeXer",
            setting=SETTING,
            split=splits.VALIDATION,
            datasets=["NP"],
            horizons=[96],
            db_path=db,
            force=True,
        )


def test_a_missing_vendored_tree_is_reported_as_a_configuration_error(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(vendor.VENDOR_DIR_ENV_VAR, str(tmp_path / "absent"))

    with pytest.raises(errors.ConfigurationError, match="get_deps"):
        vendor.ensure_tree(vendor.TIME_SERIES_LIBRARY)


def test_an_equal_validation_loss_counts_as_an_improvement():
    best, epochs_since_best = 0.5, 0

    for value in (0.5, 0.5, 0.5):
        if value <= best:
            best, epochs_since_best = value, 0
        else:
            epochs_since_best += 1

    assert epochs_since_best == 0


def test_a_config_naming_no_setting_cannot_reach_a_config_table():
    with pytest.raises(errors.ConfigurationError, match="imaginary_setting"):
        validate.registry.entry("imaginary_setting", "TimeXer")


def test_the_csv_is_written_before_a_sweep_failure_propagates(
    db, monkeypatch, tmp_path
):
    record(db, "NP", 96, seed=2021)
    for horizon in (192, 336, 720):
        record(db, "NP", horizon, seed=2021)

    def fail(*args, **kwargs):
        raise errors.SweepFailed([("BE", 720)])

    monkeypatch.setitem(
        validate.registry.MODEL_REGISTRY,
        (SETTING.name, "TimeXer"),
        validate.registry.ModelEntry(
            build=lambda config: torch.nn.Module(),
            paper_name="TimeXer",
            settled_table=lambda: [
                {"dataset_name": "BE", "pred_len": 720, "parallelism": 1}
            ],
        ),
    )
    monkeypatch.setattr(validate, "sweep", fail)
    monkeypatch.setattr(
        validate.data, "fetch", lambda names: {name: str(tmp_path) for name in names}
    )

    with pytest.raises(errors.SweepFailed):
        validate.run(
            model_name="TimeXer",
            setting=SETTING,
            split=splits.VALIDATION,
            datasets=["BE"],
            horizons=[720],
            db_path=db,
            force=True,
        )

    assert results.csv_path_for(SETTING, splits.VALIDATION, tmp_path).exists()


def test_a_model_that_cannot_offer_configs_does_not_stop_the_models_after_it(
    db, monkeypatch, tmp_path
):
    def record_each(configs, *, model_name, setting, split, seed, db_path):
        for config in configs:
            results.record_run(
                results.Run(
                    model_name=model_name,
                    dataset_name=config["dataset_name"],
                    pred_len=config["pred_len"],
                    mse=0.4,
                    mae=0.3,
                    best_epoch=3,
                    seed=seed,
                ),
                setting=setting,
                split=split,
                db_path=db_path,
            )
        return []

    monkeypatch.setitem(
        validate.registry.MODEL_REGISTRY,
        (SETTING.name, "StillSearching"),
        validate.registry.ModelEntry(
            build=lambda config: torch.nn.Module(),
            paper_name="Still searching",
            search_space=lambda: [
                {"dataset_name": "BE", "pred_len": 720, "parallelism": 1}
            ],
        ),
    )
    monkeypatch.setitem(
        validate.registry.MODEL_REGISTRY,
        (SETTING.name, "TimeXer"),
        validate.registry.ModelEntry(
            build=lambda config: torch.nn.Module(),
            paper_name="TimeXer",
            settled_table=lambda: [
                {"dataset_name": "BE", "pred_len": 720, "parallelism": 1}
            ],
        ),
    )
    monkeypatch.setattr(validate, "sweep", record_each)
    monkeypatch.setattr(
        validate.data, "fetch", lambda names: {name: str(tmp_path) for name in names}
    )

    with pytest.raises(errors.ModelsFailed, match="StillSearching"):
        validate.run_models(
            model_names=["StillSearching", "TimeXer"],
            setting=SETTING,
            split=splits.VALIDATION,
            datasets=["BE"],
            horizons=[720],
            db_path=db,
            force=True,
        )

    ran = results.read_runs(
        setting=SETTING, split=splits.VALIDATION, model_name="TimeXer", db_path=db
    )
    assert [(run.dataset_name, run.pred_len) for run in ran] == [("BE", 720)]
    assert results.csv_path_for(SETTING, splits.VALIDATION, tmp_path).exists()


def test_calibration_reports_its_measurements_when_the_baseline_dies():
    measurements = [
        calibrate.Measurement(instances=1, rates=[], died=1),
        calibrate.Measurement(instances=2, rates=[3.0, 3.0]),
    ]

    assert calibrate.suggest_parallelism(measurements) is None
    assert "3.00" in calibrate.report(measurements)
    assert "no suggestion" in calibrate.report(measurements)
