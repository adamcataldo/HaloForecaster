from __future__ import annotations

import pytest

from halo import results, search, settings, splits, validate

SETTING = settings.SHORT_TERM_EXOGENOUS
HORIZON = SETTING.horizons[0]
SEED = 2021
LABELS = ("L1-F512-B4", "L2-F512-B4", "L3-F512-B4")


@pytest.fixture(name="db")
def _db(tmp_path):
    return tmp_path / "results.db"


def config(label, *, dataset="NP"):
    return {
        "dataset_name": dataset,
        "pred_len": HORIZON,
        "parallelism": 1,
        search.GRID_POINT_KEY: label,
    }


def record(db, label, mse, *, dataset="NP", seed=SEED, model="MultiHeadTimeXer"):
    results.record_trial(
        results.Trial(
            model_name=model,
            dataset_name=dataset,
            pred_len=HORIZON,
            grid_point=label,
            mse=mse,
            mae=mse / 2,
            best_epoch=3,
            seed=seed,
            config=config(label, dataset=dataset),
        ),
        setting=SETTING,
        split=splits.VALIDATION,
        db_path=db,
    )


def record_group(db, mses, *, dataset="NP", seed=SEED):
    for label, mse in zip(LABELS, mses, strict=True):
        record(db, label, mse, dataset=dataset, seed=seed)


def pending(db, labels, *, seed=SEED, force=False):
    return validate.pending_trials(
        [config(label) for label in labels],
        model_name="MultiHeadTimeXer",
        setting=SETTING,
        seed=seed,
        db_path=db,
        force=force,
    )


def winners(db, labels, *, seed=SEED, datasets=("NP",)):
    return validate.record_winners(
        [config(label, dataset=name) for name in datasets for label in labels],
        model_name="MultiHeadTimeXer",
        setting=SETTING,
        seed=seed,
        db_path=db,
    )


def test_a_trial_recorded_at_this_seed_is_not_pending(db):
    record(db, "L1-F512-B4", 0.4)

    assert pending(db, ["L1-F512-B4"]) == []


def test_a_sibling_grid_point_is_still_pending(db):
    record(db, "L1-F512-B4", 0.4)

    todo = pending(db, LABELS)

    assert [c[search.GRID_POINT_KEY] for c in todo] == ["L2-F512-B4", "L3-F512-B4"]


def test_a_trial_recorded_at_another_seed_is_run_again(db):
    record(db, "L1-F512-B4", 0.4, seed=7)

    assert [c[search.GRID_POINT_KEY] for c in pending(db, ["L1-F512-B4"])] == [
        "L1-F512-B4"
    ]


def test_force_keeps_every_trial_regardless_of_what_is_recorded(db):
    record_group(db, [0.4, 0.2, 0.3])

    assert len(pending(db, LABELS, force=True)) == 3


def test_another_model_s_trials_do_not_satisfy_this_one(db):
    record(db, "L1-F512-B4", 0.4, model="TimeXer")

    assert len(pending(db, ["L1-F512-B4"])) == 1


def test_an_unfinished_group_publishes_nothing(db):
    record(db, "L1-F512-B4", 0.4)

    assert winners(db, LABELS) == []
    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def test_a_finished_group_publishes_its_best_trial_as_an_ordinary_run(db):
    record_group(db, [0.4, 0.2, 0.3])

    winners(db, LABELS)

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.model_name, r.dataset_name, r.pred_len) for r in stored] == [
        ("MultiHeadTimeXer", "NP", HORIZON)
    ]
    assert (stored[0].mse, stored[0].mae, stored[0].seed) == (0.2, 0.1, SEED)


def test_the_published_winner_rolls_up_without_a_special_case(db):
    record_group(db, [0.4, 0.2, 0.3])

    winners(db, LABELS)

    rolled = results.roll_up(
        results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db), SETTING
    )
    assert [(r.dataset_name, r.mse) for r in rolled] == [("NP", 0.2)]


def test_every_market_publishes_before_the_average_appears(db):
    for market in SETTING.datasets:
        record_group(db, [0.4, 0.2, 0.3], dataset=market)

    winners(db, LABELS, datasets=SETTING.datasets)

    rolled = results.roll_up(
        results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db), SETTING
    )
    assert [r.dataset_name for r in rolled] == [*sorted(SETTING.datasets), "Avg"]
    assert rolled[-1].mse == 0.2


def test_winners_are_picked_from_the_table_not_from_this_invocation(db):
    record_group(db, [0.4, 0.2, 0.3])

    assert [r.dataset_name for r in winners(db, LABELS)] == ["NP"]


def test_a_group_recorded_at_another_seed_is_not_published(db):
    record_group(db, [0.4, 0.2, 0.3], seed=7)

    assert winners(db, LABELS) == []
    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def test_an_interrupted_re_tune_at_a_new_seed_withdraws_the_stale_published_row(db):
    record_group(db, [0.4, 0.2, 0.3])
    winners(db, LABELS)
    assert (
        len(results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db))
        == 1
    )

    pending(db, LABELS, seed=7)
    record(db, LABELS[0], 0.35, seed=7)

    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def test_a_re_tune_at_a_new_seed_republishes_once_every_trial_lands(db):
    record_group(db, [0.4, 0.2, 0.3])
    winners(db, LABELS)

    pending(db, LABELS, seed=7)
    record_group(db, [0.9, 0.8, 0.7], seed=7)
    winners(db, LABELS, seed=7)

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.mse, r.seed) for r in stored] == [(0.7, 7)]


def test_resuming_at_the_same_seed_leaves_the_published_row_alone(db):
    record_group(db, [0.4, 0.2, 0.3])
    winners(db, LABELS)

    pending(db, LABELS)

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.mse, r.seed) for r in stored] == [(0.2, SEED)]


def test_forcing_a_re_tune_at_the_same_seed_withdraws_the_stale_row(db):
    record_group(db, [0.4, 0.2, 0.3])
    winners(db, LABELS)

    pending(db, LABELS, force=True)
    record(db, LABELS[0], 0.35)

    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def test_forcing_a_re_tune_at_a_new_seed_also_withdraws_the_stale_row(db):
    record_group(db, [0.4, 0.2, 0.3])
    winners(db, LABELS)

    pending(db, LABELS, seed=7, force=True)

    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def publish_ordinary_run(db, mse, *, dataset="NP", seed=SEED):
    results.record_run(
        results.Run(
            model_name="MultiHeadTimeXer",
            dataset_name=dataset,
            pred_len=HORIZON,
            mse=mse,
            mae=mse / 2,
            best_epoch=7,
            seed=seed,
        ),
        setting=SETTING,
        split=splits.VALIDATION,
        db_path=db,
    )


def test_a_row_published_by_an_ordinary_run_survives_an_interrupted_re_tune(db):
    publish_ordinary_run(db, 0.25)
    record_group(db, [0.4, 0.2, 0.3])

    pending(db, LABELS, force=True)
    record(db, LABELS[0], 0.35)

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.dataset_name, r.mse, r.best_epoch) for r in stored] == [("NP", 0.25, 7)]


def test_a_row_published_by_an_ordinary_run_survives_a_re_tune_at_a_new_seed(db):
    publish_ordinary_run(db, 0.25)
    record_group(db, [0.4, 0.2, 0.3])

    pending(db, LABELS, seed=7)

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.dataset_name, r.mse) for r in stored] == [("NP", 0.25)]


def test_a_row_matching_its_winning_trial_is_still_withdrawn(db):
    record_group(db, [0.4, 0.2, 0.3])
    winners(db, LABELS)
    published = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [(r.mse, r.best_epoch, r.seed) for r in published] == [(0.2, 3, SEED)]

    pending(db, LABELS, force=True)

    assert results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db) == []


def test_withdrawing_one_market_leaves_the_others_published(db):
    for market in ("NP", "PJM"):
        record_group(db, [0.4, 0.2, 0.3], dataset=market)
    winners(db, LABELS, datasets=("NP", "PJM"))

    validate.pending_trials(
        [config(label, dataset="NP") for label in LABELS],
        model_name="MultiHeadTimeXer",
        setting=SETTING,
        seed=7,
        db_path=db,
    )

    stored = results.read_runs(setting=SETTING, split=splits.VALIDATION, db_path=db)
    assert [r.dataset_name for r in stored] == ["PJM"]
