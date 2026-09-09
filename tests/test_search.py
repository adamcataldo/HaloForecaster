from __future__ import annotations

import pytest

from halo import results, search

LABELS = ("L1-F512-B4", "L2-F512-B4", "L3-F512-B4")
SEED = 2021


def trial(grid_point, mse, *, seed=SEED, mae=None):
    return results.Trial(
        model_name="MultiHeadTimeXer",
        dataset_name="NP",
        pred_len=24,
        grid_point=grid_point,
        mse=mse,
        mae=mse / 2 if mae is None else mae,
        best_epoch=3,
        seed=seed,
        config={},
    )


def test_the_lowest_validation_error_wins():
    chosen = search.best(
        [trial("L1-F512-B4", 0.4), trial("L2-F512-B4", 0.2), trial("L3-F512-B4", 0.3)]
    )

    assert chosen.grid_point == "L2-F512-B4"


def test_a_tie_falls_to_the_label_that_sorts_first_whatever_the_row_order():
    tied = [
        trial("L3-F512-B4", 0.2),
        trial("L1-F512-B4", 0.2),
        trial("L2-F512-B4", 0.2),
    ]

    assert search.best(tied).grid_point == "L1-F512-B4"
    assert search.best(reversed(tied)).grid_point == "L1-F512-B4"
    assert search.best(sorted(tied, key=lambda t: t.grid_point)).grid_point == (
        "L1-F512-B4"
    )


def test_choosing_between_no_trials_is_refused():
    with pytest.raises(ValueError, match="no trials"):
        search.best([])


def test_a_group_missing_one_trial_has_no_winner_yet():
    recorded = [trial("L1-F512-B4", 0.4), trial("L2-F512-B4", 0.2)]

    assert search.winner(recorded, expected=LABELS, seed=SEED) is None


def test_a_group_with_every_trial_present_has_a_winner():
    recorded = [
        trial(label, mse) for label, mse in zip(LABELS, (0.4, 0.2, 0.3), strict=True)
    ]

    chosen = search.winner(recorded, expected=LABELS, seed=SEED)

    assert chosen is not None
    assert chosen.grid_point == "L2-F512-B4"


def test_trials_measured_at_another_seed_do_not_complete_a_group():
    recorded = [
        trial("L1-F512-B4", 0.4),
        trial("L2-F512-B4", 0.2),
        trial("L3-F512-B4", 0.1, seed=7),
    ]

    assert search.winner(recorded, expected=LABELS, seed=SEED) is None


def test_a_better_trial_at_another_seed_never_wins():
    recorded = [trial(label, 0.4) for label in LABELS]
    recorded.append(trial("L2-F512-B4", 0.01, seed=7))

    chosen = search.winner(recorded, expected=LABELS, seed=SEED)

    assert chosen is not None
    assert chosen.mse == 0.4


def test_a_label_outside_the_search_cannot_win():
    recorded = [trial(label, 0.4) for label in LABELS]
    recorded.append(trial("L9-F512-B4", 0.01))

    chosen = search.winner(recorded, expected=LABELS, seed=SEED)

    assert chosen is not None
    assert chosen.grid_point in LABELS


def test_an_empty_search_has_no_winner():
    assert search.winner([], expected=(), seed=SEED) is None


def test_a_config_without_a_label_is_refused_by_name():
    with pytest.raises(ValueError, match=search.GRID_POINT_KEY):
        search.grid_point_of({"dataset_name": "NP", "pred_len": 24})

    with pytest.raises(ValueError, match=search.GRID_POINT_KEY):
        search.grid_point_of({search.GRID_POINT_KEY: ""})
