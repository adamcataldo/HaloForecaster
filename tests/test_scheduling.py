from __future__ import annotations

import pytest

from halo import scheduling
from halo.registry import MODEL_REGISTRY


def config(parallelism, name="Atlantis", horizon=96):
    return {
        "dataset_name": name,
        "pred_len": horizon,
        scheduling.PARALLELISM_KEY: parallelism,
    }


def test_uniform_parallelism_fills_the_budget_exactly():
    budget, costs = scheduling.slot_costs([config(4) for _ in range(4)])

    assert budget == 4
    assert costs == [1, 1, 1, 1]
    assert sum(costs) == budget


def test_budget_is_the_lcm_so_mixed_numbers_stay_exact():
    budget, costs = scheduling.slot_costs([config(2), config(3), config(4)])

    assert budget == 12
    assert costs == [6, 4, 3]
    assert all(isinstance(cost, int) for cost in costs)


def test_three_thirds_add_up_to_exactly_one_where_floats_would_not():
    budget, costs = scheduling.slot_costs([config(3) for _ in range(3)])

    assert sum(costs) == budget
    assert scheduling.admits(budget, sum(costs[:2]), costs[2])


def test_a_serial_job_never_shares():
    budget, costs = scheduling.slot_costs([config(1, "Lemuria"), config(4, "Atlantis")])
    serial, small = costs

    assert scheduling.admits(budget, 0, serial)
    assert not scheduling.admits(budget, serial, small)
    assert not scheduling.admits(budget, small, serial)


def test_four_p4_jobs_fit_and_a_fifth_does_not():
    budget, costs = scheduling.slot_costs([config(4) for _ in range(5)])

    assert scheduling.admits(budget, sum(costs[:3]), costs[3])
    assert not scheduling.admits(budget, sum(costs[:4]), costs[4])


def test_one_p2_and_two_p4_exactly_fill():
    budget, costs = scheduling.slot_costs([config(2), config(4), config(4)])
    half, quarter_a, quarter_b = costs

    assert half + quarter_a + quarter_b == budget
    assert scheduling.admits(budget, half + quarter_a, quarter_b)
    assert not scheduling.admits(budget, budget, 1)


def test_empty_selection_is_harmless():
    assert scheduling.slot_costs([]) == (1, [])


@pytest.mark.parametrize("bad", [0, -1, 1.5, None, True, "4"])
def test_a_missing_or_nonsense_parallelism_is_refused(bad):
    with pytest.raises(ValueError):
        scheduling.parallelism_of(config(bad))


@pytest.mark.parametrize("registered", sorted(MODEL_REGISTRY))
def test_every_registered_config_carries_a_parallelism_number(registered):
    for config in MODEL_REGISTRY[registered].known_configs():
        assert scheduling.parallelism_of(config) >= 1
