from __future__ import annotations

import pytest
import torch

from halo import search, seeding, tslib
from halo.registry import MODEL_REGISTRY

SEED = 20260811
BATCH_SIZE = 2

SEARCHED = ("short_term_exogenous", "MultiHeadTimeXer_sweep")


def small_config(registered: tuple[str, str]) -> dict:
    config = dict(MODEL_REGISTRY[registered].known_configs()[0])
    config.update(
        d_model=32,
        d_ff=64,
        e_layers=1,
        n_heads=4,
        enc_in=3,
        dec_in=3,
        c_out=3,
        pred_len=16,
    )
    return config


@pytest.fixture(params=sorted(MODEL_REGISTRY), name="registered")
def _registered(request):
    return request.param


def test_same_seed_gives_the_same_forward_pass(registered):
    config = small_config(registered)
    batch = tslib.example_batch(config, BATCH_SIZE, seeding.torch_generator(SEED))

    seeding.seed_everything(SEED)
    model = MODEL_REGISTRY[registered].build(config).eval()

    with torch.no_grad():
        seeding.seed_everything(SEED)
        first = model(*batch)
        seeding.seed_everything(SEED)
        second = model(*batch)

    assert torch.equal(first, second)


def test_two_independent_builds_at_one_seed_give_identical_weights(registered):
    config = small_config(registered)

    seeding.seed_everything(SEED)
    first = MODEL_REGISTRY[registered].build(config)

    seeding.seed_everything(SEED)
    second = MODEL_REGISTRY[registered].build(config)

    first_params = dict(first.named_parameters())
    second_params = dict(second.named_parameters())
    assert first_params.keys() == second_params.keys()
    for name, parameter in first_params.items():
        assert torch.equal(parameter, second_params[name]), name


def test_seeding_that_ignored_its_argument_would_be_caught(registered):
    config = small_config(registered)

    seeding.seed_everything(SEED)
    first = MODEL_REGISTRY[registered].build(config)

    seeding.seed_everything(SEED + 1)
    second = MODEL_REGISTRY[registered].build(config)

    second_params = dict(second.named_parameters())
    assert any(
        not torch.equal(parameter, second_params[name])
        for name, parameter in first.named_parameters()
    )


def test_a_config_that_declares_nothing_seeds_from_its_model_name():
    assert seeding.seed_identity_of({"model": "TimeXer"}) == "TimeXer"

    for registered, entry in MODEL_REGISTRY.items():
        if registered[1] != "TimeXer":
            continue
        for config in entry.known_configs():
            assert seeding.seed_identity_of(config) == config["model"]


def test_a_declared_identity_replaces_the_model_name_in_the_derivation():
    declared = {"model": "Working_name", seeding.SEED_IDENTITY_KEY: "Published"}

    assert seeding.seed_identity_of(declared) == "Published"
    assert seeding.derive_seed(
        2021, seeding.seed_identity_of(declared), "NP", 24
    ) == seeding.derive_seed(2021, "Published", "NP", 24)
    assert seeding.derive_seed(
        2021, seeding.seed_identity_of(declared), "NP", 24
    ) != seeding.derive_seed(2021, "Working_name", "NP", 24)


def test_renaming_a_model_that_declared_an_identity_changes_no_seed():
    before = {"model": "Working_name", seeding.SEED_IDENTITY_KEY: "Published"}
    after = {"model": "Published", seeding.SEED_IDENTITY_KEY: "Published"}

    assert seeding.derive_seed(
        2021, seeding.seed_identity_of(before), "NP", 24
    ) == seeding.derive_seed(2021, seeding.seed_identity_of(after), "NP", 24)


def _derived(config):
    return seeding.derive_seed(
        2021,
        seeding.seed_identity_of(config),
        config["dataset_name"],
        config["pred_len"],
    )


def test_the_searched_hyperparameters_do_not_reach_the_seed():
    trials = [
        config
        for config in MODEL_REGISTRY[SEARCHED].trials()
        if config["dataset_name"] == "NP"
    ]

    assert len(trials) == 7
    assert len({_derived(config) for config in trials}) == 1


def test_a_chosen_configuration_derives_the_seed_its_winning_trial_used():
    entry = MODEL_REGISTRY[SEARCHED]
    trials = {
        (config["dataset_name"], config[search.GRID_POINT_KEY]): config
        for config in entry.trials()
    }

    for config in entry.configs():
        trial = trials[(config["dataset_name"], config[search.GRID_POINT_KEY])]

        assert _derived(config) == _derived(trial)


@pytest.mark.parametrize("bad", ["", 7, None])
def test_an_identity_that_is_not_a_name_is_refused(bad):
    config = {"model": "MultiHeadTimeXer", seeding.SEED_IDENTITY_KEY: bad}

    if bad is None:
        assert seeding.seed_identity_of(config) == "MultiHeadTimeXer"
    else:
        with pytest.raises(ValueError):
            seeding.seed_identity_of(config)


def test_derive_seed_is_stable_and_distinguishes_runs():
    assert seeding.derive_seed(2021, "TimeXer", "Atlantis", 96) == seeding.derive_seed(
        2021, "TimeXer", "Atlantis", 96
    )
    distinct = {
        seeding.derive_seed(2021, "TimeXer", dataset, horizon)
        for dataset in ("Atlantis", "Lemuria")
        for horizon in (96, 192, 336, 720)
    }
    assert len(distinct) == 8
    assert seeding.derive_seed(2021, "TimeXer", "Atlantis", 96) != seeding.derive_seed(
        2022, "TimeXer", "Atlantis", 96
    )
