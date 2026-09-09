from __future__ import annotations

from typing import Any

import pytest
import torch

from halo import multi_head_timexer, seeding, timexer_configs, tslib

SEED = 20260826
BATCH_SIZE = 3
MARKET = "NP"

VENDORED_HEAD = "head."
OUR_HEAD = "heads.0."

SMALL: dict[str, Any] = {"d_model": 64, "d_ff": 128, "e_layers": 2, "n_heads": 4}


def config(**overrides: Any) -> dict[str, Any]:
    inherited = next(
        config
        for config in timexer_configs.exogenous_configs()
        if config["dataset_name"] == MARKET
    )
    return {**inherited, **SMALL, **overrides}


def vendored_and_ours(config: dict[str, Any]):
    seeding.seed_everything(SEED)
    vendored = tslib.model_builder(timexer_configs.MODEL_NAME)(config).eval()
    seeding.seed_everything(SEED)
    ours = multi_head_timexer.builder()(config).eval()
    return vendored, ours


def under_the_vendored_name(key: str) -> str:
    return key.replace(OUR_HEAD, VENDORED_HEAD, 1) if key.startswith(OUR_HEAD) else key


def varying_across_batch_and_time_and_channel(
    config: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    x_enc, x_mark_enc, x_dec, x_mark_dec = tslib.example_batch(
        config, BATCH_SIZE, seeding.torch_generator(SEED)
    )
    spread = (
        torch.arange(float(BATCH_SIZE)).reshape(-1, 1, 1)
        + torch.arange(float(config["seq_len"])).reshape(1, -1, 1)
        + torch.arange(float(config["enc_in"])).reshape(1, 1, -1)
    )
    return x_enc + spread, x_mark_enc, x_dec, x_mark_dec


@pytest.fixture(params=[1, 0], ids=["standardized", "unstandardized"], name="use_norm")
def _use_norm(request) -> int:
    return request.param


def test_the_probe_input_varies_across_batch_and_time_and_channel():
    x_enc, _, _, _ = varying_across_batch_and_time_and_channel(config())

    assert (x_enc.std(dim=0) > 0).all()
    assert (x_enc.std(dim=1) > 0).all()
    assert (x_enc.std(dim=2) > 0).all()


def test_the_probe_configuration_keeps_timexers_nonzero_dropout():
    assert config()["dropout"] > 0


def test_the_stored_tensors_correspond_one_for_one_in_both_directions(use_norm):
    vendored, ours = vendored_and_ours(config(use_norm=use_norm))

    theirs = vendored.state_dict()
    mine = {under_the_vendored_name(key): t for key, t in ours.state_dict().items()}

    assert len(mine) == len(ours.state_dict())
    assert set(mine) == set(theirs)
    assert len(mine) == len(theirs)


def test_corresponding_tensors_are_already_equal_without_anything_being_copied(
    use_norm,
):
    vendored, ours = vendored_and_ours(config(use_norm=use_norm))

    theirs = vendored.state_dict()
    mine = {under_the_vendored_name(key): t for key, t in ours.state_dict().items()}

    for name, tensor in mine.items():
        assert torch.equal(tensor, theirs[name]), name


def test_one_head_and_both_callables_defaulted_reproduces_timexers_forecast(use_norm):
    probe = config(use_norm=use_norm)
    vendored, ours = vendored_and_ours(probe)
    batch = varying_across_batch_and_time_and_channel(probe)

    with torch.no_grad():
        theirs = vendored(*batch)
        mine = ours(*batch)

    assert mine.shape == theirs.shape
    assert torch.allclose(mine, theirs, rtol=1e-5, atol=1e-6)
