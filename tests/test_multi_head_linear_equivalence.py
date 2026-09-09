from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from torch import nn

from halo import crosslinear_configs, multi_head_linear, seeding, tslib

SEED = 20260827
BATCH_SIZE = 3
MARKET = "NP"

VENDORED_HEAD = "head."
OUR_HEAD = "heads.0."

SMALL: dict[str, Any] = {"d_model": 32, "d_ff": 64}


def config(**overrides: Any) -> dict[str, Any]:
    inherited = next(
        config
        for config in crosslinear_configs.exogenous_configs()
        if config["dataset_name"] == MARKET
    )
    return {**inherited, **SMALL, **overrides}


def vendored_and_ours(config: dict[str, Any]) -> tuple[nn.Module, nn.Module]:
    seeding.seed_everything(SEED)
    vendored = crosslinear_configs.build(config).eval()
    seeding.seed_everything(SEED)
    ours = multi_head_linear.builder()(config).eval()
    return vendored, ours


def padding_of(model: nn.Module) -> int:
    return int(cast(Any, model).value_embedding.pad_num)


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


@pytest.fixture(params=[16, 24], ids=["padded", "dividing"], name="patch_len")
def _patch_len(request) -> int:
    return request.param


@pytest.fixture(params=[1.0, 2.0], ids=["alpha_one", "alpha_two"], name="alpha")
def _alpha(request) -> float:
    return request.param


def test_the_probe_input_varies_across_batch_and_time_and_channel():
    x_enc, _, _, _ = varying_across_batch_and_time_and_channel(config())

    assert (x_enc.std(dim=0) > 0).all()
    assert (x_enc.std(dim=1) > 0).all()
    assert (x_enc.std(dim=2) > 0).all()


def test_the_two_patch_lengths_are_the_padded_case_and_the_dividing_case(patch_len):
    seq_len = config()["seq_len"]
    padding = math.ceil(seq_len / patch_len) * patch_len - seq_len

    assert (patch_len, padding) in ((16, 8), (24, 0))


def test_the_padding_the_two_models_carry_is_the_same_padding(patch_len):
    vendored, ours = vendored_and_ours(config(patch_len=patch_len))

    assert padding_of(ours) == padding_of(vendored)


def test_the_pipeline_neither_samples_nor_drops_out_so_no_reseeding_is_needed(
    patch_len, alpha
):
    probe = config(patch_len=patch_len, alpha=alpha)
    _, ours = vendored_and_ours(probe)
    batch = varying_across_batch_and_time_and_channel(probe)

    assert not any(isinstance(module, nn.Dropout) for module in ours.modules())

    with torch.no_grad():
        ours.train()
        twice_in_training = [ours(*batch) for _ in range(2)]
        ours.eval()
        twice_in_evaluation = [ours(*batch) for _ in range(2)]

    assert torch.equal(*twice_in_training)
    assert torch.equal(*twice_in_evaluation)
    assert torch.equal(twice_in_training[0], twice_in_evaluation[0])


def test_the_stored_tensors_correspond_one_for_one_in_both_directions(patch_len, alpha):
    vendored, ours = vendored_and_ours(config(patch_len=patch_len, alpha=alpha))

    theirs = vendored.state_dict()
    mine = {under_the_vendored_name(key): t for key, t in ours.state_dict().items()}

    assert len(mine) == len(ours.state_dict())
    assert set(mine) == set(theirs)
    assert len(mine) == len(theirs)


def test_corresponding_tensors_are_already_equal_without_anything_being_copied(
    patch_len, alpha
):
    vendored, ours = vendored_and_ours(config(patch_len=patch_len, alpha=alpha))

    theirs = vendored.state_dict()
    mine = {under_the_vendored_name(key): t for key, t in ours.state_dict().items()}

    for name, tensor in mine.items():
        assert torch.equal(tensor, theirs[name]), name


def test_one_head_and_both_callables_defaulted_reproduces_crosslinears_forecast(
    patch_len, alpha
):
    probe = config(patch_len=patch_len, alpha=alpha)
    vendored, ours = vendored_and_ours(probe)
    batch = varying_across_batch_and_time_and_channel(probe)

    with torch.no_grad():
        theirs = vendored(*batch)
        mine = ours(*batch)

    assert mine.shape == theirs.shape
    assert torch.allclose(mine, theirs, rtol=1e-5, atol=1e-6)
