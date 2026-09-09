from __future__ import annotations

import pytest
import torch

from halo import (
    multi_head_gcgnet,
    multi_head_linear,
    multi_head_timexer,
    registry,
    seeding,
    settings,
    side_by_side,
    tslib,
    validate,
)

SETTING = settings.SHORT_TERM_EXOGENOUS
REFERENCE = "TimeXer"
MARKET = "NP"
HORIZON = 24
BATCH_SIZE = 2
SEED = 20260813
CPU = torch.device("cpu")

OURS = tuple(model for model in registry.models_for(SETTING.name) if model != REFERENCE)

DIRECT_ONLY_PARAMETERS = frozenset(
    multi_head_gcgnet.DIRECT_ONLY_PARAMETERS
    + multi_head_linear.DIRECT_ONLY_PARAMETERS
    + multi_head_timexer.DIRECT_ONLY_PARAMETERS
    + side_by_side.DIRECT_ONLY_PARAMETERS
)

LOADER_KEYS = (
    "seq_len",
    "label_len",
    "features",
    "freq",
    "data",
    "data_path",
    "enc_in",
    "dec_in",
    "c_out",
    "pred_len",
    "target",
    "embed",
    "task_name",
)


class Recorder(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: tuple[torch.Tensor, ...] = ()

    def forward(self, *arguments: torch.Tensor) -> torch.Tensor:
        self.seen = arguments
        return self.inner(*arguments)


def config_for(model_name: str, market: str = MARKET) -> dict:
    entry = registry.entry(SETTING.name, model_name)
    return next(
        config
        for config in entry.known_configs()
        if config["dataset_name"] == market and config["pred_len"] == HORIZON
    )


def loader_batch(config: dict) -> tuple[torch.Tensor, ...]:
    x_enc, x_mark_enc, x_dec, x_mark_dec = tslib.example_batch(
        config, BATCH_SIZE, seeding.torch_generator(SEED)
    )
    return x_enc, x_dec, x_mark_enc, x_mark_dec


def scored(model_name: str, config: dict, batch: tuple[torch.Tensor, ...]):
    seeding.seed_everything(SEED)
    recorder = Recorder(registry.entry(SETTING.name, model_name).build(config).eval())
    with torch.no_grad():
        prediction, target = validate._forward(recorder, batch, config, CPU)
    return recorder.seen, prediction, target


@pytest.fixture(name="reference")
def _reference() -> dict:
    return config_for(REFERENCE)


@pytest.fixture(params=OURS, name="model_name")
def _model_name(request) -> str:
    return request.param


def test_every_model_agrees_with_timexer_on_what_the_loader_emits(
    reference, model_name
):
    ours = config_for(model_name)

    for key in LOADER_KEYS:
        assert reference[key] == ours[key], key


def test_every_model_is_handed_the_identical_tensors_timexer_is(reference, model_name):
    batch = loader_batch(reference)

    reference_seen, _, _ = scored(REFERENCE, reference, batch)
    ours_seen, _, _ = scored(model_name, config_for(model_name), batch)

    assert len(reference_seen) == len(ours_seen) == 4
    for handed_to_timexer, handed_to_ours in zip(
        reference_seen, ours_seen, strict=True
    ):
        assert torch.equal(handed_to_timexer, handed_to_ours)


def test_every_model_gives_back_an_identically_shaped_forecast(reference, model_name):
    batch = loader_batch(reference)

    _, reference_prediction, _ = scored(REFERENCE, reference, batch)
    _, our_prediction, _ = scored(model_name, config_for(model_name), batch)

    assert reference_prediction.shape == our_prediction.shape
    assert our_prediction.shape == (BATCH_SIZE, HORIZON, 1)


def test_every_model_is_scored_against_the_same_target(reference, model_name):
    batch = loader_batch(reference)

    _, _, reference_target = scored(REFERENCE, reference, batch)
    _, _, our_target = scored(model_name, config_for(model_name), batch)

    assert torch.equal(reference_target, our_target)


@pytest.mark.parametrize("model_name", registry.models_for(SETTING.name))
def test_the_shared_path_is_reached_through_one_construction_call_per_model(
    model_name,
):
    entry = registry.entry(SETTING.name, model_name)

    assert isinstance(entry.build(config_for(model_name)), torch.nn.Module)


@pytest.mark.parametrize("model_name", registry.models_for(SETTING.name))
def test_no_checked_in_configuration_carries_a_constructor_only_parameter(model_name):
    entry = registry.entry(SETTING.name, model_name)

    for config in entry.known_configs():
        assert not set(config) & DIRECT_ONLY_PARAMETERS


@pytest.mark.parametrize("market", SETTING.datasets)
def test_every_market_declares_the_variable_count_its_loader_actually_emits(
    market, model_name
):
    config = config_for(model_name, market)

    x_enc, _, _, _ = tslib.example_batch(
        config, BATCH_SIZE, seeding.torch_generator(SEED)
    )

    assert x_enc.size(2) == config["enc_in"]
