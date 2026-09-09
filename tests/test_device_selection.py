from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest
import torch

from halo import registry, settings, validate

CPU = torch.device("cpu")
MPS = torch.device("mps")

GCGNET = (settings.SHORT_TERM_EXOGENOUS.name, "GCGNet")

MULTI_HEAD_GCGNET = (settings.SHORT_TERM_EXOGENOUS.name, "MultiHeadGCGNet")

CPU_PINNED = (GCGNET, MULTI_HEAD_GCGNET)

MACHINE_DEFAULTED = tuple(
    key for key in registry.MODEL_REGISTRY if key not in CPU_PINNED
)


@pytest.mark.parametrize("key", MACHINE_DEFAULTED, ids=lambda key: f"{key[1]}/{key[0]}")
def test_a_model_naming_no_device_takes_the_metal_one_when_it_exists(key):
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        assert validate.select_device(registry.MODEL_REGISTRY[key]) == MPS


@pytest.mark.parametrize("key", MACHINE_DEFAULTED, ids=lambda key: f"{key[1]}/{key[0]}")
def test_a_model_naming_no_device_falls_back_to_cpu_on_a_machine_without_metal(key):
    with patch.object(torch.backends.mps, "is_available", return_value=False):
        assert validate.select_device(registry.MODEL_REGISTRY[key]) == CPU


@pytest.mark.parametrize("key", CPU_PINNED, ids=lambda key: f"{key[1]}/{key[0]}")
def test_a_gcgnet_trains_on_cpu_even_where_the_metal_device_is_available(key):
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        assert validate.select_device(registry.MODEL_REGISTRY[key]) == CPU


def test_the_two_gcgnets_are_the_only_models_that_override_the_machines_choice():
    overriding = {
        model
        for (_, model), entry in registry.MODEL_REGISTRY.items()
        if entry.device is not None
    }
    assert overriding == {"GCGNet", "MultiHeadGCGNet"}


def test_an_entrys_named_device_wins_over_the_cpu_fallback():
    pinned = dataclasses.replace(registry.MODEL_REGISTRY[GCGNET], device=MPS)
    with patch.object(torch.backends.mps, "is_available", return_value=False):
        assert validate.select_device(pinned) == MPS


def test_an_entrys_named_device_wins_over_an_available_metal_device():
    pinned = dataclasses.replace(registry.MODEL_REGISTRY[GCGNET], device=CPU)
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        assert validate.select_device(pinned) == CPU
