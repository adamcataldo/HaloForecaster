from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from halo import objectives, seeding, tslib

LoaderBuilder = Callable[[dict[str, Any], int, str], DataLoader]

Forward = Callable[
    [nn.Module, tuple[Tensor, ...], dict[str, Any], torch.device],
    tuple[Tensor, Tensor],
]


TRAIN_FLAG = "train"


def vendored_loader(config: dict[str, Any], seed: int, flag: str) -> DataLoader:
    data_factory = tslib.data_provider()
    args = tslib.as_namespace(config)

    dataset, _ = data_factory.data_provider(args, flag)
    training = flag == TRAIN_FLAG
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=training,
        num_workers=0,
        drop_last=False,
        generator=seeding.torch_generator(seed) if training else None,
    )


def vendored_forward(
    model: nn.Module,
    batch: tuple[Tensor, ...],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    prepared = objectives.prepare(batch, config, device)
    outputs = model(*prepared.inputs())
    return objectives.scored(outputs, config), prepared.target


@dataclass(frozen=True)
class DataPipeline:
    build_loader: LoaderBuilder = vendored_loader

    forward: Forward = vendored_forward


VENDORED = DataPipeline()
