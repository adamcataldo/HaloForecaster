from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

Objective = Callable[
    [nn.Module, tuple[Tensor, ...], dict[str, Any], torch.device], Tensor
]


@dataclass(frozen=True)
class Prepared:
    x_enc: Tensor
    x_mark_enc: Tensor
    x_dec: Tensor
    x_mark_dec: Tensor
    target: Tensor
    future: Tensor

    def inputs(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec


def scored_channel(config: dict[str, Any]) -> int:
    features = config["features"]
    if features != "MS":
        raise ValueError(
            f"features={features!r} is not implemented. This loop handles "
            "'MS', which scores the target column alone; anything else would "
            "need its own scored-column rule."
        )
    return -1


def prepare(
    batch: tuple[Tensor, ...],
    config: dict[str, Any],
    device: torch.device,
) -> Prepared:
    f_dim = scored_channel(config)

    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    pred_len = config["pred_len"]
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
    dec_inp = torch.cat([batch_y[:, : config["label_len"], :], dec_inp], dim=1)

    future = batch_y[:, -pred_len:, :]
    return Prepared(
        x_enc=batch_x,
        x_mark_enc=batch_x_mark,
        x_dec=dec_inp,
        x_mark_dec=batch_y_mark,
        target=future[:, :, f_dim:],
        future=future,
    )


def scored(outputs: Tensor, config: dict[str, Any]) -> Tensor:
    return outputs[:, -config["pred_len"] :, scored_channel(config) :]


def squared_error(
    model: nn.Module,
    batch: tuple[Tensor, ...],
    config: dict[str, Any],
    device: torch.device,
) -> Tensor:
    prepared = prepare(batch, config, device)
    prediction = scored(model(*prepared.inputs()), config)
    return torch.nn.functional.mse_loss(prediction, prepared.target)
