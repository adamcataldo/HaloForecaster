from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn.functional import softplus

from halo import beta_nll, objectives
from halo.forecast import DistributionalForecaster

LOCATION = 0
SCALE = 1

OUTPUT_HEADS = 2


def location(parameters: Tensor) -> Tensor:
    return parameters[..., LOCATION : LOCATION + 1]


def scale_in_target_units(
    parameters: Tensor, window_mean: Tensor, window_stdev: Tensor
) -> Tensor:
    return torch.cat(
        (
            parameters[..., :SCALE],
            window_stdev * softplus(parameters[..., SCALE:]),
        ),
        dim=-1,
    )


def beta_nll_objective(
    model: torch.nn.Module,
    batch: tuple[Tensor, ...],
    config: dict[str, Any],
    device: torch.device,
) -> Tensor:
    prepared = objectives.prepare(batch, config, device)
    forecast = cast(DistributionalForecaster, model).predict(
        prepared.x_enc, prepared.x_mark_enc
    )
    scale = forecast.parameters[..., SCALE : SCALE + 1]
    return beta_nll.beta_nll_loss(
        forecast.point_estimate, scale * scale, prepared.target
    )
