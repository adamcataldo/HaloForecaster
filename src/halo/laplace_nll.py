from __future__ import annotations

import torch
from torch import Tensor

SCALE_FLOOR = 1e-6


def laplace_nll_loss(location: Tensor, scale: Tensor, target: Tensor) -> Tensor:
    floored = scale.clone()
    with torch.no_grad():
        floored.clamp_(min=SCALE_FLOOR)
    return (torch.log(floored) + (target - location).abs() / floored).mean()
