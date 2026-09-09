from __future__ import annotations

from torch import Tensor
from torch.nn.functional import gaussian_nll_loss

BETA = 0.5


def beta_nll_loss(
    mean: Tensor,
    var: Tensor,
    target: Tensor,
    beta: float = BETA,
) -> Tensor:
    nll = gaussian_nll_loss(mean, target, var, reduction="none")
    weight = var.detach() ** beta
    return (weight * nll).mean()
