from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True)
class Forecast:
    parameters: Tensor

    point_estimate: Tensor


@dataclass(frozen=True)
class RawForecast:
    raw: Tensor

    window_mean: Tensor

    window_stdev: Tensor


ConstrainingTransformation = Callable[[Tensor, Tensor, Tensor], Tensor]


class DistributionalForecaster(Protocol):
    def predict(self, x_enc: Tensor, x_mark_enc: Tensor) -> Forecast: ...


class RawForecaster(Protocol):
    def raw_heads(self, x_enc: Tensor, x_mark_enc: Tensor) -> RawForecast: ...
