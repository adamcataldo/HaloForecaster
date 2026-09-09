from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from halo import crosslinear
from halo.errors import ConfigurationError
from halo.forecast import ConstrainingTransformation, Forecast, RawForecast

DIRECT_ONLY_PARAMETERS = (
    "output_heads",
    "constraining_transformation",
    "point_estimate",
)

ENDOGENOUS_CHANNELS = 1

EXOGENOUS_FEATURES = "MS"

STANDARDIZATION_EPSILON = 1e-5


class MultiHeadLinear(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        dec_in: int,
        d_model: int,
        d_ff: int,
        alpha: float,
        beta: float,
        output_heads: int = 1,
        constraining_transformation: ConstrainingTransformation | None = None,
        point_estimate: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if output_heads < 1:
            raise ConfigurationError(
                f"output_heads={output_heads}; a model has to end in at least "
                "one projection."
            )
        if dec_in < ENDOGENOUS_CHANNELS:
            raise ConfigurationError(
                f"dec_in={dec_in}, so the window carries no target channel to "
                "forecast. This model reads the last column as the target and "
                "the rest as covariates, and it builds one head bank over that "
                "single endogenous channel: the bank stacks on the axis "
                "upstream's multivariate path uses for channels, so a head and "
                "a channel would be indistinguishable, and the loop slices the "
                "last axis of a forecast to find the scored column. It would "
                "take a scale where it expected a forecast and train to a "
                "number of entirely plausible magnitude."
            )

        self.constraining_transformation = (
            _identity
            if constraining_transformation is None
            else constraining_transformation
        )
        self.point_estimate = _first_head if point_estimate is None else point_estimate

        vendored = crosslinear.model_module()
        patch_num = math.ceil(seq_len / patch_len)
        self.alpha = nn.Parameter(torch.ones([1]) * alpha)
        self.beta = nn.Parameter(torch.ones([1]) * beta)
        self.correlation_embedding = nn.Conv1d(
            dec_in, ENDOGENOUS_CHANNELS, 3, padding="same"
        )
        self.value_embedding = vendored.Patch_Embedding(
            seq_len, patch_num, patch_len, d_model, d_ff, ENDOGENOUS_CHANNELS
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, ENDOGENOUS_CHANNELS, patch_num, d_model)
        )
        self.heads = nn.ModuleList(
            vendored.De_Patch_Embedding(
                pred_len, patch_num, d_model, d_ff, ENDOGENOUS_CHANNELS
            )
            for _ in range(output_heads)
        )

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor,
        x_dec: Tensor,
        x_mark_dec: Tensor,
    ) -> Tensor:
        return self.predict(x_enc, x_mark_enc).point_estimate

    def raw_heads(self, x_enc: Tensor, x_mark_enc: Tensor) -> RawForecast:
        channels_first = x_enc.permute(0, 2, 1)
        window_mean, window_stdev = _target_statistics(channels_first)
        standardized = _standardize(channels_first)

        mixed = self.alpha * standardized[:, -ENDOGENOUS_CHANNELS:, :] + (
            1 - self.alpha
        ) * self.correlation_embedding(standardized)
        embedded = (
            self.beta * self.value_embedding(mixed)
            + (1 - self.beta) * self.pos_embedding
        )

        raw = torch.cat([head(embedded) for head in self.heads], dim=1).permute(0, 2, 1)
        return RawForecast(raw=raw, window_mean=window_mean, window_stdev=window_stdev)

    def predict(self, x_enc: Tensor, x_mark_enc: Tensor) -> Forecast:
        heads = self.raw_heads(x_enc, x_mark_enc)
        parameters = self.constraining_transformation(
            heads.raw, heads.window_mean, heads.window_stdev
        )
        point = self.point_estimate(parameters) * heads.window_stdev + heads.window_mean
        return Forecast(parameters=parameters, point_estimate=point)


def _target_statistics(channels_first: Tensor) -> tuple[Tensor, Tensor]:
    target = channels_first[:, -ENDOGENOUS_CHANNELS:, :]
    mean = target.mean(dim=-1, keepdim=True).detach()
    stdev = target.std(dim=-1, keepdim=True).detach()
    return mean, stdev


def _standardize(channels_first: Tensor) -> Tensor:
    centered = channels_first - channels_first.mean(dim=-1, keepdim=True)
    return centered / (
        channels_first.std(dim=-1, keepdim=True) + STANDARDIZATION_EPSILON
    )


def _identity(parameters: Tensor, window_mean: Tensor, window_stdev: Tensor) -> Tensor:
    return parameters


def _first_head(parameters: Tensor) -> Tensor:
    return parameters[..., :1]


def builder(
    *,
    output_heads: int = 1,
    constraining_transformation: ConstrainingTransformation | None = None,
    point_estimate: Callable[[Tensor], Tensor] | None = None,
) -> Callable[[dict[str, Any]], nn.Module]:
    def build(config: dict[str, Any]) -> nn.Module:
        carried = [name for name in DIRECT_ONLY_PARAMETERS if name in config]
        if carried:
            raise ConfigurationError(
                f"config carries {', '.join(carried)}, which the builder does "
                "not read. These are constructor parameters: build the model "
                "directly to use them. Two of the three are callables and "
                "cannot survive a config dict at all, since configs are pickled "
                "between processes and stored as JSON -- so wiring the third "
                "through alone would let a config ask for heads it has no way "
                "to interpret."
            )

        features = config["features"]
        if features != EXOGENOUS_FEATURES:
            raise ConfigurationError(
                f"features={features!r}, and this model is built for the "
                f"short-term exogenous setting alone, which is "
                f"{EXOGENOUS_FEATURES!r}: one target channel in the last "
                "column and covariates in the rest. Anything else puts the "
                "vendored model on its multivariate path, where every channel "
                "is endogenous and the head's leading axis indexes channels "
                "rather than heads. The head bank has already taken that axis, "
                "so the model would build without complaint, and the loop "
                "would slice a scale out of the last axis where it expected a "
                "forecast."
            )

        return MultiHeadLinear(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            patch_len=config["patch_len"],
            dec_in=config["dec_in"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            alpha=config["alpha"],
            beta=config["beta"],
            output_heads=output_heads,
            constraining_transformation=constraining_transformation,
            point_estimate=point_estimate,
        ).float()

    return build
