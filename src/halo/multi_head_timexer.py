from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from halo import tslib
from halo.errors import ConfigurationError
from halo.forecast import ConstrainingTransformation, Forecast, RawForecast

DIRECT_ONLY_PARAMETERS = (
    "output_heads",
    "constraining_transformation",
    "point_estimate",
)

VENDORED_MODEL = "TimeXer"

ENDOGENOUS_CHANNELS = 1

EXOGENOUS_FEATURES = "MS"

KNOWN_ACTIVATIONS = ("relu", "gelu")

STANDARDIZATION_EPSILON = 1e-5


class MultiHeadTimeXer(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        latent_dim: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        layers: int,
        activation: str = "gelu",
        use_norm: bool = True,
        output_heads: int = 1,
        constraining_transformation: ConstrainingTransformation | None = None,
        point_estimate: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if seq_len % patch_len:
            raise ConfigurationError(
                f"a look-back of {seq_len} does not divide into patches of "
                f"{patch_len}. The patching operation would drop the final "
                f"{seq_len % patch_len} steps of every window and every shape "
                "downstream would still be right."
            )
        if output_heads < 1:
            raise ConfigurationError(
                f"output_heads={output_heads}; a model has to end in at least "
                "one projection."
            )
        if latent_dim % n_heads:
            raise ConfigurationError(
                f"a latent width of {latent_dim} does not divide among "
                f"{n_heads} attention heads. The reshape would still succeed, "
                "into a split that is wrong rather than invalid."
            )
        if activation not in KNOWN_ACTIVATIONS:
            raise ConfigurationError(
                f"activation={activation!r} is neither of the "
                f"{' nor the '.join(KNOWN_ACTIVATIONS)} the encoder layer "
                "knows. It treats anything that is not the first as the "
                "second, so a typo silently trains a different network."
            )

        self.use_norm = use_norm
        self.constraining_transformation = (
            _identity
            if constraining_transformation is None
            else constraining_transformation
        )
        self.point_estimate = _first_head if point_estimate is None else point_estimate

        timexer = tslib.model_module(VENDORED_MODEL)
        self.en_embedding = timexer.EnEmbedding(
            ENDOGENOUS_CHANNELS, latent_dim, patch_len, dropout
        )
        self.ex_embedding = timexer.DataEmbedding_inverted(
            seq_len, latent_dim, dropout=dropout
        )
        self.encoder = timexer.Encoder(
            [
                timexer.EncoderLayer(
                    timexer.AttentionLayer(
                        timexer.FullAttention(
                            False, attention_dropout=dropout, output_attention=False
                        ),
                        latent_dim,
                        n_heads,
                    ),
                    timexer.AttentionLayer(
                        timexer.FullAttention(
                            False, attention_dropout=dropout, output_attention=False
                        ),
                        latent_dim,
                        n_heads,
                    ),
                    latent_dim,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(layers)
            ],
            norm_layer=nn.LayerNorm(latent_dim),
        )
        self.heads = nn.ModuleList(
            timexer.FlattenHead(
                ENDOGENOUS_CHANNELS,
                latent_dim * (seq_len // patch_len + 1),
                pred_len,
                head_dropout=dropout,
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
        standardized, window_mean, window_stdev = self._standardize(x_enc)

        endogenous, _ = self.en_embedding(
            standardized[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        )
        exogenous = self.ex_embedding(standardized[:, :, :-1], x_mark_enc)
        encoded = self.encoder(endogenous, exogenous)
        encoded = encoded.reshape(
            -1, ENDOGENOUS_CHANNELS, encoded.shape[-2], encoded.shape[-1]
        ).permute(0, 1, 3, 2)

        raw = torch.cat([head(encoded) for head in self.heads], dim=1).permute(0, 2, 1)
        return RawForecast(raw=raw, window_mean=window_mean, window_stdev=window_stdev)

    def predict(self, x_enc: Tensor, x_mark_enc: Tensor) -> Forecast:
        heads = self.raw_heads(x_enc, x_mark_enc)
        parameters = self.constraining_transformation(
            heads.raw, heads.window_mean, heads.window_stdev
        )
        point = self.point_estimate(parameters) * heads.window_stdev + heads.window_mean
        return Forecast(parameters=parameters, point_estimate=point)

    def _standardize(self, x_enc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self.use_norm:
            zeros = torch.zeros_like(x_enc[:, :1, -1:])
            return x_enc, zeros, torch.ones_like(zeros)

        mean = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - mean
        stdev = torch.sqrt(
            centered.var(dim=1, keepdim=True, correction=0) + STANDARDIZATION_EPSILON
        ).detach()
        return centered / stdev, mean[:, :, -1:], stdev[:, :, -1:]


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
                f"short-term exogenous setting alone, which is {EXOGENOUS_FEATURES!r}: "
                "one target channel in the last column and covariates in the "
                "rest. The vendored model answers anything else on its "
                "multivariate path, where every channel is endogenous and the "
                "flattened head's trailing axis indexes channels rather than "
                "heads. This model has no such path, so it would build without "
                "complaint, forecast the last column while the loop scored the "
                "first, and train to a number of entirely plausible magnitude."
            )

        return MultiHeadTimeXer(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            patch_len=config["patch_len"],
            latent_dim=config["d_model"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            layers=config["e_layers"],
            activation=config["activation"],
            use_norm=bool(config["use_norm"]),
            output_heads=output_heads,
            constraining_transformation=constraining_transformation,
            point_estimate=point_estimate,
        ).float()

    return build
