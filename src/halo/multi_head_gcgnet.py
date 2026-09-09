from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from halo import gcgnet
from halo.errors import ConfigurationError
from halo.forecast import ConstrainingTransformation, Forecast
from halo.gcgnet_adapter import endogenous_first

DIRECT_ONLY_PARAMETERS = (
    "output_heads",
    "constraining_transformation",
    "point_estimate",
)

ENDOGENOUS_CHANNELS = 1

EXOGENOUS_FEATURES = "MS"

USE_FUTURE_EXOG_KEY = "use_future_exog"

STANDARDIZATION_EPSILON = 1e-5


class MultiHeadGCGNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        enc_in: int,
        series_dim: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        e_layers: int,
        rank: int,
        dropout: float,
        use_norm: bool = True,
        output_heads: int = 1,
        constraining_transformation: ConstrainingTransformation | None = None,
        point_estimate: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if seq_len % patch_len:
            raise ConfigurationError(
                f"a look-back of {seq_len} does not divide into patches of "
                f"{patch_len}. The model reads the patch length times the "
                f"number of whole patches, so the oldest {seq_len % patch_len} "
                "steps of every window would be dropped and every shape "
                "downstream would still be right."
            )
        if output_heads < 1:
            raise ConfigurationError(
                f"output_heads={output_heads}; a model has to end in at least "
                "one projection."
            )
        if series_dim > ENDOGENOUS_CHANNELS:
            raise ConfigurationError(
                f"series_dim={series_dim}, and the head bank stacks on the "
                "axis the vendored head uses for endogenous channels, so with "
                "more than one of them a head and a channel are "
                "indistinguishable. The loop slices the last axis of a "
                "forecast to find the scored column, so it would take a scale "
                "where it expected a forecast and train to a number of "
                "entirely plausible magnitude."
            )

        self.series_dim = series_dim
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.var_num = enc_in
        self.exogenous_dim = enc_in - series_dim
        self.d_model = d_model
        self.input_patch_num = seq_len // patch_len
        self.input_len = patch_len * self.input_patch_num
        self.pred_patch_num = math.ceil(pred_len / patch_len)
        self.nodes = enc_in * (self.input_patch_num + self.pred_patch_num)
        self.constraining_transformation = (
            _identity
            if constraining_transformation is None
            else constraining_transformation
        )
        self.point_estimate = _first_head if point_estimate is None else point_estimate

        vendored = gcgnet.model_module()
        self.graph_criterion = nn.L1Loss()
        self.patch_embedding = vendored.PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=patch_len,
            dropout=dropout,
        )
        self.vae = vendored.VAE(
            input_len=d_model * self.input_patch_num,
            output_len=pred_len,
            d_model=d_model,
            d_ff=d_ff,
        )
        self.graph_discriminator = vendored.GraphDiscriminator(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            nodes_num=self.nodes,
            rank=rank,
        )
        self.sparsifier = vendored.Sparsifier(n_vars=enc_in, patch_num=self.nodes)
        self.gcn = vendored.GCNStack(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.heads = nn.ModuleList(
            vendored.FlattenHead(
                n_vars=enc_in,
                endo_num=series_dim,
                d_ff=d_ff,
                nf=d_model * (self.input_patch_num + self.pred_patch_num),
                target_window=pred_len,
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
        return self.predict(x_enc).point_estimate

    def predict(self, x_enc: Tensor) -> Forecast:
        forecast, _ = self.predict_with_auxiliary(x_enc, self._absent_future(x_enc))
        return forecast

    def predict_with_auxiliary(
        self, x_enc: Tensor, future: Tensor
    ) -> tuple[Forecast, Tensor]:
        history = endogenous_first(x_enc, self.series_dim)
        rotated = endogenous_first(future[:, -self.pred_len :, :], self.series_dim)
        return self._run(
            history,
            rotated[..., : self.series_dim],
            rotated[..., self.series_dim :],
        )

    def _run(
        self,
        history: Tensor,
        endogenous_future: Tensor,
        exogenous_future: Tensor,
    ) -> tuple[Forecast, Tensor]:
        endogenous_history = history[:, -self.input_len :, : self.series_dim]
        exogenous_history = history[:, -self.input_len :, self.series_dim :]
        endogenous_future = endogenous_future[:, : self.pred_len, :]
        exogenous_future = exogenous_future[:, : self.pred_len, :]

        window_mean, window_stdev = self._statistics(endogenous_history)
        exogenous_mean, exogenous_stdev = self._statistics(exogenous_history)
        endogenous_history = _standardize(endogenous_history, window_mean, window_stdev)
        endogenous_future = _standardize(endogenous_future, window_mean, window_stdev)
        exogenous_history = _standardize(
            exogenous_history, exogenous_mean, exogenous_stdev
        )
        exogenous_future = _standardize(
            exogenous_future, exogenous_mean, exogenous_stdev
        )

        observed = torch.cat((endogenous_history, exogenous_history), dim=-1).permute(
            0, 2, 1
        )

        patched_observed, _ = self.patch_embedding(observed)
        generated, latent_mean, latent_log_variance = self.vae(
            patched_observed.reshape(
                -1, self.var_num, self.input_patch_num * self.d_model
            )
        )
        divergence = _divergence(latent_mean, latent_log_variance)

        nodes = self._nodes_of(torch.cat((observed, generated), dim=-1))
        adjacency, graph_mean, graph_log_variance = self.graph_discriminator(nodes)
        graph_divergence = _divergence(graph_mean, graph_log_variance)

        alignment: Tensor | float = 0.0
        if self.training:
            witnessed = torch.cat(
                (endogenous_future, exogenous_future), dim=-1
            ).permute(0, 2, 1)
            witnessed_adjacency, _, _ = self.graph_discriminator(
                self._nodes_of(torch.cat((observed, witnessed), dim=-1))
            )
            alignment = self.graph_criterion(adjacency, witnessed_adjacency.detach())

        adjacency, routing = self.sparsifier(adjacency, is_training=self.training)
        encoded = self.gcn(adjacency, nodes).reshape(
            -1,
            self.var_num,
            self.input_patch_num + self.pred_patch_num,
            self.d_model,
        )

        raw = torch.cat([head(encoded) for head in self.heads], dim=1).permute(0, 2, 1)
        parameters = self.constraining_transformation(raw, window_mean, window_stdev)
        point = self.point_estimate(parameters) * window_stdev + window_mean
        return (
            Forecast(parameters=parameters, point_estimate=point),
            alignment + routing + divergence + graph_divergence,
        )

    def _nodes_of(self, sequence: Tensor) -> Tensor:
        patched, _ = self.patch_embedding(sequence)
        return patched.reshape(-1, self.nodes, self.d_model)

    def _statistics(self, window: Tensor) -> tuple[Tensor, Tensor]:
        if not self.use_norm:
            zeros = torch.zeros_like(window[:, :1, :])
            return zeros, torch.ones_like(zeros)

        mean = window.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(
            window.var(dim=1, keepdim=True, correction=0) + STANDARDIZATION_EPSILON
        ).detach()
        return mean, stdev

    def _absent_future(self, x_enc: Tensor) -> Tensor:
        return x_enc.new_zeros(
            x_enc.size(0), self.pred_len, self.series_dim + self.exogenous_dim
        )


def _standardize(window: Tensor, mean: Tensor, stdev: Tensor) -> Tensor:
    return (window - mean) / stdev


def _divergence(mean: Tensor, log_variance: Tensor) -> Tensor:
    return -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())


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
                "column and covariates in the rest. Anything else asks for a "
                "forecast of every channel, and the head bank has already "
                "taken the axis those channels would come back on."
            )

        if config[USE_FUTURE_EXOG_KEY]:
            raise ConfigurationError(
                f"config asks for {USE_FUTURE_EXOG_KEY}=True, which lets "
                "GCGNet read the true future covariates over the horizon it is "
                "forecasting. That is upstream's headline setting and it is "
                "not a setting anything else in this table runs under. We run "
                "the paper's Table 3 protocol, where the future exogenous "
                "variables are unavailable and the model's own generator "
                "forecasts them -- and this model has no path that reads them "
                "at all."
            )

        return MultiHeadGCGNet(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            patch_len=config["patch_len"],
            enc_in=config["enc_in"],
            series_dim=config["series_dim"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            n_heads=config["n_heads"],
            e_layers=config["e_layers"],
            rank=config["rank"],
            dropout=config["dropout"],
            use_norm=bool(config["use_norm"]),
            output_heads=output_heads,
            constraining_transformation=constraining_transformation,
            point_estimate=point_estimate,
        ).float()

    return build
