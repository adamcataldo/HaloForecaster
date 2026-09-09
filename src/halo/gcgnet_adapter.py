from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from halo import gcgnet


def endogenous_first(window: Tensor, series_dim: int) -> Tensor:
    return torch.cat((window[..., -series_dim:], window[..., :-series_dim]), dim=-1)


class GCGNetAdapter(nn.Module):
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
        use_norm: bool,
        use_future_exog: bool,
    ) -> None:
        super().__init__()
        self.series_dim = series_dim
        self.pred_len = pred_len
        self.exogenous_dim = enc_in - series_dim
        self.inner = gcgnet.model_class()(
            SimpleNamespace(
                seq_len=seq_len,
                pred_len=pred_len,
                patch_len=patch_len,
                enc_in=enc_in,
                series_dim=series_dim,
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                e_layers=e_layers,
                rank=rank,
                dropout=dropout,
                use_norm=use_norm,
                use_future_exog=use_future_exog,
            )
        )

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor,
        x_dec: Tensor,
        x_mark_dec: Tensor,
    ) -> Tensor:
        forecast, _ = self._run(x_enc, self._absent_future(x_enc))
        return forecast

    def forecast_with_auxiliary(
        self, x_enc: Tensor, future: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._run(x_enc, future)

    def _run(self, x_enc: Tensor, future: Tensor) -> tuple[Tensor, Tensor]:
        history = endogenous_first(x_enc, self.series_dim)
        rotated = endogenous_first(future[:, -self.pred_len :, :], self.series_dim)
        endogenous_future = rotated[..., : self.series_dim]
        exogenous_future = rotated[..., self.series_dim :]
        return self.inner(history, exogenous_future, endogenous_future)

    def _absent_future(self, x_enc: Tensor) -> Tensor:
        return x_enc.new_zeros(
            x_enc.size(0), self.pred_len, self.series_dim + self.exogenous_dim
        )
