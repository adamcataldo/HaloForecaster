from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader

from halo import validate

BATCH = 2
SEQ_LEN = 8
LABEL_LEN = 2
PRED_LEN = 3
CHANNELS = 3
DEVICE = torch.device("cpu")


class ConstantModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        return self.output


def channel_marked(length, channels, offset=0.0):
    return (torch.arange(float(channels)) + offset).expand(BATCH, length, channels)


def config_for(features):
    return {"features": features, "pred_len": PRED_LEN, "label_len": LABEL_LEN}


def make_batch(target):
    return (
        torch.zeros(BATCH, SEQ_LEN, CHANNELS),
        target,
        torch.zeros(BATCH, SEQ_LEN, 4),
        torch.zeros(BATCH, LABEL_LEN + PRED_LEN, 4),
    )


def test_the_exogenous_path_scores_the_target_column_and_only_the_target_column():
    prediction = channel_marked(PRED_LEN, CHANNELS, offset=100.0)
    target = channel_marked(LABEL_LEN + PRED_LEN, CHANNELS)

    scored, expected = validate._forward(
        ConstantModel(prediction), make_batch(target), config_for("MS"), DEVICE
    )

    assert scored.shape == (BATCH, PRED_LEN, 1)
    assert expected.shape == (BATCH, PRED_LEN, 1)
    assert torch.equal(scored, torch.full((BATCH, PRED_LEN, 1), 102.0))
    assert torch.equal(expected, torch.full((BATCH, PRED_LEN, 1), 2.0))


def test_a_single_channel_prediction_survives_the_exogenous_slice_unchanged():
    prediction = torch.full((BATCH, PRED_LEN, 1), 7.0)
    target = channel_marked(LABEL_LEN + PRED_LEN, CHANNELS)

    scored, expected = validate._forward(
        ConstantModel(prediction), make_batch(target), config_for("MS"), DEVICE
    )

    assert torch.equal(scored, prediction)
    assert torch.equal(expected, torch.full((BATCH, PRED_LEN, 1), 2.0))


def test_the_exogenous_metric_is_the_target_columns_error_not_every_columns():
    prediction = channel_marked(PRED_LEN, CHANNELS, offset=100.0)
    batch = make_batch(torch.zeros(BATCH, LABEL_LEN + PRED_LEN, CHANNELS))
    model = ConstantModel(prediction)

    loader = cast(DataLoader[Any], [batch])

    exogenous_mse, _ = validate.evaluate(
        model, loader, config_for("MS"), DEVICE, validate._forward
    )

    assert exogenous_mse == pytest.approx(102.0**2)


@pytest.mark.parametrize("features", ["M", "S", "MS ", "m"])
def test_an_unimplemented_feature_mode_is_refused(features):
    with pytest.raises(ValueError, match="not implemented"):
        validate._forward(cast(torch.nn.Module, None), (), config_for(features), DEVICE)
