from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import Size, Tensor, nn

from halo.errors import ConfigurationError
from halo.forecast import (
    ConstrainingTransformation,
    Forecast,
    RawForecast,
    RawForecaster,
)

DIRECT_ONLY_PARAMETERS = (
    "output_heads",
    "constraining_transformation",
    "point_estimate",
)

HEADS_PER_BACKBONE = 1

LOCATION_BRANCH = 0

SETTLED_BY_TORCH = ("training",)

SETTING_TYPES = (bool, int, float, str, type(None))

PROBE_ROWS = 2

PROBE_STEPS = 3


class SideBySide(nn.Module):
    def __init__(
        self,
        *,
        backbones: Sequence[nn.Module],
        constraining_transformation: ConstrainingTransformation | None = None,
        point_estimate: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        _refuse_a_model_with_no_branches(backbones)
        heads = [
            _single_head_of(backbone, branch)
            for branch, backbone in enumerate(backbones)
        ]
        _refuse_branches_that_cannot_report_their_raw_heads(backbones)
        _refuse_branches_whose_own_output_stage_would_be_dropped(backbones)
        _refuse_branches_drawn_from_more_than_one_backbone(backbones)
        _refuse_branches_that_would_standardize_unalike(backbones)
        _refuse_branches_whose_heads_would_not_stack(heads)

        self.constraining_transformation = (
            _identity
            if constraining_transformation is None
            else constraining_transformation
        )
        self.point_estimate = _first_head if point_estimate is None else point_estimate
        self.backbones = nn.ModuleList(backbones)

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor,
        x_dec: Tensor,
        x_mark_dec: Tensor,
    ) -> Tensor:
        return self.predict(x_enc, x_mark_enc).point_estimate

    def predict(self, x_enc: Tensor, x_mark_enc: Tensor) -> Forecast:
        joined = self._joined_heads(x_enc, x_mark_enc)
        parameters = self.constraining_transformation(
            joined.raw, joined.window_mean, joined.window_stdev
        )
        point = (
            self.point_estimate(parameters) * joined.window_stdev + joined.window_mean
        )
        return Forecast(parameters=parameters, point_estimate=point)

    def _joined_heads(self, x_enc: Tensor, x_mark_enc: Tensor) -> RawForecast:
        branches = [
            cast(RawForecaster, backbone).raw_heads(x_enc, x_mark_enc)
            for backbone in self.backbones
        ]
        return RawForecast(
            raw=torch.cat([branch.raw for branch in branches], dim=-1),
            window_mean=branches[LOCATION_BRANCH].window_mean,
            window_stdev=branches[LOCATION_BRANCH].window_stdev,
        )


def _refuse_a_model_with_no_branches(backbones: Sequence[nn.Module]) -> None:
    if len(backbones) < 1:
        raise ConfigurationError(
            f"{len(backbones)} backbones were given; a model has to end in at "
            "least one projection, and this one takes every projection from a "
            "network of its own."
        )


def _single_head_of(backbone: nn.Module, branch: int) -> nn.Module:
    bank = getattr(backbone, "heads", None)
    if not isinstance(bank, nn.ModuleList):
        raise ConfigurationError(
            f"branch {branch} exposes no bank of heads, so how many columns it "
            "contributes cannot be read until it has run. Every branch here "
            "stands for one column of the parameter tensor, and one that "
            "cannot say how many it produces would be found out by the shape "
            "of a forecast nothing downstream is in a position to question."
        )
    if len(bank) != HEADS_PER_BACKBONE:
        raise ConfigurationError(
            f"branch {branch} carries {len(bank)} heads, and this model builds "
            "one whole network per column, so each branch has to carry exactly "
            "one. The columns are concatenated in the order the branches are "
            "given, so a branch contributing more than one shifts every column "
            "after it: the constraint would be handed a tensor whose second "
            "column is another location rather than the scale, and the "
            "likelihood would read it as the scale regardless."
        )
    return bank[0]


def _refuse_branches_that_cannot_report_their_raw_heads(
    backbones: Sequence[nn.Module],
) -> None:
    silent = [
        branch
        for branch, backbone in enumerate(backbones)
        if not callable(getattr(backbone, "raw_heads", None))
    ]
    if silent:
        raise ConfigurationError(
            f"branches {silent} cannot report their raw heads, and every "
            "branch here is read that way: the columns are joined before the "
            "constraint, so what a branch has to offer is its head's output "
            "and the window it was standardized against, not a forecast it has "
            "already constrained and put back in the target's units. A bank of "
            "heads is not enough to tell the two apart, since several "
            "backbones alongside this one carry a bank and answer only in "
            "finished forecasts. Left to the first forward pass this is a "
            "missing attribute raised inside a training job that has already "
            "started."
        )


def _refuse_branches_whose_own_output_stage_would_be_dropped(
    backbones: Sequence[nn.Module],
) -> None:
    dropped = [
        branch
        for branch, backbone in enumerate(backbones)
        if not _leaves_its_own_head_untouched(backbone)
    ]
    if dropped:
        raise ConfigurationError(
            f"branches {dropped} carry a constraint or a point estimate of "
            "their own that changes the column they produce. This model reads "
            "its branches at their raw heads, so a branch's own output stage "
            "never runs: the constraint and the point estimate that decide "
            "what the model reports are the ones this constructor was given. "
            "Passing them to the branch instead builds a model that quietly "
            "does something else -- pass them here, or build the branches "
            "without them."
        )


def _leaves_its_own_head_untouched(backbone: nn.Module) -> bool:
    probe = torch.arange(float(PROBE_ROWS * PROBE_STEPS * HEADS_PER_BACKBONE)).reshape(
        PROBE_ROWS, PROBE_STEPS, HEADS_PER_BACKBONE
    )
    window_mean = torch.zeros(PROBE_ROWS, 1, 1)
    window_stdev = torch.ones(PROBE_ROWS, 1, 1)
    constrain = getattr(backbone, "constraining_transformation", None)
    estimate = getattr(backbone, "point_estimate", None)
    try:
        constrained = (
            probe if constrain is None else constrain(probe, window_mean, window_stdev)
        )
        estimated = probe if estimate is None else estimate(probe)
    except Exception:
        return False
    return torch.equal(constrained, probe) and torch.equal(estimated, probe)


def _refuse_branches_drawn_from_more_than_one_backbone(
    backbones: Sequence[nn.Module],
) -> None:
    location = type(backbones[LOCATION_BRANCH])
    foreign = [
        branch
        for branch, backbone in enumerate(backbones)
        if type(backbone) is not location
    ]
    if foreign:
        raise ConfigurationError(
            f"branches {foreign} are not {location.__name__} as branch "
            f"{LOCATION_BRANCH} is. This model is one backbone standing beside "
            "itself, and the whole reading of it depends on that: the only "
            "difference between it and the shared-trunk model it is measured "
            "against is meant to be where the second column comes from. Two "
            "backbones also read their window under two rules, and only the "
            "first branch's is kept."
        )


def _refuse_branches_that_would_standardize_unalike(
    backbones: Sequence[nn.Module],
) -> None:
    settings = [_settings_of(backbone) for backbone in backbones]
    disagreeing = [
        branch
        for branch, setting in enumerate(settings)
        if setting != settings[LOCATION_BRANCH]
    ]
    if disagreeing:
        raise ConfigurationError(
            f"branches {disagreeing} are configured unlike branch "
            f"{LOCATION_BRANCH}, which is {settings[LOCATION_BRANCH]}. Only the "
            "first branch's window mean and deviation are kept, and every "
            "column is de-standardized against them, so branches that read "
            "their window under different rules put the scale back in units it "
            "was never in. Nothing downstream can see the difference: the "
            "model trains, the loss falls, and the scale is wrong by whatever "
            "factor separates the two windows."
        )


def _settings_of(backbone: nn.Module) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(backbone).items()
        if not name.startswith("_")
        and name not in SETTLED_BY_TORCH
        and isinstance(value, SETTING_TYPES)
    }


def _refuse_branches_whose_heads_would_not_stack(heads: Sequence[nn.Module]) -> None:
    shapes = [_stored_shapes_of(head) for head in heads]
    disagreeing = [
        branch
        for branch, shape in enumerate(shapes)
        if shape != shapes[LOCATION_BRANCH]
    ]
    if disagreeing:
        raise ConfigurationError(
            f"the heads of branches {disagreeing} are shaped unlike branch "
            f"{LOCATION_BRANCH}'s, so the columns they produce would not stack. "
            "The concatenation refuses them a step later, naming a tensor "
            "dimension rather than the configuration that set it; one horizon "
            "here and another there is the usual way to arrive at this."
        )


def _stored_shapes_of(head: nn.Module) -> dict[str, Size]:
    return {name: tensor.shape for name, tensor in head.state_dict().items()}


def _identity(parameters: Tensor, window_mean: Tensor, window_stdev: Tensor) -> Tensor:
    return parameters


def _first_head(parameters: Tensor) -> Tensor:
    return parameters[..., :1]


def builder(
    backbone: Callable[[dict[str, Any]], nn.Module],
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
                "through alone would let a config ask for branches it has no "
                "way to interpret."
            )

        return SideBySide(
            backbones=[backbone(config) for _ in range(output_heads)],
            constraining_transformation=constraining_transformation,
            point_estimate=point_estimate,
        )

    return build
