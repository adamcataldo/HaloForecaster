from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from halo import (
    crosslinear_configs,
    crosslinear_sbs_configs,
    gaussian_head,
    gcgnet_configs,
    multi_head_gcgnet_configs,
    multi_head_linear_configs,
    multi_head_timexer_configs,
    multi_head_timexer_sweep_configs,
    objectives,
    pipelines,
    settings,
    timexer_configs,
    timexer_sbs_configs,
    tslib,
)
from halo.errors import ConfigurationError
from halo.splits import Split

Configs = Callable[[], list[dict[str, Any]]]


@dataclass(frozen=True)
class ModelEntry:
    build: Callable[[dict[str, Any]], torch.nn.Module]

    paper_name: str

    settled_table: Configs | None = None

    search_space: Configs | None = None

    objective: objectives.Objective = objectives.squared_error

    pipeline: pipelines.DataPipeline = pipelines.VENDORED

    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.settled_table is None and self.search_space is None:
            raise ValueError(
                "a model entry has to offer some configurations: give it a "
                "settled_table, a search_space, or both"
            )

    @property
    def is_tunable(self) -> bool:
        return self.settled_table is None

    @property
    def has_search_space(self) -> bool:
        return self.search_space is not None

    def configs(self) -> list[dict[str, Any]]:
        if self.settled_table is None:
            raise ConfigurationError(
                "this model has no chosen configuration yet -- it is still "
                "being searched. Tune it first, then check its winning values "
                "in as a config table."
            )
        return self.settled_table()

    def trials(self) -> list[dict[str, Any]]:
        if self.search_space is None:
            raise ConfigurationError(
                "this model has no search space: its configuration is a "
                "checked-in table and there is nothing to search."
            )
        return self.search_space()

    def known_configs(self) -> list[dict[str, Any]]:
        return self.trials() if self.is_tunable else self.configs()


MODEL_REGISTRY: dict[tuple[str, str], ModelEntry] = {
    (settings.SHORT_TERM_EXOGENOUS.name, timexer_configs.MODEL_NAME): ModelEntry(
        build=tslib.model_builder(timexer_configs.MODEL_NAME),
        paper_name="TimeXer",
        settled_table=timexer_configs.exogenous_configs,
    ),
    (
        settings.SHORT_TERM_EXOGENOUS.name,
        multi_head_timexer_configs.MODEL_NAME,
    ): ModelEntry(
        build=multi_head_timexer_configs.build,
        paper_name="TimeXer + dual-head Halo, untuned",
        settled_table=multi_head_timexer_configs.exogenous_configs,
        objective=gaussian_head.beta_nll_objective,
    ),
    (
        settings.SHORT_TERM_EXOGENOUS.name,
        multi_head_timexer_sweep_configs.MODEL_NAME,
    ): ModelEntry(
        build=multi_head_timexer_configs.build,
        paper_name="TimeXer + dual-head Halo",
        settled_table=multi_head_timexer_sweep_configs.exogenous_configs,
        search_space=multi_head_timexer_sweep_configs.exogenous_search_space,
        objective=gaussian_head.beta_nll_objective,
    ),
    (settings.SHORT_TERM_EXOGENOUS.name, gcgnet_configs.MODEL_NAME): ModelEntry(
        build=gcgnet_configs.build,
        paper_name="GCGNet",
        settled_table=gcgnet_configs.exogenous_configs,
        objective=gcgnet_configs.gcgnet_objective,
        device=torch.device("cpu"),
    ),
    (settings.SHORT_TERM_EXOGENOUS.name, crosslinear_configs.MODEL_NAME): ModelEntry(
        build=crosslinear_configs.build,
        paper_name="CrossLinear",
        settled_table=crosslinear_configs.exogenous_configs,
    ),
    (
        settings.SHORT_TERM_EXOGENOUS.name,
        multi_head_gcgnet_configs.MODEL_NAME,
    ): ModelEntry(
        build=multi_head_gcgnet_configs.build,
        paper_name="GCGNet + dual-head Halo",
        settled_table=multi_head_gcgnet_configs.exogenous_configs,
        search_space=multi_head_gcgnet_configs.exogenous_search_space,
        objective=multi_head_gcgnet_configs.laplace_nll_objective,
        device=multi_head_gcgnet_configs.DEVICE,
    ),
    (
        settings.SHORT_TERM_EXOGENOUS.name,
        multi_head_linear_configs.MODEL_NAME,
    ): ModelEntry(
        build=multi_head_linear_configs.build,
        paper_name="CrossLinear + dual-head Halo",
        settled_table=multi_head_linear_configs.exogenous_configs,
        search_space=multi_head_linear_configs.exogenous_search_space,
        objective=gaussian_head.beta_nll_objective,
    ),
    (
        settings.SHORT_TERM_EXOGENOUS.name,
        crosslinear_sbs_configs.MODEL_NAME,
    ): ModelEntry(
        build=crosslinear_sbs_configs.build,
        paper_name="CrossLinear + parallel Halo",
        settled_table=crosslinear_sbs_configs.exogenous_configs,
        search_space=crosslinear_sbs_configs.exogenous_search_space,
        objective=gaussian_head.beta_nll_objective,
    ),
    (
        settings.SHORT_TERM_EXOGENOUS.name,
        timexer_sbs_configs.MODEL_NAME,
    ): ModelEntry(
        build=timexer_sbs_configs.build,
        paper_name="TimeXer + parallel Halo",
        settled_table=timexer_sbs_configs.exogenous_configs,
        search_space=timexer_sbs_configs.exogenous_search_space,
        objective=gaussian_head.beta_nll_objective,
    ),
}

MODEL_NAMES: tuple[str, ...] = tuple(
    sorted({model_name for _, model_name in MODEL_REGISTRY})
)


def refuse_colliding_paper_names(
    entries: dict[tuple[str, str], ModelEntry],
) -> None:
    claimed: dict[tuple[str, str], str] = {}
    for (setting_name, model_name), model in entries.items():
        key = (setting_name, model.paper_name)
        rival = claimed.get(key)
        if rival is not None:
            raise ConfigurationError(
                f"{rival} and {model_name} both call themselves "
                f"{model.paper_name!r} in the {setting_name} setting. A paper "
                "name is a column of the test roll-up, so two models sharing "
                "one would read as a single model measured twice."
            )
        claimed[key] = model_name


refuse_colliding_paper_names(MODEL_REGISTRY)


def models_for(setting_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(model for name, model in MODEL_REGISTRY if name == setting_name)
    )


def entry(setting_name: str, model_name: str) -> ModelEntry:
    found = MODEL_REGISTRY.get((setting_name, model_name))
    if found is None:
        known = ", ".join(models_for(setting_name)) or "no models at all"
        raise ConfigurationError(
            f"{model_name} has no configurations for the {setting_name} setting, "
            f"which knows {known}."
        )
    return found


def recorded_name(setting_name: str, model_name: str, split: Split) -> str:
    if not split.paper_names:
        return model_name
    return entry(setting_name, model_name).paper_name
