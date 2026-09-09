from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import numpy as np
import torch
from ray.tune import Callback
from torch import nn
from torch.utils.data import DataLoader

from halo import (
    data,
    paths,
    pipelines,
    registry,
    results,
    scheduling,
    search,
    seeding,
    splits,
    tslib,
    vendor,
)
from halo.errors import ModelsFailed, SweepFailed
from halo.settings import Setting
from halo.splits import Split

CPUS_PER_TRIAL = 1

PROGRESS_EVERY_STEPS = 100


def _coordinates_of(config: dict[str, Any]) -> str:
    where = f"{config['dataset_name']}/S={config['pred_len']}"
    point = config.get(search.GRID_POINT_KEY)
    return f"{where}/{point}" if point else where


def select_device(entry: registry.ModelEntry) -> torch.device:
    if entry.device is not None:
        return entry.device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_forward = pipelines.vendored_forward


class _PooledError:
    def __init__(self) -> None:
        self.squared = 0.0
        self.absolute = 0.0
        self.count = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = (prediction - target).detach().cpu().to(torch.float64)
        self.squared += float((error * error).sum())
        self.absolute += float(error.abs().sum())
        self.count += error.numel()

    def metrics(self) -> tuple[float, float]:
        if self.count == 0:
            raise ValueError("no errors accumulated")
        return self.squared / self.count, self.absolute / self.count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    forward: pipelines.Forward,
) -> tuple[float, float]:
    model.eval()
    pooled = _PooledError()
    for batch in loader:
        prediction, target = forward(model, batch, config, device)
        pooled.update(prediction, target)
    model.train()
    return pooled.metrics()


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to("cpu", copy=True)
        for name, parameter in model.state_dict().items()
    }


def run_trial(
    config: dict[str, Any],
    seed: int = seeding.DEFAULT_SEED,
    split: Split = splits.VALIDATION,
) -> dict[str, Any]:
    label = f"{config['model']}/{_coordinates_of(config)}"
    trial_seed = seeding.derive_seed(
        seed,
        seeding.seed_identity_of(config),
        config["dataset_name"],
        config["pred_len"],
    )
    seeding.seed_everything(trial_seed)

    entry = registry.entry(config["setting"], config["model"])
    device = select_device(entry)
    train_loader = entry.pipeline.build_loader(config, trial_seed, pipelines.TRAIN_FLAG)
    val_loader = entry.pipeline.build_loader(
        config, trial_seed, splits.VALIDATION.loader_flag
    )

    model = entry.build(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    objective = entry.objective
    adjust_learning_rate = tslib.adjust_learning_rate()
    args = tslib.as_namespace(config)

    best_mse = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    best_weights: dict[str, torch.Tensor] | None = None
    epochs_since_best = 0
    started = time.time()

    steps_per_epoch = len(train_loader)
    for epoch in range(config["train_epochs"]):
        model.train()
        train_losses = []
        epoch_started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = objective(model, batch, config, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if step % PROGRESS_EVERY_STEPS == 0:
                rate = step / (time.time() - epoch_started)
                print(
                    f"[{label}] epoch {epoch + 1} step {step}/{steps_per_epoch} "
                    f"({rate:.1f} steps/s)",
                    flush=True,
                )

        val_mse, val_mae = evaluate(
            model, val_loader, config, device, entry.pipeline.forward
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        print(
            f"[{label}] epoch {epoch + 1}/{config['train_epochs']} "
            f"train {np.mean(train_losses):.7f} "
            f"val MSE {val_mse:.7f} val MAE {val_mae:.7f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

        if val_mse <= best_mse:
            best_mse, best_mae, best_epoch = val_mse, val_mae, epoch + 1
            if split != splits.VALIDATION:
                best_weights = _snapshot(model)
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= config["patience"]:
                print(f"[{label}] early stopping", flush=True)
                break

        adjust_learning_rate(optimizer, epoch + 1, args)

    if best_epoch == 0 or not (math.isfinite(best_mse) and math.isfinite(best_mae)):
        raise RuntimeError(
            f"{label} produced no finite validation loss in "
            f"{config['train_epochs']} epochs (best MSE {best_mse}); "
            "treating it as a failed trial rather than a result."
        )

    reported_mse, reported_mae = best_mse, best_mae
    if best_weights is not None:
        model.load_state_dict(best_weights)
        reported_mse, reported_mae = evaluate(
            model,
            entry.pipeline.build_loader(config, trial_seed, split.loader_flag),
            config,
            device,
            entry.pipeline.forward,
        )
        if not (math.isfinite(reported_mse) and math.isfinite(reported_mae)):
            raise RuntimeError(
                f"{label} scored a non-finite {split.name} loss at epoch "
                f"{best_epoch} (MSE {reported_mse}); treating it as a failed "
                "trial rather than a result."
            )

    alongside = (
        ""
        if split == splits.VALIDATION
        else f", val MSE {best_mse:.7f} val MAE {best_mae:.7f}"
    )
    print(
        f"[{label}] finished in {time.time() - started:.0f}s, "
        f"best epoch {best_epoch}, "
        f"{split.name} MSE {reported_mse:.7f} {split.name} MAE {reported_mae:.7f}"
        f"{alongside}",
        flush=True,
    )
    return {
        "dataset_name": config["dataset_name"],
        "pred_len": config["pred_len"],
        "mse": reported_mse,
        "mae": reported_mae,
        "best_epoch": best_epoch,
        "seed": seed,
        "val_mse": best_mse,
        "val_mae": best_mae,
    }


def _trainable(tune_config: dict[str, Any]) -> None:
    from ray import tune as ray_tune

    ray_tune.report(
        run_trial(
            tune_config["trial"],
            seed=tune_config["seed"],
            split=splits.SPLITS[tune_config["split"]],
        )
    )


class _RecordRun(Callback):
    def __init__(
        self,
        *,
        model_name: str,
        setting: Setting,
        split: Split,
        db_path: Path | None,
    ) -> None:
        self.model_name = model_name
        self.setting = setting
        self.split = split
        self.db_path = db_path
        self.stored: list[results.Run] = []
        self.completed: list[results.Result] = []
        self.failures: list[tuple[str, int]] = []

    def on_trial_result(self, iteration, trials, trial, result, **info) -> None:
        if "mse" not in result:
            return
        run = results.Run(
            model_name=registry.recorded_name(
                self.setting.name, self.model_name, self.split
            ),
            dataset_name=result["dataset_name"],
            pred_len=result["pred_len"],
            mse=result["mse"],
            mae=result["mae"],
            best_epoch=result["best_epoch"],
            seed=result["seed"],
        )
        self.stored.append(run)
        alongside = (
            ""
            if self.split == splits.VALIDATION
            else f" [val MSE {result['val_mse']:.5f} MAE {result['val_mae']:.5f}]"
        )
        print(
            f"done {run.dataset_name}/S={run.pred_len}: "
            f"{self.split.name} MSE {run.mse:.5f} MAE {run.mae:.5f} "
            f"(best epoch {run.best_epoch}){alongside}",
            flush=True,
        )
        rolled = results.record_run(
            run, setting=self.setting, split=self.split, db_path=self.db_path
        )
        if rolled is not None:
            self.completed.append(rolled)
            print(
                f"  -> {rolled.dataset_name} complete: "
                f"MSE {rolled.mse:.4f} MAE {rolled.mae:.4f} (in the CSV)",
                flush=True,
            )

    def on_trial_error(self, iteration, trials, trial, **info) -> None:
        config = trial.config.get("trial", {})
        dataset_name = config.get("dataset_name", "?")
        pred_len = config.get("pred_len", 0)
        self.failures.append((dataset_name, pred_len))
        print(
            f"FAILED {dataset_name}/S={pred_len} -- its dataset stays out of the CSV",
            flush=True,
        )


class _RecordTrial(Callback):
    def __init__(
        self,
        *,
        model_name: str,
        setting: Setting,
        db_path: Path | None,
    ) -> None:
        self.model_name = model_name
        self.setting = setting
        self.db_path = db_path
        self.stored: list[results.Trial] = []
        self.failures: list[tuple[str, int]] = []

    def on_trial_result(self, iteration, trials, trial, result, **info) -> None:
        if "mse" not in result:
            return
        config = trial.config["trial"]
        record = results.Trial(
            model_name=self.model_name,
            dataset_name=result["dataset_name"],
            pred_len=result["pred_len"],
            grid_point=search.grid_point_of(config),
            mse=result["mse"],
            mae=result["mae"],
            best_epoch=result["best_epoch"],
            seed=result["seed"],
            config=config,
        )
        self.stored.append(record)
        print(
            f"done {_coordinates_of(config)}: "
            f"MSE {record.mse:.5f} MAE {record.mae:.5f} "
            f"(best epoch {record.best_epoch})",
            flush=True,
        )
        results.record_trial(
            record,
            setting=self.setting,
            split=splits.VALIDATION,
            db_path=self.db_path,
        )

    def on_trial_error(self, iteration, trials, trial, **info) -> None:
        config = trial.config.get("trial", {})
        dataset_name = config.get("dataset_name", "?")
        pred_len = config.get("pred_len", 0)
        point = config.get(search.GRID_POINT_KEY, "?")
        self.failures.append((f"{dataset_name}/{point}", pred_len))
        print(
            f"FAILED {dataset_name}/S={pred_len}/{point} -- its dataset cannot "
            "pick a winner until this trial is re-run",
            flush=True,
        )


def _pending(
    configs: list[dict[str, Any]],
    *,
    recorded: dict[tuple[Any, ...], int],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
    seed: int,
    force: bool,
    superseded: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    todo = []
    for config in configs:
        stored_seed = recorded.get(key(config))
        if stored_seed == seed and not force:
            print(
                f"skipping {_coordinates_of(config)}: "
                "already recorded (use --force to redo)",
                flush=True,
            )
            continue
        if stored_seed is not None:
            reason = (
                f"recorded at seed {stored_seed}, asked for {seed}"
                if stored_seed != seed
                else "recorded at this seed, redone because --force"
            )
            print(f"re-running {_coordinates_of(config)}: {reason}", flush=True)
            if superseded is not None:
                superseded(config)
        todo.append(config)
    return todo


def pending(
    configs: list[dict[str, Any]],
    *,
    model_name: str,
    setting: Setting,
    split: Split,
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    return _pending(
        configs,
        recorded={
            (run.dataset_name, run.pred_len): run.seed
            for run in results.read_runs(
                model_name=registry.recorded_name(setting.name, model_name, split),
                setting=setting,
                split=split,
                db_path=db_path,
            )
        },
        key=lambda config: (config["dataset_name"], config["pred_len"]),
        seed=seed,
        force=force,
    )


def pending_trials(
    configs: list[dict[str, Any]],
    *,
    model_name: str,
    setting: Setting,
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    recorded_trials = results.read_trials(
        model_name=model_name,
        setting=setting,
        split=splits.VALIDATION,
        db_path=db_path,
    )
    published = {
        (run.dataset_name, run.pred_len): (run.mse, run.mae, run.best_epoch, run.seed)
        for run in results.read_runs(
            model_name=model_name,
            setting=setting,
            split=splits.VALIDATION,
            db_path=db_path,
        )
    }
    derived = {
        (trial.dataset_name, trial.pred_len)
        for trial in recorded_trials
        if published.get((trial.dataset_name, trial.pred_len))
        == (trial.mse, trial.mae, trial.best_epoch, trial.seed)
    }

    def withdraw_published_row(config: dict[str, Any]) -> None:
        if (config["dataset_name"], config["pred_len"]) not in derived:
            return
        if results.delete_run(
            model_name=model_name,
            dataset_name=config["dataset_name"],
            pred_len=config["pred_len"],
            setting=setting,
            split=splits.VALIDATION,
            db_path=db_path,
        ):
            print(
                f"withdrawing the published {config['dataset_name']}"
                f"/S={config['pred_len']} row: the trials it was chosen from "
                "are being redone",
                flush=True,
            )

    return _pending(
        configs,
        recorded={
            (trial.dataset_name, trial.pred_len, trial.grid_point): trial.seed
            for trial in recorded_trials
        },
        key=lambda config: (
            config["dataset_name"],
            config["pred_len"],
            search.grid_point_of(config),
        ),
        seed=seed,
        force=force,
        superseded=withdraw_published_row,
    )


def _fit(
    configs: list[dict[str, Any]],
    *,
    callback: Callback,
    seed: int,
    split: Split = splits.VALIDATION,
) -> None:
    import ray
    from ray import tune as ray_tune
    from ray.tune.execution.placement_groups import PlacementGroupFactory

    budget, costs = scheduling.slot_costs(configs)
    print(f"{len(configs)} runs, slot budget {budget}", flush=True)
    for config, cost in zip(configs, costs, strict=True):
        print(
            f"  {_coordinates_of(config):28s} "
            f"P={scheduling.parallelism_of(config)} cost={cost}",
            flush=True,
        )

    def trial_resources(tune_config: dict[str, Any]) -> PlacementGroupFactory:
        cost = budget // scheduling.parallelism_of(tune_config["trial"])
        return PlacementGroupFactory(
            [{"CPU": CPUS_PER_TRIAL, scheduling.SLOT_RESOURCE: cost}]
        )

    ray.init(
        ignore_reinit_error=True,
        resources={scheduling.SLOT_RESOURCE: budget},
        runtime_env={"env_vars": {vendor.VENDOR_DIR_ENV_VAR: str(vendor.vendor_dir())}},
    )
    try:
        tuner = ray_tune.Tuner(
            ray_tune.with_resources(_trainable, resources=trial_resources),
            param_space={
                "trial": ray_tune.grid_search(configs),
                "seed": seed,
                "split": split.name,
            },
            run_config=ray_tune.RunConfig(callbacks=[callback]),
        )
        tuner.fit()
    finally:
        ray.shutdown()


def sweep(
    configs: list[dict[str, Any]],
    *,
    model_name: str,
    setting: Setting,
    split: Split,
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
) -> list[results.Result]:
    callback = _RecordRun(
        model_name=model_name, setting=setting, split=split, db_path=db_path
    )
    _fit(configs, callback=callback, seed=seed, split=split)
    if callback.failures:
        raise SweepFailed(callback.failures)
    return callback.completed


def record_winners(
    trials: list[dict[str, Any]],
    *,
    model_name: str,
    setting: Setting,
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
) -> list[results.Result]:
    expected: dict[tuple[str, int], set[str]] = {}
    for config in trials:
        key = (config["dataset_name"], config["pred_len"])
        expected.setdefault(key, set()).add(search.grid_point_of(config))

    completed: list[results.Result] = []
    for (dataset_name, pred_len), labels in expected.items():
        best_trial = search.winner(
            results.read_trials(
                model_name=model_name,
                dataset_name=dataset_name,
                pred_len=pred_len,
                setting=setting,
                split=splits.VALIDATION,
                db_path=db_path,
            ),
            expected=labels,
            seed=seed,
        )
        if best_trial is None:
            continue
        print(
            f"{dataset_name}/S={pred_len}: {best_trial.grid_point} wins "
            f"{len(labels)} configurations with "
            f"MSE {best_trial.mse:.5f} MAE {best_trial.mae:.5f}",
            flush=True,
        )
        rolled = results.record_run(
            results.Run(
                model_name=model_name,
                dataset_name=dataset_name,
                pred_len=pred_len,
                mse=best_trial.mse,
                mae=best_trial.mae,
                best_epoch=best_trial.best_epoch,
                seed=best_trial.seed,
            ),
            setting=setting,
            split=splits.VALIDATION,
            db_path=db_path,
        )
        if rolled is not None:
            completed.append(rolled)
            print(
                f"  -> {rolled.dataset_name} complete: "
                f"MSE {rolled.mse:.4f} MAE {rolled.mae:.4f} (in the CSV)",
                flush=True,
            )
    return completed


def _report(*, setting: Setting, split: Split, db_path: Path | None) -> None:
    if not results.database_exists(split, db_path):
        print("\nno results database yet; nothing to export", flush=True)
        return
    exported = results.export_tables(setting=setting, split=split, db_path=db_path)
    where = db_path if db_path is not None else results.default_db_path(split)
    print(f"\n{results.table_for(setting)} @ {where}", flush=True)
    rolled = results.roll_up(
        results.read_runs(setting=setting, split=split, db_path=db_path), setting
    )
    for row in rolled:
        print(
            f"  {row.model_name:32s} {row.dataset_name:8s} "
            f"MSE {row.mse:.4f}  MAE {row.mae:.4f}",
            flush=True,
        )
    print(f"CSV: {exported.csv}", flush=True)
    print(f"XLSX: {exported.xlsx}", flush=True)


def _prepare(
    *, setting: Setting, split: Split, datasets: list[str], db_path: Path | None
) -> None:
    if db_path is None:
        paths.results_dir()
    results.check_schema(setting=setting, split=split, db_path=db_path)
    tslib.ensure_available()
    data.ensure_sources(datasets)


def _run_one(
    *,
    model_name: str,
    setting: Setting,
    split: Split,
    datasets: list[str],
    horizons: list[int],
    seed: int,
    db_path: Path | None,
    force: bool,
) -> tuple[list[results.Result], SweepFailed | None]:
    selected = [
        config
        for config in registry.entry(setting.name, model_name).configs()
        if config["dataset_name"] in datasets and config["pred_len"] in horizons
    ]
    todo = pending(
        selected,
        model_name=model_name,
        seed=seed,
        setting=setting,
        split=split,
        db_path=db_path,
        force=force,
    )
    if not todo:
        print("nothing to run; every selected config is already recorded", flush=True)
        return [], None

    roots = data.fetch(sorted({c["dataset_name"] for c in todo}))
    for config in todo:
        config["root_path"] = roots[config["dataset_name"]]

    try:
        return (
            sweep(
                todo,
                model_name=model_name,
                seed=seed,
                setting=setting,
                split=split,
                db_path=db_path,
            ),
            None,
        )
    except SweepFailed as error:
        return [], error


def run(
    *,
    model_name: str,
    setting: Setting,
    split: Split,
    datasets: list[str],
    horizons: list[int],
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
    force: bool = False,
) -> list[results.Result]:
    _prepare(setting=setting, split=split, datasets=datasets, db_path=db_path)
    completed, failure = _run_one(
        model_name=model_name,
        setting=setting,
        split=split,
        datasets=datasets,
        horizons=horizons,
        seed=seed,
        db_path=db_path,
        force=force,
    )
    _report(setting=setting, split=split, db_path=db_path)
    if failure is not None:
        raise failure
    return completed


def run_models(
    *,
    model_names: list[str],
    setting: Setting,
    split: Split,
    datasets: list[str],
    horizons: list[int],
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
    force: bool = False,
) -> list[results.Result]:
    _prepare(setting=setting, split=split, datasets=datasets, db_path=db_path)

    completed: list[results.Result] = []
    failures: list[tuple[str, Exception]] = []
    for model_name in model_names:
        print(f"\n=== {model_name} on the {split.name} split ===", flush=True)
        try:
            finished, failure = _run_one(
                model_name=model_name,
                setting=setting,
                split=split,
                datasets=datasets,
                horizons=horizons,
                seed=seed,
                db_path=db_path,
                force=force,
            )
        except Exception as error:
            print(f"[{model_name}] failed: {error}", flush=True)
            failures.append((model_name, error))
            continue
        completed.extend(finished)
        if failure is not None:
            failures.append((model_name, failure))

    _report(setting=setting, split=split, db_path=db_path)
    if failures:
        raise ModelsFailed(failures)
    return completed


def tune(
    *,
    model_name: str,
    setting: Setting,
    datasets: list[str],
    horizons: list[int],
    seed: int = seeding.DEFAULT_SEED,
    db_path: Path | None = None,
    force: bool = False,
) -> list[results.Result]:
    _prepare(
        setting=setting, split=splits.VALIDATION, datasets=datasets, db_path=db_path
    )

    selected = [
        config
        for config in registry.entry(setting.name, model_name).trials()
        if config["dataset_name"] in datasets and config["pred_len"] in horizons
    ]
    todo = pending_trials(
        selected,
        model_name=model_name,
        seed=seed,
        setting=setting,
        db_path=db_path,
        force=force,
    )

    failure: SweepFailed | None = None
    if todo:
        roots = data.fetch(sorted({c["dataset_name"] for c in todo}))
        for config in todo:
            config["root_path"] = roots[config["dataset_name"]]

        callback = _RecordTrial(model_name=model_name, setting=setting, db_path=db_path)
        _fit(todo, callback=callback, seed=seed)
        if callback.failures:
            failure = SweepFailed(callback.failures)
    else:
        print("nothing to run; every selected trial is already recorded", flush=True)

    completed = record_winners(
        selected,
        model_name=model_name,
        setting=setting,
        seed=seed,
        db_path=db_path,
    )
    _report(setting=setting, split=splits.VALIDATION, db_path=db_path)
    if failure is not None:
        raise failure
    return completed
