from __future__ import annotations

import argparse
from pathlib import Path

from halo import data, registry, results, search, seeding, settings, splits
from halo.errors import ConfigurationError, ModelsFailed, SweepFailed
from halo.settings import Setting


def _add_setting(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--setting",
        default=settings.DEFAULT.name,
        choices=sorted(settings.SETTINGS),
        help="experiment setting; names both the table and the CSV, and decides "
        "which datasets and horizons are meaningful",
    )


def _add_split(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default=splits.DEFAULT.name,
        choices=sorted(splits.SPLITS),
        help="which split's database and roll-up to read",
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    _add_setting(parser)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="override the database location (default: the results directory)",
    )


def _add_scope(parser: argparse.ArgumentParser, *, redoes: str) -> None:
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="default: every dataset the chosen setting covers, cheapest first",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="default: every prediction length the chosen setting names",
    )
    parser.add_argument("--seed", type=int, default=seeding.DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"re-run {redoes} already present in the database",
    )
    _add_common(parser)


def _add_selection(parser: argparse.ArgumentParser, *, redoes: str) -> None:
    parser.add_argument("--model", required=True, choices=registry.MODEL_NAMES)
    _add_scope(parser, redoes=redoes)


DESCRIPTION = (
    "Train and search forecasting models end to end. One command fetches the "
    "data, schedules the runs against each config's parallelism number, trains, "
    "and writes the results database and the roll-up beside it -- there is "
    "nothing to run afterwards. `run` trains a model whose configuration is "
    "settled; `tune` searches one that is not, keeping every trial and "
    "publishing the best per dataset; `test` retrains every settled model and "
    "scores the test split into its own database. Jobs are meant to run from "
    "`main`."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="halo", description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="train a model over its datasets and horizons, and store results"
    )
    _add_selection(run, redoes="configs")

    tune = subparsers.add_parser(
        "tune",
        help="search a model's configuration space, keeping every trial and "
        "publishing the best per dataset",
    )
    _add_selection(tune, redoes="trials")

    test = subparsers.add_parser(
        "test",
        help="retrain every settled model and score the test split, into the "
        "test database and its roll-up",
    )
    test.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=registry.MODEL_NAMES,
        metavar="MODEL",
        help="default: every model the chosen setting knows",
    )
    _add_scope(test, redoes="configs")

    calibrate = subparsers.add_parser(
        "calibrate",
        help="measure a config's parallelism number before checking it in",
    )
    calibrate.add_argument("--model", required=True, choices=registry.MODEL_NAMES)
    calibrate.add_argument(
        "--dataset", required=True, help="must belong to the chosen setting"
    )
    calibrate.add_argument(
        "--horizon", type=int, required=True, help="must belong to the chosen setting"
    )
    calibrate.add_argument(
        "--grid-point",
        default=None,
        help="which point of a tunable model's search space to measure; a "
        "settled model has one config per dataset and horizon and needs none",
    )
    calibrate.add_argument(
        "--instances",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="instance counts to compare; must include 1 for the baseline",
    )
    calibrate.add_argument(
        "--steps",
        type=int,
        default=150,
        help="timed training steps per instance, after a warmup",
    )
    calibrate.add_argument("--seed", type=int, default=seeding.DEFAULT_SEED)
    _add_setting(calibrate)

    export = subparsers.add_parser(
        "export", help="roll the measurements up into the setting's CSV and XLSX"
    )
    _add_split(export)
    _add_common(export)

    return parser


def _setting_of(args: argparse.Namespace) -> Setting:
    return settings.SETTINGS[args.setting]


def _datasets_of(setting: Setting, chosen: list[str] | None) -> list[str]:
    if chosen is None:
        return list(setting.datasets)
    unknown = [name for name in chosen if name not in setting.datasets]
    if unknown:
        raise ConfigurationError(
            f"the {setting.name} setting does not cover "
            f"{', '.join(unknown)}. It covers {', '.join(setting.datasets)}."
        )
    return list(chosen)


def _horizons_of(setting: Setting, chosen: list[int] | None) -> list[int]:
    if chosen is None:
        return list(setting.horizons)
    unknown = [horizon for horizon in chosen if horizon not in setting.horizons]
    if unknown:
        raise ConfigurationError(
            f"the {setting.name} setting does not use horizon "
            f"{', '.join(str(h) for h in unknown)}. It uses "
            f"{', '.join(str(h) for h in setting.horizons)}."
        )
    return list(chosen)


def _run(args: argparse.Namespace) -> None:
    from halo import validate

    setting = _setting_of(args)
    validate.run(
        model_name=args.model,
        setting=setting,
        split=splits.VALIDATION,
        datasets=_datasets_of(setting, args.datasets),
        horizons=_horizons_of(setting, args.horizons),
        seed=args.seed,
        db_path=args.db_path,
        force=args.force,
    )


def _models_of(setting: Setting, chosen: list[str] | None) -> list[str]:
    known = registry.models_for(setting.name)
    if chosen is None:
        return list(known)
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise ConfigurationError(
            f"the {setting.name} setting does not know "
            f"{', '.join(unknown)}. It knows {', '.join(known)}."
        )
    return list(chosen)


def _test(args: argparse.Namespace) -> None:
    from halo import validate

    setting = _setting_of(args)
    validate.run_models(
        model_names=_models_of(setting, args.models),
        setting=setting,
        split=splits.TEST,
        datasets=_datasets_of(setting, args.datasets),
        horizons=_horizons_of(setting, args.horizons),
        seed=args.seed,
        db_path=args.db_path,
        force=args.force,
    )


def _tune(args: argparse.Namespace) -> None:
    from halo import validate

    setting = _setting_of(args)
    validate.tune(
        model_name=args.model,
        setting=setting,
        datasets=_datasets_of(setting, args.datasets),
        horizons=_horizons_of(setting, args.horizons),
        seed=args.seed,
        db_path=args.db_path,
        force=args.force,
    )


def _config_to_measure(
    setting: Setting,
    model_name: str,
    dataset: str,
    horizon: int,
    grid_point: str | None,
) -> dict:
    entry = registry.entry(setting.name, model_name)
    if grid_point is not None and not entry.has_search_space:
        raise ConfigurationError(
            f"{model_name} was never searched, so it has no grid points to "
            "measure at -- it has one configuration per dataset. Drop "
            "--grid-point."
        )
    searching = grid_point is not None and entry.has_search_space
    matches = [
        config
        for config in (entry.trials() if searching else entry.known_configs())
        if config["dataset_name"] == dataset and config["pred_len"] == horizon
    ]
    labels = sorted(
        str(config[search.GRID_POINT_KEY])
        for config in matches
        if search.GRID_POINT_KEY in config
    )
    if grid_point is not None:
        matches = [
            config
            for config in matches
            if config.get(search.GRID_POINT_KEY) == grid_point
        ]
    if not matches:
        raise ConfigurationError(
            f"{model_name} has no configuration for {dataset}/S={horizon}"
            + (f" at grid point {grid_point!r}" if grid_point is not None else "")
            + (f". It knows {', '.join(labels)}." if labels else ".")
        )
    if len(matches) > 1:
        raise ConfigurationError(
            f"{model_name} has {len(matches)} configurations for "
            f"{dataset}/S={horizon}, so --grid-point has to say which one to "
            f"measure. Choose from {', '.join(labels)}."
        )
    return matches[0]


def _calibrate(args: argparse.Namespace) -> None:
    from halo import calibrate

    if 1 not in args.instances:
        raise SystemExit("--instances must include 1: it is the baseline")

    setting = _setting_of(args)
    dataset = _datasets_of(setting, [args.dataset])[0]
    horizon = _horizons_of(setting, [args.horizon])[0]

    config = _config_to_measure(setting, args.model, dataset, horizon, args.grid_point)
    config["root_path"] = data.fetch([dataset])[dataset]

    measurements = []
    for count in sorted(set(args.instances)):
        print(f"measuring {count} instance(s) x {args.steps} steps...", flush=True)
        measurements.append(
            calibrate.measure(config, instances=count, steps=args.steps, seed=args.seed)
        )
    print(f"\n{args.model} {dataset} S={horizon}")
    print(calibrate.report(measurements))

    if calibrate.suggest_parallelism(measurements) is None:
        raise SystemExit(
            "halo: no parallelism number could be suggested -- the baseline "
            "instance died. The measurements above are still valid."
        )


def _export(args: argparse.Namespace) -> None:
    exported = results.export_tables(
        setting=_setting_of(args),
        split=splits.SPLITS[args.split],
        db_path=args.db_path,
    )
    print(exported.csv)
    print(exported.xlsx)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        {
            "run": _run,
            "tune": _tune,
            "test": _test,
            "calibrate": _calibrate,
            "export": _export,
        }[args.command](args)
    except (ConfigurationError, ModelsFailed, SweepFailed) as error:
        raise SystemExit(f"halo: {error}") from error


if __name__ == "__main__":
    from halo.cli import main as _main

    _main()
