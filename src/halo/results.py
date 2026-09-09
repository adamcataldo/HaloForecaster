from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halo import paths, workbook
from halo.errors import ConfigurationError
from halo.settings import Setting
from halo.splits import Split

RUN_COLUMNS = (
    "model_name",
    "dataset_name",
    "pred_len",
    "MSE",
    "MAE",
    "best_epoch",
    "seed",
)
TRIAL_COLUMNS = (
    "model_name",
    "dataset_name",
    "pred_len",
    "grid_point",
    "MSE",
    "MAE",
    "best_epoch",
    "seed",
    "config",
)


@dataclass(frozen=True)
class Run:
    model_name: str
    dataset_name: str
    pred_len: int
    mse: float
    mae: float
    best_epoch: int
    seed: int


@dataclass(frozen=True)
class Trial:
    model_name: str
    dataset_name: str
    pred_len: int
    grid_point: str
    mse: float
    mae: float
    best_epoch: int
    seed: int
    config: dict[str, Any]


@dataclass(frozen=True)
class Result:
    model_name: str
    dataset_name: str
    mse: float
    mae: float


@dataclass(frozen=True)
class Exported:
    csv: Path
    xlsx: Path


def default_db_path(split: Split) -> Path:
    return paths.db_path(split)


def table_for(setting: Setting) -> str:
    return f"{setting.name}_runs"


def trials_table_for(setting: Setting) -> str:
    return f"{setting.name}_trials"


def csv_path_for(
    setting: Setting, split: Split, directory: Path | str | None = None
) -> Path:
    base = Path(directory) if directory is not None else paths.results_dir()
    return base / f"{setting.name}_{split.file_prefix}results.csv"


def xlsx_path_for(
    setting: Setting, split: Split, directory: Path | str | None = None
) -> Path:
    return csv_path_for(setting, split, directory).with_suffix(".xlsx")


def _quote(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum():
        raise ConfigurationError(
            f"unsafe setting name: {identifier!r}. A setting names a SQL table "
            "and a filename, so it may contain only letters, digits and "
            "underscores."
        )
    return f'"{identifier}"'


def validate_setting(setting: Setting) -> None:
    _quote(table_for(setting))
    _quote(trials_table_for(setting))


def database_exists(split: Split, db_path: Path | str | None = None) -> bool:
    return _resolve_db_path(db_path, split).exists()


def _resolve_db_path(db_path: Path | str | None, split: Split) -> Path:
    return Path(db_path) if db_path is not None else default_db_path(split)


def _connect_for_write(db_path: Path | str | None, split: Split) -> sqlite3.Connection:
    path = _resolve_db_path(db_path, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _connect_for_read(
    db_path: Path | str | None, split: Split
) -> sqlite3.Connection | None:
    path = _resolve_db_path(db_path, split)
    if not path.exists():
        return None
    return sqlite3.connect(path)


def _create_table(conn: sqlite3.Connection, setting: Setting) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote(table_for(setting))} (
            model_name   TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            pred_len     INTEGER NOT NULL,
            MSE          REAL NOT NULL,
            MAE          REAL NOT NULL,
            best_epoch   INTEGER NOT NULL,
            seed         INTEGER NOT NULL,
            PRIMARY KEY (model_name, dataset_name, pred_len)
        )
        """
    )


def _create_trials_table(conn: sqlite3.Connection, setting: Setting) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote(trials_table_for(setting))} (
            model_name   TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            pred_len     INTEGER NOT NULL,
            grid_point   TEXT NOT NULL,
            MSE          REAL NOT NULL,
            MAE          REAL NOT NULL,
            best_epoch   INTEGER NOT NULL,
            seed         INTEGER NOT NULL,
            config       TEXT NOT NULL,
            PRIMARY KEY (model_name, dataset_name, pred_len, grid_point)
        )
        """
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    found = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return found is not None


def _check_columns(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...], db_path: Path
) -> None:
    present = {row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})")}
    missing = set(columns) - present
    if missing:
        raise ConfigurationError(
            f"{db_path} has an out-of-date {table} table: missing "
            f"{', '.join(sorted(missing))}. Everything here is a record of "
            "something that was trained, so rebuild the database rather than "
            "migrating it."
        )


def roll_up(runs: list[Run], setting: Setting) -> list[Result]:
    by_pair: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        if run.pred_len in setting.horizons:
            by_pair.setdefault((run.model_name, run.dataset_name), []).append(run)

    rolled = []
    seeds_by_pair: dict[tuple[str, str], int] = {}
    for (model_name, dataset_name), group in by_pair.items():
        if {run.pred_len for run in group} != set(setting.horizons):
            continue
        seeds = {run.seed for run in group}
        if len(seeds) > 1:
            print(
                f"{model_name}/{dataset_name}: horizons disagree on seed "
                f"({', '.join(str(s) for s in sorted(seeds))}), so it stays out "
                "of the CSV until the rest are re-run",
                flush=True,
            )
            continue
        seeds_by_pair[(model_name, dataset_name)] = seeds.pop()
        rolled.append(
            Result(
                model_name=model_name,
                dataset_name=dataset_name,
                mse=sum(run.mse for run in group) / len(group),
                mae=sum(run.mae for run in group) / len(group),
            )
        )

    rolled.extend(_cross_dataset_averages(rolled, seeds_by_pair, setting))
    return sorted(
        rolled,
        key=lambda r: (
            r.model_name,
            r.dataset_name == setting.average_name,
            r.dataset_name,
        ),
    )


def _cross_dataset_averages(
    rolled: list[Result],
    seeds_by_pair: dict[tuple[str, str], int],
    setting: Setting,
) -> list[Result]:
    if setting.average_name is None:
        return []

    by_model: dict[str, dict[str, Result]] = {}
    for result in rolled:
        by_model.setdefault(result.model_name, {})[result.dataset_name] = result

    averages = []
    for model_name, published in by_model.items():
        group = [published[name] for name in setting.datasets if name in published]
        if len(group) != len(setting.datasets):
            continue
        seeds = {seeds_by_pair[(model_name, result.dataset_name)] for result in group}
        if len(seeds) > 1:
            print(
                f"{model_name}/{setting.average_name}: {len(setting.datasets)} "
                f"datasets disagree on seed "
                f"({', '.join(str(s) for s in sorted(seeds))}), so the average "
                "stays out of the CSV until the rest are re-run",
                flush=True,
            )
            continue
        averages.append(
            Result(
                model_name=model_name,
                dataset_name=setting.average_name,
                mse=sum(result.mse for result in group) / len(group),
                mae=sum(result.mae for result in group) / len(group),
            )
        )
    return averages


def record_run(
    run: Run,
    *,
    setting: Setting,
    split: Split,
    db_path: Path | str | None = None,
    csv_dir: Path | str | None = None,
) -> Result | None:
    if not (math.isfinite(run.mse) and math.isfinite(run.mae)):
        raise ValueError(
            f"refusing to store a non-finite result for {run.dataset_name}"
            f"/S={run.pred_len}: MSE={run.mse}, MAE={run.mae}. A diverged run is"
            " a failed run, not a measurement."
        )

    with _connect_for_write(db_path, split) as conn:
        _create_table(conn, setting)
        conn.execute(
            f"""
            INSERT INTO {_quote(table_for(setting))}
                (model_name, dataset_name, pred_len, MSE, MAE, best_epoch, seed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, dataset_name, pred_len) DO UPDATE SET
                MSE = excluded.MSE,
                MAE = excluded.MAE,
                best_epoch = excluded.best_epoch,
                seed = excluded.seed
            """,
            (
                run.model_name,
                run.dataset_name,
                run.pred_len,
                run.mse,
                run.mae,
                run.best_epoch,
                run.seed,
            ),
        )

    export_tables(setting=setting, split=split, db_path=db_path, csv_dir=csv_dir)

    complete = roll_up(
        read_runs(
            model_name=run.model_name,
            dataset_name=run.dataset_name,
            setting=setting,
            split=split,
            db_path=db_path,
        ),
        setting,
    )
    return next(
        (result for result in complete if result.dataset_name == run.dataset_name),
        None,
    )


def delete_run(
    *,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    setting: Setting,
    split: Split,
    db_path: Path | str | None = None,
    csv_dir: Path | str | None = None,
) -> bool:
    conn = _connect_for_read(db_path, split)
    if conn is None:
        return False
    with conn:
        if not _table_exists(conn, table_for(setting)):
            return False
        cursor = conn.execute(
            f"""
            DELETE FROM {_quote(table_for(setting))}
            WHERE model_name = ? AND dataset_name = ? AND pred_len = ?
            """,
            (model_name, dataset_name, pred_len),
        )
        removed = cursor.rowcount > 0
    if removed:
        export_tables(setting=setting, split=split, db_path=db_path, csv_dir=csv_dir)
    return removed


def record_trial(
    trial: Trial,
    *,
    setting: Setting,
    split: Split,
    db_path: Path | str | None = None,
) -> None:
    if not (math.isfinite(trial.mse) and math.isfinite(trial.mae)):
        raise ValueError(
            f"refusing to store a non-finite trial for {trial.dataset_name}"
            f"/S={trial.pred_len}/{trial.grid_point}: MSE={trial.mse}, "
            f"MAE={trial.mae}. A diverged trial is a failed trial, not a "
            "measurement -- and one stored here would be eligible to win."
        )

    with _connect_for_write(db_path, split) as conn:
        _create_trials_table(conn, setting)
        conn.execute(
            f"""
            INSERT INTO {_quote(trials_table_for(setting))}
                (model_name, dataset_name, pred_len, grid_point,
                 MSE, MAE, best_epoch, seed, config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, dataset_name, pred_len, grid_point) DO UPDATE SET
                MSE = excluded.MSE,
                MAE = excluded.MAE,
                best_epoch = excluded.best_epoch,
                seed = excluded.seed,
                config = excluded.config
            """,
            (
                trial.model_name,
                trial.dataset_name,
                trial.pred_len,
                trial.grid_point,
                trial.mse,
                trial.mae,
                trial.best_epoch,
                trial.seed,
                json.dumps(trial.config, sort_keys=True, default=str),
            ),
        )


def read_trials(
    *,
    setting: Setting,
    split: Split,
    model_name: str | None = None,
    dataset_name: str | None = None,
    pred_len: int | None = None,
    db_path: Path | str | None = None,
) -> list[Trial]:
    where = []
    params: list[object] = []
    if model_name is not None:
        where.append("model_name = ?")
        params.append(model_name)
    if dataset_name is not None:
        where.append("dataset_name = ?")
        params.append(dataset_name)
    if pred_len is not None:
        where.append("pred_len = ?")
        params.append(pred_len)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    table = _quote(trials_table_for(setting))

    conn = _connect_for_read(db_path, split)
    if conn is None:
        return []
    resolved = _resolve_db_path(db_path, split)
    with conn:
        if not _table_exists(conn, trials_table_for(setting)):
            return []
        _check_columns(conn, trials_table_for(setting), TRIAL_COLUMNS, resolved)
        rows = conn.execute(
            f"""
            SELECT model_name, dataset_name, pred_len, grid_point,
                   MSE, MAE, best_epoch, seed, config
            FROM {table}
            {clause}
            ORDER BY model_name, dataset_name, pred_len, grid_point
            """,
            params,
        ).fetchall()
    return [
        Trial(
            model_name=row[0],
            dataset_name=row[1],
            pred_len=row[2],
            grid_point=row[3],
            mse=row[4],
            mae=row[5],
            best_epoch=row[6],
            seed=row[7],
            config=json.loads(row[8]),
        )
        for row in rows
    ]


def check_schema(
    *, setting: Setting, split: Split, db_path: Path | str | None = None
) -> None:
    validate_setting(setting)
    conn = _connect_for_read(db_path, split)
    if conn is None:
        return
    resolved = _resolve_db_path(db_path, split)
    with conn:
        if _table_exists(conn, table_for(setting)):
            _check_columns(conn, table_for(setting), RUN_COLUMNS, resolved)
        if _table_exists(conn, trials_table_for(setting)):
            _check_columns(conn, trials_table_for(setting), TRIAL_COLUMNS, resolved)


def read_runs(
    *,
    setting: Setting,
    split: Split,
    model_name: str | None = None,
    dataset_name: str | None = None,
    db_path: Path | str | None = None,
) -> list[Run]:
    where = []
    params: list[object] = []
    if model_name is not None:
        where.append("model_name = ?")
        params.append(model_name)
    if dataset_name is not None:
        where.append("dataset_name = ?")
        params.append(dataset_name)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    table = _quote(table_for(setting))

    conn = _connect_for_read(db_path, split)
    if conn is None:
        return []
    resolved = _resolve_db_path(db_path, split)
    with conn:
        if not _table_exists(conn, table_for(setting)):
            return []
        _check_columns(conn, table_for(setting), RUN_COLUMNS, resolved)
        rows = conn.execute(
            f"""
            SELECT model_name, dataset_name, pred_len, MSE, MAE, best_epoch, seed
            FROM {table}
            {clause}
            ORDER BY model_name, dataset_name, pred_len
            """,
            params,
        ).fetchall()
    return [Run(*row) for row in rows]


METRICS = ("MSE", "MAE")


@dataclass(frozen=True)
class Grid:
    models: list[str]
    labels: list[str]
    cells: dict[tuple[str, str], tuple[float, float]]


def _grid(rolled: list[Result], setting: Setting) -> Grid:
    labels = list(setting.datasets)
    if setting.average_name is not None:
        labels.append(setting.average_name)
    return Grid(
        models=sorted({result.model_name for result in rolled}),
        labels=labels,
        cells={
            (result.model_name, result.dataset_name): (result.mse, result.mae)
            for result in rolled
        },
    )


def _grid_rows(grid: Grid) -> list[list[str]]:
    rows = [
        ["", *(cell for model in grid.models for cell in (model, ""))],
        ["", *list(METRICS) * len(grid.models)],
    ]
    for label in grid.labels:
        row = [label]
        for model in grid.models:
            pair = grid.cells.get((model, label))
            row += ["", ""] if pair is None else [str(value) for value in pair]
        rows.append(row)
    return rows


def _temp_beside(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    return Path(name)


def export_tables(
    *,
    setting: Setting,
    split: Split,
    db_path: Path | str | None = None,
    csv_dir: Path | str | None = None,
) -> Exported:
    validate_setting(setting)

    if not database_exists(split, db_path):
        resolved = _resolve_db_path(db_path, split)
        raise ConfigurationError(
            f"no results database at {resolved}, so there is nothing to roll "
            "up. Refusing to touch the CSV: an empty table here would look "
            "exactly like having lost the results. Check "
            f"${paths.RESULTS_DIR_ENV_VAR} points where you think it does."
        )

    if csv_dir is not None:
        directory = Path(csv_dir)
    elif db_path is not None:
        directory = Path(db_path).parent
    else:
        directory = paths.results_dir()
    exported = Exported(
        csv=csv_path_for(setting, split, directory),
        xlsx=xlsx_path_for(setting, split, directory),
    )
    directory.mkdir(parents=True, exist_ok=True)

    rolled = roll_up(read_runs(setting=setting, split=split, db_path=db_path), setting)
    grid = _grid(rolled, setting)

    csv_temp = _temp_beside(exported.csv)
    xlsx_temp = _temp_beside(exported.xlsx)
    try:
        with csv_temp.open("w", newline="") as handle:
            csv.writer(handle).writerows(_grid_rows(grid))
        workbook.write_grid(
            xlsx_temp,
            models=grid.models,
            labels=grid.labels,
            metrics=METRICS,
            cells=grid.cells,
            sheet_title=setting.name,
        )
        os.replace(csv_temp, exported.csv)
        os.replace(xlsx_temp, exported.xlsx)
    except BaseException:
        csv_temp.unlink(missing_ok=True)
        xlsx_temp.unlink(missing_ok=True)
        raise
    return exported
