from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import time
from dataclasses import dataclass
from typing import Any

WARMUP_STEPS = 10

POLL_SECONDS = 0.5

JOIN_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Measurement:
    instances: int
    rates: list[float]

    died: int = 0

    @property
    def complete(self) -> bool:
        return self.died == 0 and len(self.rates) == self.instances

    @property
    def mean_rate(self) -> float:
        return sum(self.rates) / len(self.rates)

    @property
    def total_rate(self) -> float:
        return sum(self.rates)


def _worker(config: dict[str, Any], steps: int, seed: int, out: Any) -> None:
    import torch

    from halo import pipelines, registry, seeding, validate

    seeding.seed_everything(seed)
    entry = registry.entry(config["setting"], config["model"])
    device = validate.select_device(entry)
    train_loader = entry.pipeline.build_loader(config, seed, pipelines.TRAIN_FLAG)
    model = entry.build(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()

    started = 0.0
    done = 0
    for index, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        prediction, target = entry.pipeline.forward(model, batch, config, device)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        if index == WARMUP_STEPS:
            if device.type == "mps":
                torch.mps.synchronize()
            started = time.time()
        elif index > WARMUP_STEPS:
            done += 1
            if done >= steps:
                break
    if device.type == "mps":
        torch.mps.synchronize()
    out.put(done / (time.time() - started))


def measure(
    config: dict[str, Any], *, instances: int, steps: int, seed: int
) -> Measurement:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    processes = [
        context.Process(target=_worker, args=(config, steps, seed + i, queue))
        for i in range(instances)
    ]
    for process in processes:
        process.start()

    rates: list[float] = []
    try:
        while len(rates) < instances:
            try:
                rates.append(queue.get(timeout=POLL_SECONDS))
                continue
            except queue_module.Empty:
                pass
            if all(not process.is_alive() for process in processes):
                try:
                    while True:
                        rates.append(queue.get_nowait())
                except queue_module.Empty:
                    pass
                break
    finally:
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(timeout=JOIN_TIMEOUT_SECONDS)

    died = instances - len(rates)
    if died:
        exit_codes = [p.exitcode for p in processes if p.exitcode not in (0, None)]
        print(
            f"  {died} of {instances} instance(s) died without reporting "
            f"(exit codes {exit_codes or 'unknown'}) -- the machine cannot carry "
            f"{instances} of this config at once",
            flush=True,
        )
    return Measurement(instances=instances, rates=rates, died=died)


def _baseline(measurements: list[Measurement]) -> float | None:
    solo = next((m for m in measurements if m.instances == 1), None)
    if solo is None or not solo.rates:
        return None
    return solo.mean_rate


def suggest_parallelism(measurements: list[Measurement]) -> int | None:
    baseline = _baseline(measurements)
    if baseline is None:
        return None
    best = 1
    for measurement in sorted(measurements, key=lambda m: m.instances):
        if not measurement.complete:
            break
        if measurement.total_rate / baseline >= 1.0:
            best = max(best, measurement.instances)
    return best


def report(measurements: list[Measurement]) -> str:
    baseline = _baseline(measurements)
    lines = [
        f"{'N':>3}  {'per-instance steps/s':>22}  {'aggregate':>9}  verdict",
    ]
    for measurement in sorted(measurements, key=lambda m: m.instances):
        rates = ", ".join(f"{r:.2f}" for r in measurement.rates) or "-"
        if not measurement.complete:
            lines.append(
                f"{measurement.instances:>3}  {rates:>22}  {'-':>9}  "
                f"{measurement.died} died -- too many for this machine"
            )
            continue
        if baseline is None:
            lines.append(f"{measurement.instances:>3}  {rates:>22}  {'-':>9}  -")
            continue
        aggregate = measurement.total_rate / baseline
        verdict = "worth it" if aggregate >= 1.0 else "slower than sequential"
        lines.append(
            f"{measurement.instances:>3}  {rates:>22}  {aggregate:>9.2f}  {verdict}"
        )

    suggestion = suggest_parallelism(measurements)
    if suggestion is None:
        lines.append(
            "\nno suggestion: the N=1 baseline produced no rate, and every "
            "aggregate is a ratio against it. Re-run the baseline alone."
        )
    else:
        lines.append(f"\nsuggested parallelism: {suggestion}")
    return "\n".join(lines)
