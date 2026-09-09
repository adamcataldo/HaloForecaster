from __future__ import annotations


class ConfigurationError(RuntimeError):
    pass


class SweepFailed(RuntimeError):
    def __init__(self, failures: list[tuple[str, int]]) -> None:
        self.failures = failures
        listed = ", ".join(f"{name}/S={horizon}" for name, horizon in failures)
        super().__init__(
            f"{len(failures)} trial(s) failed: {listed}. Their datasets are "
            "absent from the CSV; re-run to fill them in."
        )


class ModelsFailed(RuntimeError):
    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        self.failures = failures
        listed = "; ".join(f"{name}: {error}" for name, error in failures)
        super().__init__(
            f"{len(failures)} model(s) failed. {listed} The models that "
            "succeeded are in the CSV; re-run to fill in the rest."
        )
