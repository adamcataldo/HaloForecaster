from __future__ import annotations

from dataclasses import dataclass

from halo.data import DATASETS


@dataclass(frozen=True)
class Setting:
    name: str

    datasets: tuple[str, ...]

    horizons: tuple[int, ...]

    seq_len: int

    label_len: int

    average_name: str | None = None

    def __post_init__(self) -> None:
        unknown = [name for name in self.datasets if name not in DATASETS]
        if unknown:
            raise ValueError(
                f"{self.name} names datasets that do not exist: {', '.join(unknown)}"
            )
        if self.average_name in self.datasets:
            raise ValueError(
                f"{self.name} labels its average {self.average_name!r}, which "
                "is also one of its datasets"
            )


SHORT_TERM_EXOGENOUS = Setting(
    name="short_term_exogenous",
    datasets=("NP", "PJM", "BE", "FR", "DE"),
    horizons=(24,),
    seq_len=168,
    label_len=48,
    average_name="Avg",
)

SETTINGS: dict[str, Setting] = {
    setting.name: setting for setting in (SHORT_TERM_EXOGENOUS,)
}

DEFAULT = SHORT_TERM_EXOGENOUS
