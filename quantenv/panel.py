"""Long-format panels: one row per (timestamp, asset).

Nothing here knows where the data came from. A panel from the simulator and
one from real data look the same to the rest of the package. Time slices are
lazy timestamp filters, which polars runs fast on the sorted timestamp column.
"""

import dataclasses
import functools
from typing import Any

import polars as pl


@dataclasses.dataclass(frozen=True)
class Schema:
    """Column roles in a panel.

    Attributes:
        features: Feature columns given to the alpha.
        timestamp: Time column, Date or Datetime. Rows are sorted by it.
        asset: Asset identifier column.
        ret: Return realised at the row's timestamp, from the previous period
            to this one. Known at that time, so the alpha sees it.
        target: Return from this period to the next. Hidden from the alpha
            except as a training label for past rows.
    """

    features: tuple[str, ...]
    timestamp: str = "timestamp"
    asset: str = "asset"
    ret: str = "ret"
    target: str = "fwd_return"


def between(
    frame: pl.DataFrame, timestamp: str, start: Any = None, end: Any = None
) -> pl.DataFrame:
    """Rows with start <= timestamp < end; a missing bound is open.

    Args:
        frame: Frame sorted by `timestamp`.
        timestamp: Time column.
        start: Inclusive lower bound, or None.
        end: Exclusive upper bound, or None.

    Returns:
        The matching rows.
    """
    ts = pl.col(timestamp)
    if start is not None and end is not None:
        condition = ts.is_between(start, end, closed="left")
    elif start is not None:
        condition = ts >= start
    elif end is not None:
        condition = ts < end
    else:
        return frame
    return frame.lazy().filter(condition).collect()


@dataclasses.dataclass(frozen=True)
class Panel:
    """A panel and its schema.

    Attributes:
        frame: All columns, sorted by timestamp.
        schema: Column roles.

    Raises:
        ValueError: If the frame is not sorted by timestamp.
    """

    frame: pl.DataFrame
    schema: Schema

    def __post_init__(self) -> None:
        ts = self.schema.timestamp
        if not self.frame.get_column(ts).is_sorted():
            raise ValueError("panel must be sorted by timestamp")
        # Mark the column sorted so timestamp filters take polars' fast path.
        sorted_frame = self.frame.with_columns(pl.col(ts).set_sorted())
        object.__setattr__(self, "frame", sorted_frame)

    @functools.cached_property
    def observable(self) -> pl.DataFrame:
        """Everything except the target: what the alpha sees. Zero-copy."""
        return self.frame.drop(self.schema.target)

    @functools.cached_property
    def timestamps(self) -> pl.Series:
        """Distinct timestamps, ascending."""
        return self.frame.get_column(self.schema.timestamp).unique(maintain_order=True)

    def until(self, cut: Any) -> "Panel":
        """The rows at or before `cut`."""
        ts = pl.col(self.schema.timestamp)
        return Panel(self.frame.lazy().filter(ts <= cut).collect(), self.schema)

    def with_frame(self, frame: pl.DataFrame) -> "Panel":
        """The same schema over another frame."""
        return Panel(frame, self.schema)


@dataclasses.dataclass(frozen=True)
class EvalData:
    """What to evaluate an alpha on.

    Attributes:
        panel: The data. Timestamps before `start` are used only for training.
        start: First scored timestamp; None means after the walk-forward's
            `min_train` periods.
        suspects: Columns for the column dependence check.
        check_seed: Seed for the checks' random choices: where the
            truncation check cuts and how the column dependence check
            permutes. Agent code must not be able to learn it, or it can
            special-case those choices. None draws a fresh one per score.
    """

    panel: Panel
    start: Any = None
    suspects: tuple[str, ...] = ()
    check_seed: int | None = None
