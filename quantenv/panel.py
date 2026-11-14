"""Long-format panels: one row per (timestamp, asset).

Nothing here knows where the data came from. A panel from the simulator and
one from real data look the same to the rest of the package.
"""

import dataclasses
import functools
from typing import Any

import polars as pl


@dataclasses.dataclass(frozen=True)
class Schema:
    """Column roles in a panel.

    Attributes:
        features: Feature columns given to the signal.
        timestamp: Time column, Date or Datetime. Rows are sorted by it.
        asset: Asset identifier column.
        ret: Return realised at the row's timestamp, from the previous period
            to this one. Known at that time, so the signal sees it.
        target: Return from this period to the next. Hidden from the signal
            except as a training label for past rows.
    """

    features: tuple[str, ...]
    timestamp: str = "timestamp"
    asset: str = "asset"
    ret: str = "ret"
    target: str = "fwd_return"


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
        if not self.frame.get_column(self.schema.timestamp).is_sorted():
            raise ValueError("panel must be sorted by timestamp")

    @functools.cached_property
    def observable(self) -> pl.DataFrame:
        """Everything except the target: what the signal sees. Zero-copy."""
        return self.frame.drop(self.schema.target)

    @functools.cached_property
    def timestamps(self) -> pl.Series:
        """Distinct timestamps, ascending."""
        return self.frame.get_column(self.schema.timestamp).unique(maintain_order=True)

    def row_offset(self, timestamp: Any) -> int:
        """Index of the first row at or after `timestamp`."""
        column = self.frame.get_column(self.schema.timestamp)
        return column.search_sorted(timestamp, side="left")

    def until(self, timestamp: Any) -> "Panel":
        """The rows at or before `timestamp`, as a zero-copy slice."""
        column = self.frame.get_column(self.schema.timestamp)
        return Panel(
            self.frame.head(column.search_sorted(timestamp, side="right")), self.schema
        )

    def with_frame(self, frame: pl.DataFrame) -> "Panel":
        """The same schema over another frame."""
        return Panel(frame, self.schema)
