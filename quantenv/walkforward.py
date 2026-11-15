"""Runs an Alpha walk-forward over a panel.

The environment, not the alpha, decides what each step sees: `features`
gets the observable columns, `fit` gets only rows before the block (whose
targets are realised by the block's first timestamp), and `predict` gets only
the block. Blocks are timestamp ranges; rows are selected with lazy filters.

The work is split in two. A trusted driver (`run`, `engineer`,
`predict_blocks`) owns the panel, decides the blocks and validates every
output. A session runs the agent's steps. It only ever holds the observable
columns and the targets realised before the block it is predicting: the
driver sends each block's newly realised targets just before asking for that
block. A `Runner` opens sessions: `Local` runs them in this process, and
`quantenv.sandbox.Sandbox` runs each in a fresh, isolated process.

The loop over blocks stays in Python because each block calls agent code.
"""

import contextlib
import dataclasses
import random
from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import polars as pl

from quantenv.alpha import Alpha, AlphaError
from quantenv.panel import Panel, between

PREDICTION = "prediction"
# Rank IC drops null predictions, so an alpha that predicts only where it is
# confident would be scored on its best rows alone. A run may leave at most
# this share of its rows null (NaN and inf count as null).
MAX_NULL_SHARE = 0.1


@dataclasses.dataclass(frozen=True)
class WalkForward:
    """Walk-forward settings.

    Attributes:
        refit_every: Timestamps per block; the model is refit before each.
        min_train: Timestamps used only for training when no start is given.
    """

    refit_every: int = 250
    min_train: int = 250


@dataclasses.dataclass(frozen=True)
class Block:
    """One walk-forward block: timestamps from `start` up to `end`.

    Attributes:
        index: Block number, from 0. Seeds the random state of its calls.
        start: First timestamp of the block.
        end: First timestamp after the block, or None for the last block.
    """

    index: int
    start: Any
    end: Any = None


@dataclasses.dataclass(frozen=True)
class Run:
    """Output of a walk-forward run.

    Attributes:
        start: First predicted timestamp.
        engineered: Output of `features` on the whole panel.
        predictions: Columns timestamp and "prediction" (Float64; NaN and inf
            become null), one row per panel row from `start` on.
    """

    start: Any
    engineered: pl.DataFrame
    predictions: pl.DataFrame


class Session(Protocol):
    """The agent's side of one pipeline run. Calls arrive in order:
    `features` or `use` once, then `predict` for blocks in time order."""

    def features(self, observable: pl.DataFrame) -> pl.DataFrame:
        """Calls `features` on every observable row."""

    def use(self, engineered: pl.DataFrame) -> None:
        """Takes an already validated `features` output instead."""

    def predict(self, block: Block, labels: pl.Series) -> pl.Series:
        """Fits a fresh instance on the rows before the block and predicts it.

        `labels` holds the targets realised since the previous call, in row
        order: together, the calls so far hold the target of every row
        before `block.start`.
        """


class Runner(Protocol):
    """Opens sessions for one alpha."""

    def session(self, timestamp: str) -> contextlib.AbstractContextManager[Session]:
        """A session over panels whose time column is `timestamp`."""


class Executor:
    """Runs an alpha's steps in this process; implements `Session`.

    Used directly by `Local`, and inside the sandbox's child process.

    Args:
        cls: The alpha class.
        timestamp: Time column.
    """

    def __init__(self, cls: type[Alpha], timestamp: str) -> None:
        self._cls = cls
        self._timestamp = timestamp
        self._engineered: pl.DataFrame | None = None
        self._labels: list[pl.Series] = []

    def features(self, observable: pl.DataFrame) -> pl.DataFrame:
        _seed(0)
        out = _call(self._cls().features, observable)
        if not isinstance(out, pl.DataFrame):
            raise AlphaError(
                f"features must return pl.DataFrame, got {type(out).__name__}"
            )
        self._engineered = out
        return out

    def use(self, engineered: pl.DataFrame) -> None:
        self._engineered = engineered

    def predict(self, block: Block, labels: pl.Series) -> pl.Series:
        if self._engineered is None:
            raise AlphaError("predict before features")
        self._labels.append(labels)
        ts = self._timestamp
        train = between(self._engineered, ts, end=block.start).with_columns(
            pl.concat(self._labels)
        )
        model = self._cls()
        _seed(block.index + 1)
        _call(model.fit, train)
        _seed(block.index + 1)
        rows = between(self._engineered, ts, block.start, block.end)
        out = _call(model.predict, rows)
        try:
            if isinstance(out, pl.Expr):
                # with_columns broadcasts scalar expressions to every row.
                return rows.with_columns(out.alias(PREDICTION)).get_column(PREDICTION)
        except Exception as e:
            raise AlphaError(f"predict output is invalid: {e!r}") from e
        if not isinstance(out, pl.Series):
            raise AlphaError(
                f"predict must return pl.Series or pl.Expr, got {type(out).__name__}"
            )
        return out


@dataclasses.dataclass(frozen=True)
class Local:
    """Runs sessions in this process. For trusted code and tests: the alpha
    shares memory with the caller, panel included."""

    cls: type[Alpha]

    @contextlib.contextmanager
    def session(self, timestamp: str) -> Iterator[Session]:
        yield Executor(self.cls, timestamp)


def start_timestamp(panel: Panel, wf: WalkForward, start: Any = None) -> Any:
    """The first predicted timestamp: `start`, or the one after `min_train`."""
    if start is not None:
        return start
    if wf.min_train >= len(panel.timestamps):
        raise ValueError(
            f"min_train ({wf.min_train}) leaves nothing to predict "
            f"out of {len(panel.timestamps)} timestamps"
        )
    return panel.timestamps[wf.min_train]


def blocks(panel: Panel, wf: WalkForward, start: Any) -> list[Block]:
    """The blocks from `start` to the end of the panel."""
    starts = panel.timestamps.filter(panel.timestamps >= start)
    starts = starts.gather_every(wf.refit_every).to_list()
    ends = [*starts[1:], None]
    return [Block(i, s, e) for i, (s, e) in enumerate(zip(starts, ends))]


def engineer(session: Session, panel: Panel) -> pl.DataFrame:
    """Calls `features` on the observable columns and validates the result.

    Raises:
        AlphaError: If `features` raises, or changes the rows or their order.
    """
    s = panel.schema
    df = panel.observable
    out = session.features(df)
    keys = [s.timestamp, s.asset]
    try:
        same_rows = out.height == df.height and out.select(keys).equals(df.select(keys))
    except Exception as e:
        raise AlphaError(f"features output is invalid: {e!r}") from e
    if not same_rows:
        raise AlphaError("features must keep the rows and their order")
    # Checked equal to the sorted input above, so the flag is valid; it keeps
    # the timestamp filters on their fast path.
    return out.with_columns(pl.col(s.timestamp).set_sorted())


def predict_blocks(session: Session, panel: Panel, todo: list[Block]) -> pl.Series:
    """Predicts blocks in time order, sending each one's realised targets first.

    Args:
        session: A session that has its engineered frame.
        panel: The data, targets included.
        todo: Blocks in time order.

    Returns:
        Predictions for the blocks' rows: Float64, with NaN and inf as null.

    Raises:
        AlphaError: If a step raises or returns a bad value.
    """
    s = panel.schema
    parts = []
    sent = None
    for block in todo:
        labels = between(
            panel.frame.select(s.timestamp, s.target), s.timestamp, sent, block.start
        ).get_column(s.target)
        sent = block.start
        out = session.predict(block, labels)
        height = between(
            panel.frame.select(s.timestamp), s.timestamp, block.start, block.end
        ).height
        parts.append(_validated(out, height))
    return pl.concat(parts)


def run(
    runner: Runner,
    panel: Panel,
    wf: WalkForward,
    start: Any = None,
    engineered: pl.DataFrame | None = None,
) -> Run:
    """Predicts every row from the first predicted timestamp on, in one session.

    Args:
        runner: Runs the alpha.
        panel: The data. Timestamps before `start` are used only for training.
        wf: Walk-forward settings.
        start: First timestamp to predict; None means after `wf.min_train`.
        engineered: Output of `engineer` on this panel, if already computed.

    Returns:
        The engineered frame and the predictions.

    Raises:
        AlphaError: If any step raises or returns a bad value, or more than
            `MAX_NULL_SHARE` of the predictions are null.
    """
    s = panel.schema
    start = start_timestamp(panel, wf, start)
    with runner.session(s.timestamp) as session:
        if engineered is None:
            engineered = engineer(session, panel)
        else:
            session.use(engineered)
        out = predict_blocks(session, panel, blocks(panel, wf, start))
    if out.null_count() > MAX_NULL_SHARE * out.len():
        raise AlphaError(
            f"{out.null_count() / out.len():.0%} of predictions are null; "
            f"at most {MAX_NULL_SHARE:.0%} may be"
        )
    stamps = between(panel.frame.select(s.timestamp), s.timestamp, start)
    return Run(start, engineered, stamps.with_columns(out))


def _validated(out: Any, height: int) -> pl.Series:
    """Checks one block's predictions and normalizes them to Float64."""
    if not isinstance(out, pl.Series):
        raise AlphaError(f"predict must return pl.Series, got {type(out).__name__}")
    if out.len() != height:
        raise AlphaError(
            f"predict must return one value per row ({height}), got {out.len()}"
        )
    try:
        out = out.cast(pl.Float64).alias(PREDICTION)
    except Exception as e:
        raise AlphaError(f"predict output is invalid: {e!r}") from e
    return pl.select(pl.when(out.is_finite()).then(out)).to_series()


def _call(method, df: pl.DataFrame):
    # clone() is shallow: it only stops `df[...] = ...` in agent code from
    # replacing columns of the shared frame.
    try:
        return method(df.clone())
    except AlphaError:
        raise
    except Exception as e:
        raise AlphaError(f"{method.__name__} raised: {e!r}") from e


def _seed(n: int) -> None:
    # Agent code may draw from either global RNG, the stdlib's or numpy's.
    # Seeding per step keeps every step's draws identical across reruns, even
    # when an earlier step drew a different amount (as on a truncated panel).
    random.seed(n)
    np.random.seed(n)
