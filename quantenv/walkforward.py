"""Runs an Alpha walk-forward over a panel.

The environment, not the alpha, decides what each step sees: `features`
gets the observable columns, `fit` gets only rows before the block (whose
targets are realised by the block's first timestamp), and `predict` gets only
the block. Blocks are timestamp ranges; rows are selected with lazy filters.

The loop over blocks stays in Python because each block calls agent code.
"""

import dataclasses
import random
from typing import Any

import numpy as np
import polars as pl

from quantenv.alpha import Alpha, AlphaError
from quantenv.panel import Panel, between

PREDICTION = "prediction"


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


def engineer(cls: type[Alpha], panel: Panel) -> pl.DataFrame:
    """Calls `features` on the observable columns and validates the result.

    Raises:
        AlphaError: If `features` raises, or changes the rows or their order.
    """
    s = panel.schema
    df = panel.observable
    _seed(0)
    out = _call(cls().features, df)
    if not isinstance(out, pl.DataFrame):
        raise AlphaError(f"features must return pl.DataFrame, got {type(out).__name__}")
    keys = [s.timestamp, s.asset]
    if out.height != df.height or not out.select(keys).equals(df.select(keys)):
        raise AlphaError("features must keep the rows and their order")
    # Checked equal to the sorted input above, so the flag is valid; it keeps
    # the timestamp filters on their fast path.
    return out.with_columns(pl.col(s.timestamp).set_sorted())


def fit_predict(
    cls: type[Alpha], labelled: pl.DataFrame, timestamp: str, target: str, block: Block
) -> pl.Series:
    """Fits a fresh instance on rows before the block and predicts the block.

    Args:
        cls: The alpha class.
        labelled: Output of `engineer` plus the target column.
        timestamp: Time column.
        target: Target column, dropped before `predict`.
        block: The block.

    Returns:
        Predictions for the block's rows.

    Raises:
        AlphaError: If `fit` or `predict` raises or returns a bad value.
    """
    model = cls()
    _seed(block.index + 1)
    _call(model.fit, between(labelled, timestamp, end=block.start))
    _seed(block.index + 1)
    rows = between(labelled, timestamp, block.start, block.end).drop(target)
    return _predict(model, rows)


def run(
    cls: type[Alpha],
    panel: Panel,
    wf: WalkForward,
    start: Any = None,
    engineered: pl.DataFrame | None = None,
) -> Run:
    """Predicts every row from the first predicted timestamp on.

    Args:
        cls: The alpha class.
        panel: The data. Timestamps before `start` are used only for training.
        wf: Walk-forward settings.
        start: First timestamp to predict; None means after `wf.min_train`.
        engineered: Output of `engineer` on this panel, if already computed.

    Returns:
        The engineered frame and the predictions.

    Raises:
        AlphaError: If any step raises or returns a bad value.
    """
    s = panel.schema
    start = start_timestamp(panel, wf, start)
    if engineered is None:
        engineered = engineer(cls, panel)
    labelled = engineered.with_columns(panel.frame.get_column(s.target))
    parts = [
        fit_predict(cls, labelled, s.timestamp, s.target, b)
        for b in blocks(panel, wf, start)
    ]
    stamps = between(panel.frame.select(s.timestamp), s.timestamp, start)
    return Run(start, engineered, stamps.with_columns(pl.concat(parts)))


def _predict(model: Alpha, block: pl.DataFrame) -> pl.Series:
    out = _call(model.predict, block)
    try:
        if isinstance(out, pl.Expr):
            # with_columns broadcasts scalar expressions to every row.
            out = block.with_columns(out.alias(PREDICTION)).get_column(PREDICTION)
        if not isinstance(out, pl.Series):
            raise AlphaError(
                f"predict must return pl.Series or pl.Expr, got {type(out).__name__}"
            )
        if out.len() != block.height:
            raise AlphaError(
                f"predict must return one value per row ({block.height}), "
                f"got {out.len()}"
            )
        out = out.cast(pl.Float64).alias(PREDICTION)
    except AlphaError:
        raise
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
