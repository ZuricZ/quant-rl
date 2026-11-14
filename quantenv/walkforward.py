"""Runs a Signal walk-forward over a panel.

The environment, not the signal, decides what each step sees: `features`
gets the observable columns, `fit` gets only rows before the block (whose
targets are realised by the block's first timestamp), and `predict` gets
only the block. Every slice is zero-copy.
"""

import dataclasses
import random
from typing import Any

import numpy as np
import polars as pl

from quantenv.panel import Panel
from quantenv.signal import Signal, SignalError

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


def first_prediction(panel: Panel, wf: WalkForward, start: Any = None) -> Any:
    """The first predicted timestamp: `start`, or the one after `min_train`."""
    if start is not None:
        return start
    if wf.min_train >= len(panel.timestamps):
        raise ValueError(
            f"min_train ({wf.min_train}) leaves no timestamps to predict "
            f"out of {len(panel.timestamps)}"
        )
    return panel.timestamps[wf.min_train]


def run(
    cls: type[Signal], panel: Panel, wf: WalkForward, start: Any = None
) -> pl.Series:
    """Predicts every row from the first predicted timestamp on.

    Args:
        cls: The signal class.
        panel: The data. Rows before `start` are used only for training.
        wf: Walk-forward settings.
        start: First timestamp to predict; None means after `wf.min_train`.

    Returns:
        Float64 Series named "prediction", one value per row from the row
        offset of the first predicted timestamp to the end. NaN and infinite
        values become null.

    Raises:
        SignalError: If any step raises or returns a bad value.
    """
    schema = panel.schema
    start = first_prediction(panel, wf, start)
    engineered = _features(cls, panel.observable, schema.timestamp, schema.asset)
    labelled = engineered.with_columns(panel.frame.get_column(schema.target))

    blocks = panel.timestamps.filter(panel.timestamps >= start)
    offsets = [panel.row_offset(b) for b in blocks.gather_every(wf.refit_every)]
    offsets.append(panel.frame.height)

    out = []
    for i, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
        _seed(i + 1)
        model = cls()
        _call(model.fit, labelled.head(lo))
        _seed(i + 1)
        out.append(_predict(model, engineered.slice(lo, hi - lo)))
    return pl.concat(out)


def _features(
    cls: type[Signal], df: pl.DataFrame, timestamp: str, asset: str
) -> pl.DataFrame:
    _seed(0)
    out = _call(cls().features, df)
    if not isinstance(out, pl.DataFrame):
        raise SignalError(
            f"features must return pl.DataFrame, got {type(out).__name__}"
        )
    keys = [timestamp, asset]
    if out.height != df.height or not out.select(keys).equals(df.select(keys)):
        raise SignalError("features must keep the rows and their order")
    return out


def _predict(model: Signal, block: pl.DataFrame) -> pl.Series:
    out = _call(model.predict, block)
    try:
        if isinstance(out, pl.Expr):
            # with_columns broadcasts scalar expressions to every row.
            out = block.with_columns(out.alias(PREDICTION)).get_column(PREDICTION)
        if not isinstance(out, pl.Series):
            raise SignalError(
                f"predict must return pl.Series or pl.Expr, got {type(out).__name__}"
            )
        if out.len() != block.height:
            raise SignalError(
                f"predict must return one value per row ({block.height}), "
                f"got {out.len()}"
            )
        out = out.cast(pl.Float64).alias(PREDICTION)
    except SignalError:
        raise
    except Exception as e:
        raise SignalError(f"predict output is invalid: {e!r}") from e
    return pl.select(pl.when(out.is_finite()).then(out)).to_series()


def _call(method, df: pl.DataFrame):
    # clone() is shallow: it only stops `df[...] = ...` in agent code from
    # replacing columns of the shared frame.
    try:
        return method(df.clone())
    except SignalError:
        raise
    except Exception as e:
        raise SignalError(f"{method.__name__} raised: {e!r}") from e


def _seed(n: int) -> None:
    # Agent code may draw from either global RNG, the stdlib's or numpy's.
    # Seeding per step keeps every step's draws identical across reruns, even
    # when an earlier step drew a different amount (as on a truncated panel).
    random.seed(n)
    np.random.seed(n)
