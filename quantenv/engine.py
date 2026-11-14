"""Runs agent-written signal code and scores it by rank IC."""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any, Callable

import numpy as np
import polars as pl

from quantenv.generator import ASSET, FWD_RETURN, TIMESTAMP, Dataset

SignalFn = Callable[[pl.DataFrame], Any]

SIGNAL = "signal"
RUN_SEED = 0


class SignalError(Exception):
    """Signal code failed to compile, raised, or returned a bad value."""


class SignalTimeout(SignalError):
    """Signal code ran past its time limit."""


def compile_signal(code: str) -> SignalFn:
    """Executes signal code and returns its `signal` function.

    Args:
        code: Python source defining `def signal(df) -> pl.Series | pl.Expr`.
            `pl` is predefined.

    Returns:
        The `signal` function.

    Raises:
        SignalError: If the code fails to run or defines no callable `signal`.
    """
    namespace: dict[str, Any] = {"pl": pl, "__name__": "signal_code"}
    try:
        exec(compile(code, "<signal>", "exec"), namespace)
    except Exception as e:
        raise SignalError(f"code failed to run: {e!r}") from e
    fn = namespace.get(SIGNAL)
    if not callable(fn):
        raise SignalError("code does not define a callable `signal`")
    return fn


def run_signal(fn: SignalFn, features: pl.DataFrame) -> pl.Series:
    """Calls a signal with fixed random seeds and validates its output.

    The signal may return a Series with one value per row, in row order, or
    an expression, which is evaluated on `features` (scalars broadcast).

    Args:
        fn: The signal function.
        features: Columns timestamp, asset and the features.

    Returns:
        Float64 Series named "signal", one value per row; NaN and infinite
        values become null.

    Raises:
        SignalError: If the signal raises or returns the wrong type or length.
    """
    # Two separate global RNGs that agent code may draw from: the stdlib's and
    # numpy's. The leak rule needs both identical across its two runs.
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    try:
        # clone() is shallow: it only stops `df[...] = ...` in the signal
        # from replacing columns of the shared frame.
        out = fn(features.clone())
        if isinstance(out, pl.Expr):
            out = features.with_columns(out.alias(SIGNAL)).get_column(SIGNAL)
        if not isinstance(out, pl.Series):
            raise SignalError(
                f"signal must return pl.Series or pl.Expr, got {type(out).__name__}"
            )
        if out.len() != features.height:
            raise SignalError(
                f"signal must return one value per row ({features.height}), "
                f"got {out.len()}"
            )
        out = out.cast(pl.Float64)
    except SignalError:
        raise
    except Exception as e:
        raise SignalError(f"signal failed: {e!r}") from e
    return pl.select(pl.when(out.is_finite()).then(out).alias(SIGNAL)).to_series()


def rank_ic(signal: pl.Series, data: Dataset) -> float:
    """Mean per-period Spearman correlation between a signal and next-period returns.

    Null rows are dropped period by period. A period with no usable rows or a constant
    signal counts as IC 0, so the result is always finite.

    Args:
        signal: Output of `run_signal` on `data.features`.
        data: The dataset split.

    Returns:
        The mean per-period rank IC.
    """
    return (
        data.frame.select(TIMESTAMP, FWD_RETURN)
        .with_columns(signal.alias(SIGNAL))
        .lazy()
        .drop_nulls(SIGNAL)
        .group_by(TIMESTAMP)
        .agg(ic=pl.corr(SIGNAL, FWD_RETURN, method="spearman"))
        .select(pl.col("ic").fill_nan(0.0).sum() / data.n_periods)
        .collect()
        .item()
    )


def feature_ics(data: Dataset) -> dict[str, float]:
    """Rank IC of every feature column, in one pass.

    Args:
        data: The dataset split.

    Returns:
        Feature name to rank IC.
    """
    names = data.features.drop(TIMESTAMP, ASSET).columns
    row = (
        data.frame.lazy()
        .group_by(TIMESTAMP)
        .agg(pl.corr(n, FWD_RETURN, method="spearman") for n in names)
        .select(pl.col(names).fill_nan(0.0).mean())
        .collect()
    )
    return row.row(0, named=True)


def score(fn: SignalFn, data: Dataset) -> float:
    """Rank IC of a signal function on a dataset split."""
    return rank_ic(run_signal(fn, data.features), data)


def score_code(code: str, data: Dataset) -> float:
    """Rank IC of signal code on a dataset split."""
    return score(compile_signal(code), data)


def run_isolated(target: Callable[..., Any], args: tuple, timeout: float) -> Any:
    """Runs a function in a fresh process and returns its result.

    The process is killed if it runs past `timeout`. `target` must be a
    module-level function so that it can be pickled.

    Args:
        target: Function to call.
        args: Positional arguments, all picklable.
        timeout: Wall-clock limit in seconds.

    Returns:
        Whatever `target` returns.

    Raises:
        SignalTimeout: If the process runs past `timeout`.
        SignalError: If `target` raises, or the process dies.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, target, args), daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise SignalTimeout(f"ran past {timeout:g} s")
        ok, payload = parent.recv()
    except EOFError as e:
        raise SignalError("worker process died") from e
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        parent.close()
    if not ok:
        raise SignalError(payload)
    return payload


def _child(conn, target: Callable[..., Any], args: tuple) -> None:
    try:
        result = (True, target(*args))
    except SignalError as e:
        result = (False, str(e))
    except Exception as e:
        result = (False, f"{type(e).__name__}: {e}")
    conn.send(result)
    conn.close()
