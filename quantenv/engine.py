"""Runs agent-written signal code and scores it by rank IC."""

from __future__ import annotations

import multiprocessing as mp
import random
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantenv.generator import Dataset

SignalFn = Callable[[pd.DataFrame], Any]

RUN_SEED = 0


class SignalError(Exception):
    """Signal code failed to compile, raised, or returned a bad value."""


class SignalTimeout(SignalError):
    """Signal code ran past its time limit."""


def compile_signal(code: str) -> SignalFn:
    """Executes signal code and returns its `signal` function.

    Args:
        code: Python source defining `def signal(df) -> pd.Series`. `np` and
            `pd` are predefined.

    Returns:
        The `signal` function.

    Raises:
        SignalError: If the code fails to run or defines no callable `signal`.
    """
    namespace: dict[str, Any] = {"np": np, "pd": pd, "__name__": "signal_code"}
    try:
        exec(compile(code, "<signal>", "exec"), namespace)
    except Exception as e:
        raise SignalError(f"code failed to run: {e!r}") from e
    fn = namespace.get("signal")
    if not callable(fn):
        raise SignalError("code does not define a callable `signal`")
    return fn


def run_signal(fn: SignalFn, features: pd.DataFrame) -> np.ndarray:
    """Calls a signal on a copy of the features with fixed random seeds.

    Args:
        fn: The signal function.
        features: Feature columns indexed by (date, asset).

    Returns:
        One float per row of `features`, in its order. Missing and infinite
        values are NaN.

    Raises:
        SignalError: If the signal raises or returns the wrong shape or type.
    """
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    try:
        out = fn(features.copy())
    except Exception as e:
        raise SignalError(f"signal raised: {e!r}") from e

    if isinstance(out, pd.DataFrame) and out.shape[1] == 1:
        out = out.iloc[:, 0]
    if isinstance(out, pd.Series):
        if not out.index.equals(features.index):
            if not out.index.is_unique:
                raise SignalError("signal output has a duplicated index")
            out = out.reindex(features.index)
        values = out.to_numpy()
    else:
        values = np.asarray(out)
    if values.ndim != 1 or len(values) != len(features):
        raise SignalError(
            f"signal must return one value per row ({len(features)}), "
            f"got shape {values.shape}"
        )
    try:
        values = values.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise SignalError(f"signal output is not numeric: {e}") from e
    values[~np.isfinite(values)] = np.nan
    return values


def daily_ranks(values: np.ndarray, n_assets: int) -> np.ndarray:
    """Ranks each day's values across assets; NaN stays NaN.

    Args:
        values: One value per row, rows ordered by (date, asset).
        n_assets: Assets per day.

    Returns:
        Array of shape (days, assets) with average ranks.
    """
    return pd.DataFrame(values.reshape(-1, n_assets)).rank(axis=1).to_numpy()


def rank_ic(values: np.ndarray, fwd_returns: np.ndarray, n_assets: int) -> float:
    """Mean daily Spearman correlation between a signal and next-day returns.

    Rows where either input is NaN are dropped day by day. A day with fewer
    than two usable rows or a constant signal counts as IC 0, so the result is
    always finite.

    Args:
        values: Signal values, rows ordered by (date, asset).
        fwd_returns: Next-day returns on the same rows.
        n_assets: Assets per day.

    Returns:
        The mean daily rank IC.
    """
    s = values.reshape(-1, n_assets).copy()
    r = np.asarray(fwd_returns, dtype=np.float64).reshape(-1, n_assets).copy()
    mask = np.isnan(s) | np.isnan(r)
    s[mask] = np.nan
    r[mask] = np.nan
    rs = pd.DataFrame(s).rank(axis=1).to_numpy()
    rr = pd.DataFrame(r).rank(axis=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN days.
        rs = rs - np.nanmean(rs, axis=1, keepdims=True)
        rr = rr - np.nanmean(rr, axis=1, keepdims=True)
        cov = np.nansum(rs * rr, axis=1)
        var = np.sqrt(np.nansum(rs**2, axis=1) * np.nansum(rr**2, axis=1))
        ic = cov / var
    ic[~np.isfinite(ic)] = 0.0
    return float(ic.mean())


def score(fn: SignalFn, data: Dataset) -> float:
    """Rank IC of a signal function on a dataset split."""
    values = run_signal(fn, data.features)
    return rank_ic(values, data.fwd_returns.to_numpy(), data.n_assets)


def score_code(code: str, data: Dataset) -> float:
    """Rank IC of signal code on a dataset split."""
    return score(compile_signal(code), data)


def run_isolated(
    target: Callable[..., Any], args: tuple, timeout: float
) -> Any:
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
        SignalError: If `target` raises SignalError, or the process dies.
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
