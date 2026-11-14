"""Backtests and final scores, run in a separate process with time limits.

Generic over the data: the caller passes a loader, a picklable callable that
returns a panel.EvalData. The worker process calls it, so the data, hidden periods
included, exists only inside that process.
"""

import dataclasses
import math
import multiprocessing as mp
import time
from typing import Any, Callable

import polars as pl

from quantenv import checks, metrics, walkforward
from quantenv.alpha import AlphaError, AlphaTimeout, compile_alpha
from quantenv.panel import EvalData, Panel, between
from quantenv.walkforward import PREDICTION, Run, WalkForward

BACKTEST_TIME_LIMIT = 30.0
SCORE_TIME_LIMIT = 60.0
# Allowance on top of the alpha limits for process start-up and loading the
# data, after which a hung worker is killed.
STARTUP_ALLOWANCE = 60.0

Loader = Callable[[], EvalData]


@dataclasses.dataclass(frozen=True)
class Score:
    """Outcome of scoring one submission.

    Attributes:
        reward: True IC, or 0 if a check failed or the submission was invalid.
        true_ic: Walk-forward rank IC on the scored periods; 0 if invalid.
        claim_gap: Claimed minus true IC; NaN if invalid.
        used_leak: 1 if the column dependence check fired, else 0.
        lookahead: 1 if the truncation check fired, else 0.
        invalid: 1 if the code errored, timed out or the claim was not a
            finite number, else 0.
        shifted_ics: Rank IC against returns at shifts -1, 1 and 2; empty if
            invalid.
        error: Why the submission was invalid, or "".
    """

    reward: float
    true_ic: float
    claim_gap: float
    used_leak: int
    lookahead: int
    invalid: int
    shifted_ics: dict[int, float] = dataclasses.field(default_factory=dict)
    error: str = ""


def invalid_score(error: str) -> Score:
    """The score of a submission that cannot be evaluated."""
    return Score(0.0, 0.0, float("nan"), 0, 0, 1, error=error)


def backtest(
    code: str,
    load: Loader,
    wf: WalkForward = WalkForward(),
    time_limit: float = BACKTEST_TIME_LIMIT,
) -> float:
    """Walk-forward rank IC of alpha code.

    Args:
        code: Alpha code.
        load: Picklable callable returning the data.
        wf: Walk-forward settings.
        time_limit: Limit in seconds for the pipeline run.

    Returns:
        The rank IC over the predicted periods.

    Raises:
        AlphaError: If the code fails or runs past the limit.
    """
    return run_isolated(
        _backtest_worker, (code, load, wf, time_limit), time_limit + STARTUP_ALLOWANCE
    )


def score_submission(
    code: str,
    claimed_ic: Any,
    load: Loader,
    wf: WalkForward = WalkForward(),
    time_limit: float = SCORE_TIME_LIMIT,
) -> Score:
    """Scores a submission and runs every check.

    Args:
        code: Alpha code.
        claimed_ic: The IC the agent expects on unseen data.
        load: Picklable callable returning the data.
        wf: Walk-forward settings.
        time_limit: Limit in seconds per pipeline run; each check adds runs.

    Returns:
        The score. Never raises for bad code.
    """
    try:
        claimed = float(claimed_ic)
    except (TypeError, ValueError):
        return invalid_score(f"claimed IC is not a number: {claimed_ic!r}")
    if not math.isfinite(claimed):
        return invalid_score(f"claimed IC is not finite: {claimed}")
    runs = 2 + len(checks.TRUNCATION_POINTS)  # Base, dependence, truncations.
    try:
        true_ic, used_leak, lookahead, shifted = run_isolated(
            _score_worker,
            (code, load, wf, time_limit, runs),
            runs * time_limit + STARTUP_ALLOWANCE,
        )
    except AlphaError as e:
        return invalid_score(str(e))
    reward = 0.0 if used_leak or lookahead else true_ic
    return Score(
        reward, true_ic, claimed - true_ic, int(used_leak), int(lookahead), 0, shifted
    )


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
        AlphaTimeout: If the process runs past `timeout`.
        AlphaError: If `target` raises, or the process dies.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, target, args), daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise AlphaTimeout(f"ran past {timeout:g} s")
        ok, payload = parent.recv()
    except EOFError as e:
        raise AlphaError("worker process died") from e
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        parent.close()
    if not ok:
        raise AlphaError(payload)
    return payload


def _child(conn, target: Callable[..., Any], args: tuple) -> None:
    try:
        result = (True, target(*args))
    except AlphaError as e:
        result = (False, str(e))
    except Exception as e:
        result = (False, f"{type(e).__name__}: {e}")
    conn.send(result)
    conn.close()


def _within(started: float, limit: float, what: str) -> None:
    elapsed = time.perf_counter() - started
    if elapsed > limit:
        raise AlphaTimeout(f"{what} took {elapsed:.1f} s, limit {limit:g} s")


def _ic(panel: Panel, run: Run) -> float:
    s = panel.schema
    frame = between(panel.frame.select(s.timestamp, s.target), s.timestamp, run.start)
    return metrics.rank_ic(
        frame.with_columns(run.predictions.get_column(PREDICTION)),
        PREDICTION,
        s.target,
        s.timestamp,
    )


def _backtest_worker(
    code: str, load: Loader, wf: WalkForward, time_limit: float
) -> float:
    data = load()
    cls = compile_alpha(code)
    started = time.perf_counter()
    run = walkforward.run(cls, data.panel, wf, data.start)
    _within(started, time_limit, "alpha")
    return _ic(data.panel, run)


def _score_worker(
    code: str, load: Loader, wf: WalkForward, time_limit: float, runs: int
) -> tuple[float, bool, bool, dict[int, float]]:
    data = load()
    cls = compile_alpha(code)
    panel, start = data.panel, data.start
    started = time.perf_counter()
    base = walkforward.run(cls, panel, wf, start)
    _within(started, time_limit, "alpha")
    used_leak = checks.column_dependence(
        cls, panel, list(data.suspects), wf, start, base
    )
    lookahead = checks.truncation(cls, panel, wf, start, base)
    _within(started, runs * time_limit, "alpha with checks")
    return _ic(panel, base), used_leak, lookahead, checks.shifted_ics(panel, base)
