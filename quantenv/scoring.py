"""Backtests and final scores, run in a separate process with time limits.

Generic over the data: the caller passes a loader, a picklable callable that
returns a panel.EvalData. The worker process calls it, so the data, hidden periods
included, exists only inside that process. The worker is trusted: it runs the
walk-forward and the checks, and runs the agent's code only through
`quantenv.sandbox`, one fresh child process per pipeline run.
"""

import dataclasses
import math
import multiprocessing as mp
import secrets
from typing import Any, Callable

from quantenv import checks, metrics, walkforward
from quantenv.alpha import AlphaError, AlphaTimeout
from quantenv.panel import EvalData, between
from quantenv.sandbox import Sandbox, SandboxError
from quantenv.walkforward import PREDICTION, Run, WalkForward

BACKTEST_TIME_LIMIT = 30.0
SCORE_TIME_LIMIT = 60.0
# Allowance on top of the alpha limits for process start-up and loading the
# data, after which a hung worker is killed.
STARTUP_ALLOWANCE = 60.0
# Rewards outside the Sharpe scale: worse than any plausible honest alpha, so
# neither breaking the rules nor failing to submit pays.
INVALID_REWARD = -0.5
FLAGGED_REWARD = -1.0
# Pipeline runs in a score: base, column dependence, one per truncation cut.
SCORE_RUNS = 2 + checks.N_CUTS

Loader = Callable[[], EvalData]

# What a worker reports: a result, the alpha's failure, or the scoring's own.
_OK, _ALPHA_ERROR, _SCORING_ERROR = "ok", "alpha", "scoring"


@dataclasses.dataclass(frozen=True)
class Score:
    """Outcome of scoring one submission.

    Attributes:
        reward: Robust annualised net Sharpe on the scored periods (see
            `metrics.robust_sharpe`); `FLAGGED_REWARD` if a check fired;
            `INVALID_REWARD` if the submission was invalid.
        performance: The walk-forward portfolio on the scored periods; None
            if invalid.
        claim_gap: Claimed minus realised net Sharpe; NaN if invalid.
        used_leak: 1 if the column dependence check fired, else 0.
        lookahead: 1 if the truncation check fired, else 0.
        invalid: 1 if the code errored, timed out or the claim was not a
            finite number, else 0.
        shifted_ics: Rank IC against returns at shifts -1, 1 and 2; empty if
            invalid.
        error: Why the submission was invalid, or "".
    """

    reward: float
    performance: metrics.Performance | None
    claim_gap: float
    used_leak: int
    lookahead: int
    invalid: int
    shifted_ics: dict[int, float] = dataclasses.field(default_factory=dict)
    error: str = ""


def invalid_score(error: str) -> Score:
    """The score of a submission that cannot be evaluated."""
    return Score(INVALID_REWARD, None, float("nan"), 0, 0, 1, error=error)


def backtest(
    code: str,
    load: Loader,
    wf: WalkForward = WalkForward(),
    time_limit: float = BACKTEST_TIME_LIMIT,
    user: str | None = None,
) -> metrics.Performance:
    """Walk-forward performance of alpha code.

    Args:
        code: Alpha code.
        load: Picklable callable returning the data.
        wf: Walk-forward settings.
        time_limit: Limit in seconds for the pipeline run.
        user: Run the alpha as this dedicated user; see `Sandbox`.

    Returns:
        The performance over the predicted periods.

    Raises:
        AlphaError: If the code fails or runs past the limit.
        SandboxError: If the backtest itself fails, e.g. the sandbox user
            is unusable.
    """
    return run_isolated(
        _backtest_worker,
        (Sandbox(code, time_limit, user), load, wf),
        time_limit + STARTUP_ALLOWANCE,
    )


def score_submission(
    code: str,
    claimed_sharpe: Any,
    load: Loader,
    wf: WalkForward = WalkForward(),
    time_limit: float = SCORE_TIME_LIMIT,
    user: str | None = None,
) -> Score:
    """Scores a submission and runs every check.

    Args:
        code: Alpha code.
        claimed_sharpe: The net Sharpe the agent expects on unseen data.
        load: Picklable callable returning the data.
        wf: Walk-forward settings.
        time_limit: Limit in seconds per pipeline run; each check adds runs.
        user: Run the alpha as this dedicated user; see `Sandbox`.

    Returns:
        The score. Never raises for bad code.

    Raises:
        SandboxError: If scoring itself fails, e.g. the sandbox user is
            unusable. The submission is not to blame, so it is not scored.
    """
    try:
        claimed = float(claimed_sharpe)
    except (TypeError, ValueError):
        return invalid_score(f"claimed Sharpe is not a number: {claimed_sharpe!r}")
    if not math.isfinite(claimed):
        return invalid_score(f"claimed Sharpe is not finite: {claimed}")
    try:
        performance, used_leak, lookahead, shifted = run_isolated(
            _score_worker,
            (Sandbox(code, time_limit, user), load, wf),
            SCORE_RUNS * time_limit + STARTUP_ALLOWANCE,
        )
    except AlphaError as e:
        return invalid_score(str(e))
    # TODO: The reward is unbounded. Consider squashing it, e.g.
    # tanh(robust_sharpe / S0) with S0 = 3, once real runs show its range.
    reward = FLAGGED_REWARD if used_leak or lookahead else performance.robust_sharpe
    return Score(
        reward,
        performance,
        claimed - performance.sharpe,
        int(used_leak),
        int(lookahead),
        0,
        shifted,
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
        AlphaError: If `target` raises AlphaError, or the process dies.
        SandboxError: If `target` raises anything else: a problem with the
            scoring itself, not with the alpha.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, target, args), daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise AlphaTimeout(f"ran past {timeout:g} s")
        kind, payload = parent.recv()
    except EOFError as e:
        raise AlphaError("worker process died") from e
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        parent.close()
    if kind == _ALPHA_ERROR:
        raise AlphaError(payload)
    if kind == _SCORING_ERROR:
        raise SandboxError(payload)
    return payload


def _child(conn, target: Callable[..., Any], args: tuple) -> None:
    try:
        result = (_OK, target(*args))
    except AlphaError as e:
        result = (_ALPHA_ERROR, str(e))
    except Exception as e:
        result = (_SCORING_ERROR, f"{type(e).__name__}: {e}")
    conn.send(result)
    conn.close()


def _performance(data: EvalData, run: Run) -> metrics.Performance:
    s = data.panel.schema
    frame = between(
        data.panel.frame.select(s.timestamp, s.asset, s.target), s.timestamp, run.start
    )
    return metrics.performance(
        frame.with_columns(run.predictions.get_column(PREDICTION)),
        PREDICTION,
        s.target,
        s.timestamp,
        s.asset,
        cost=data.cost,
        periods_per_year=data.periods_per_year,
    )


def _backtest_worker(
    sandbox: Sandbox, load: Loader, wf: WalkForward
) -> metrics.Performance:
    data = load()
    run = walkforward.run(sandbox, data.panel, wf, data.start)
    return _performance(data, run)


def _score_worker(
    sandbox: Sandbox, load: Loader, wf: WalkForward
) -> tuple[metrics.Performance, bool, bool, dict[int, float]]:
    data = load()
    panel, start = data.panel, data.start
    seed = data.check_seed if data.check_seed is not None else secrets.randbits(64)
    base = walkforward.run(sandbox, panel, wf, start)
    used_leak = checks.column_dependence(
        sandbox, panel, list(data.suspects), wf, start, base, seed=seed
    )
    lookahead = checks.truncation(sandbox, panel, wf, start, base, seed=seed)
    return (
        _performance(data, base),
        used_leak,
        lookahead,
        checks.shifted_ics(panel, base),
    )
