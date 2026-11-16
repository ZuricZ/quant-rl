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
import time
from typing import Any, Callable

from quantenv import checks, metrics, walkforward
from quantenv.alpha import AlphaError, AlphaTimeout
from quantenv.panel import EvalData, between
from quantenv.sandbox import Sandbox, SandboxError
from quantenv.walkforward import PREDICTION, Run, WalkForward

# Allowance for process start-up and loading the data, and again on top of
# the alpha's runs, after which a hung worker is killed.
STARTUP_ALLOWANCE = 60.0

Loader = Callable[[], EvalData]
# What a worker calls to set its own deadline, in seconds from now.
Allow = Callable[[float], None]

# What a worker reports: a new deadline, a result, the alpha's failure, or the
# scoring's own.
_ALLOW, _OK, _ALPHA_ERROR, _SCORING_ERROR = "allow", "ok", "alpha", "scoring"


@dataclasses.dataclass(frozen=True)
class TimeLimit:
    """Wall-clock limit of one pipeline run, scaled with the data.

    The limit grows with the rows the run hands to agent code
    (`walkforward.pipeline_rows`), so an alpha that fits in its backtest
    limit and costs the same per row also fits in the scored run's. See the
    spec, Decision 20, for the measurements behind the defaults.

    Attributes:
        per_million_rows: Seconds per million rows.
        minimum: Floor in seconds, for start-up, which does not scale.
    """

    per_million_rows: float = 5.0
    minimum: float = 30.0

    def seconds(self, rows: int) -> float:
        """The limit of a run that hands `rows` rows to agent code."""
        return max(self.minimum, self.per_million_rows * rows / 1e6)


@dataclasses.dataclass(frozen=True)
class Score:
    """Raw outcome of scoring one submission. Carries no reward: the host
    turns it into one with `quantenv.reward.reward`.

    Attributes:
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

    performance: metrics.Performance | None
    claim_gap: float
    used_leak: int
    lookahead: int
    invalid: int
    shifted_ics: dict[int, float] = dataclasses.field(default_factory=dict)
    error: str = ""


def invalid_score(error: str) -> Score:
    """The score of a submission that cannot be evaluated."""
    return Score(None, float("nan"), 0, 0, 1, error=error)


def backtest(
    code: str,
    load: Loader,
    wf: WalkForward = WalkForward(),
    limit: TimeLimit = TimeLimit(),
    user: str | None = None,
) -> metrics.Performance:
    """Walk-forward performance of alpha code.

    Args:
        code: Alpha code.
        load: Picklable callable returning the data.
        wf: Walk-forward settings.
        limit: Time limit of the pipeline run.
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
        (_Pipeline(code, user, limit, STARTUP_ALLOWANCE), load, wf),
        STARTUP_ALLOWANCE,
    )


def score_submission(
    code: str,
    claimed_sharpe: Any,
    load: Loader,
    wf: WalkForward = WalkForward(),
    limit: TimeLimit = TimeLimit(),
    user: str | None = None,
    check_config: checks.CheckConfig = checks.CheckConfig(),
) -> Score:
    """Scores a submission and runs every check.

    Args:
        code: Alpha code.
        claimed_sharpe: The net Sharpe the agent expects on unseen data.
        load: Picklable callable returning the data.
        wf: Walk-forward settings.
        limit: Time limit of each pipeline run; each check adds runs. All
            runs get the limit of the full one, the largest.
        user: Run the alpha as this dedicated user; see `Sandbox`.
        check_config: Settings of the lookahead checks.

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
            (_Pipeline(code, user, limit, STARTUP_ALLOWANCE), load, wf, check_config),
            STARTUP_ALLOWANCE,
        )
    except AlphaError as e:
        return invalid_score(str(e))
    return Score(
        performance,
        claimed - performance.sharpe,
        int(used_leak),
        int(lookahead),
        0,
        shifted,
    )


def run_isolated(target: Callable[..., Any], args: tuple, timeout: float) -> Any:
    """Runs a function in a fresh process and returns its result.

    The process is killed if it runs past its deadline: `timeout` seconds
    after the start, or what the target last set. `target` is called as
    `target(allow, *args)`; `allow(seconds)` moves the deadline to that many
    seconds from now, for a target that only learns how long it needs once
    it has loaded its data. `target` must be a module-level function so that
    it can be pickled.

    Args:
        target: Function to call.
        args: Positional arguments after `allow`, all picklable.
        timeout: Wall-clock limit in seconds until the target first calls
            `allow`.

    Returns:
        Whatever `target` returns.

    Raises:
        AlphaTimeout: If the process runs past its deadline.
        AlphaError: If `target` raises AlphaError, or the process dies.
        SandboxError: If `target` raises anything else: a problem with the
            scoring itself, not with the alpha.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, target, args), daemon=True)
    proc.start()
    child.close()
    started = time.monotonic()
    deadline = started + timeout
    try:
        while True:
            if not parent.poll(max(deadline - time.monotonic(), 0.0)):
                raise AlphaTimeout(f"ran past {deadline - started:.0f} s")
            kind, payload = parent.recv()
            if kind != _ALLOW:
                break
            deadline = time.monotonic() + payload
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
    def allow(seconds: float) -> None:
        conn.send((_ALLOW, seconds))

    try:
        result = (_OK, target(allow, *args))
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


@dataclasses.dataclass(frozen=True)
class _Pipeline:
    """What a worker needs to build its sandbox once the data is loaded."""

    code: str
    user: str | None
    limit: TimeLimit
    # Deadline slack on top of the runs, for checks and metrics.
    slack: float

    def sandbox(
        self, allow: Allow, data: EvalData, wf: WalkForward, runs: int
    ) -> Sandbox:
        """A sandbox whose limit fits a full run on `data`; sets the deadline
        for `runs` such runs."""
        seconds = self.limit.seconds(walkforward.run_rows(data.panel, wf, data.start))
        allow(runs * seconds + self.slack)
        return Sandbox(self.code, seconds, self.user)


def _backtest_worker(
    allow: Allow, pipeline: _Pipeline, load: Loader, wf: WalkForward
) -> metrics.Performance:
    data = load()
    sandbox = pipeline.sandbox(allow, data, wf, runs=1)
    run = walkforward.run(sandbox, data.panel, wf, data.start)
    return _performance(data, run)


def _score_worker(
    allow: Allow,
    pipeline: _Pipeline,
    load: Loader,
    wf: WalkForward,
    config: checks.CheckConfig,
) -> tuple[metrics.Performance, bool, bool, dict[int, float]]:
    data = load()
    # Pipeline runs: base, column dependence, one per truncation cut.
    sandbox = pipeline.sandbox(allow, data, wf, runs=2 + config.n_cuts)
    panel, start = data.panel, data.start
    seed = data.check_seed if data.check_seed is not None else secrets.randbits(64)
    base = walkforward.run(sandbox, panel, wf, start)
    used_leak = checks.column_dependence(
        sandbox, panel, list(data.suspects), wf, start, base, seed=seed, config=config
    )
    lookahead = checks.truncation(
        sandbox, panel, wf, start, base, seed=seed, config=config
    )
    return (
        _performance(data, base),
        used_leak,
        lookahead,
        checks.shifted_ics(panel, base),
    )
