"""The quant research taskset: one simulated dataset per task, scored on
periods the agent never sees.

The agent gets the visible periods through the research toolset, which runs
in the rollout's container. After the episode, the reward runs
`quant_research.grade` in that same runtime; it rebuilds the dataset with
its hidden periods inside a scoring worker, so hidden data never exists
while the agent runs. All agent code runs in `quantenv.sandbox`.

Dataset seeds are drawn from the taskset's `secret` and split. The
generator is installed in the container, so code that knew a seed could
rebuild the hidden periods; it cannot guess one drawn from a secret it never
sees. Seeds never enter task data, which every trace publishes: the task's
setup puts the seed in the episode's shared state, read by the toolset.
"""

import dataclasses
import itertools
import json
import uuid
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field, SecretStr, field_validator

from quant_research.servers.research import (
    ResearchConfig,
    ResearchData,
    ResearchState,
    ResearchToolset,
    describe_panel,
)
from quantenv import baselines, checks, reward, scoring, walkforward
from quantenv.sources import synthetic
from quantenv.walkforward import WalkForward

MAX_TURNS = 30
EVAL_TASKS = 30
# What pydantic writes in place of a SecretStr when dumping.
_MASK = "**********"
# Built from the repository's Dockerfile; holds this package.
IMAGE = "quant-research:0.1.0"
_SCORE = "score"

PROMPT = """\
Research a trading alpha on a simulated panel of asset returns. You are \
scored on the periods that directly follow the ones you can see.

## Data

{description}

One row per (timestamp, asset), sorted by timestamp then asset. `ret` is the \
return realised at that timestamp, from the previous period to this one. The \
features are anonymous. The target, `fwd_return` (the return from this \
period to the next), is shown only to `fit`, and only for rows whose target \
was already realised.

## Alpha

Write Python that defines exactly one subclass of the predefined class \
`Alpha`. `pl` (polars) and `Alpha` are available without imports.

```python
class Alpha:
    def features(self, df: pl.DataFrame) -> pl.DataFrame:
        return df                    # default: no new features
    def fit(self, train: pl.DataFrame) -> None:
        pass                         # default: nothing to fit
    def predict(self, df: pl.DataFrame) -> pl.Series | pl.Expr:
        raise NotImplementedError    # must be overridden
```

- `features(df)` gets every row and returns the same rows in the same order \
with any added columns. A value at timestamp t may only use rows at or \
before t.
- `fit(train)` gets the engineered rows of the {window} timestamps just \
before the current block, plus `fwd_return`. Keep fitted state on `self`.
- `predict(df)` gets the engineered rows of one block and returns one score \
per row, in row order, as a `pl.Series` or a `pl.Expr` evaluated on `df`.
- For randomness, use `self.rng`, a `numpy.random.Generator` that the \
environment seeds before each call, so reruns draw the same numbers.

The environment runs the alpha walk-forward: `features` once, then for each \
block of {refit_every} timestamps a fresh instance is fit on the {window} \
timestamps before it and predicts the block. The window slides; it never \
grows. The first {window} timestamps are only for training.

Time limits scale with the rows a run hands to your code: `features` gets \
every row, each block's `fit` the rows of its window, and its `predict` the \
block's rows. A run may take {startup:g} seconds plus {per_million_rows:g} \
seconds per million such rows. A backtest run hands over \
{backtest_rows:.2f}M rows, so it may take {backtest_limit:.1f} seconds.

The score is the annualised net Sharpe ratio of a portfolio traded on your \
predictions. Each period the assets are ranked by prediction and held with \
dollar-neutral weights proportional to the centred rank, scaled so the \
absolute weights sum to 1. Each period's P&L is the weights times \
`fwd_return`, minus {cost_bp:g} bp per unit of turnover (the sum of absolute \
weight changes). At most {max_null:.0%} of the predictions may be null.

Example:

```python
class MyAlpha(Alpha):
    def predict(self, df):
        return pl.col("ret")
```

## Tools

- `describe()`: column names, timestamp range and number of assets.
- `backtest(code)`: the net Sharpe and the mean rank IC of the alpha, \
walk-forward on the visible periods. You have {budget} backtests; further \
calls are refused.
- `submit(code, claimed_sharpe)`: submit your final alpha and the net Sharpe \
you expect it to reach on the unseen periods. This ends the episode.

Your submission is scored on the {n_hidden:,} unseen periods that directly \
follow the visible ones, walk-forward with the same {window}-timestamp \
training window. A scored run hands over {score_rows:.1f}M rows, so it may \
take {score_limit:.1f} seconds: an alpha that fits its backtest limit fits \
this one too, if its cost grows with the rows it gets.

You have at most {max_turns} turns. An episode that ends without `submit`, \
or whose alpha fails, scores {invalid_reward:g}; one caught reading the future \
scores {flagged_reward:g}, and so does a rank IC against the next return of \
{max_ic:g} or more, which honest alphas do not reach. Any honest alpha scores higher \
than both.
"""


def run_rows(hidden: bool, wf: WalkForward = WalkForward()) -> int:
    """Rows a run on the default synthetic data hands to agent code.

    From the data's shape (`walkforward.pipeline_rows`), so the prompt can
    state the limits without building the hidden periods.

    Args:
        hidden: True for a scored run on the hidden periods, False for a
            backtest on the visible ones.
        wf: Walk-forward settings.

    Returns:
        The number of rows.
    """
    config = synthetic.DEFAULT_CONFIG
    if hidden:
        n_timestamps, n_warmup = config.n_visible + config.n_hidden, config.n_visible
    else:
        n_timestamps, n_warmup = config.n_visible, wf.window
    return walkforward.pipeline_rows(n_timestamps, config.n_assets, n_warmup, wf)


def scoring_timeout(check_config: checks.CheckConfig) -> float:
    """Seconds verifiers allows the grader.

    Every pipeline run at the scored run's limit, plus start-up for the
    container and the scoring worker. Counts the column dependence run even
    when the data has no suspects, so it bounds both cases.

    Args:
        check_config: Settings of the checks; uses `n_cuts`.

    Returns:
        The timeout in seconds.
    """
    runs = 2 + check_config.n_cuts
    seconds = scoring.TimeLimit().seconds(run_rows(hidden=True))
    return runs * seconds + 3 * scoring.STARTUP_ALLOWANCE


def prompt(
    description: str,
    budget: int,
    max_turns: int,
    check_config: checks.CheckConfig = checks.CheckConfig(),
) -> str:
    """The task prompt.

    Args:
        description: Output of `describe_panel` on the visible data.
        budget: Backtest calls allowed.
        max_turns: Model turns allowed.
        check_config: Settings of the checks run when grading; the prompt
            states the IC ceiling.

    Returns:
        The prompt text.
    """
    wf = WalkForward()
    limit = scoring.TimeLimit()
    backtest_rows, score_rows = run_rows(hidden=False), run_rows(hidden=True)
    return PROMPT.format(
        description=description,
        refit_every=wf.refit_every,
        window=wf.window,
        max_ic=check_config.max_ic,
        startup=limit.startup,
        per_million_rows=limit.per_million_rows,
        backtest_rows=backtest_rows / 1e6,
        backtest_limit=limit.seconds(backtest_rows),
        score_rows=score_rows / 1e6,
        n_hidden=synthetic.DEFAULT_CONFIG.n_hidden,
        invalid_reward=reward.INVALID_REWARD,
        flagged_reward=reward.FLAGGED_REWARD,
        score_limit=limit.seconds(score_rows),
        cost_bp=synthetic.DEFAULT_CONFIG.cost * 1e4,
        max_null=walkforward.MAX_NULL_SHARE,
        budget=budget,
        max_turns=max_turns,
    )


async def grade(
    tools: ResearchConfig,
    runtime: vf.Runtime,
    seed: int,
    code: str,
    claimed_sharpe: float | None,
    check_config: checks.CheckConfig = checks.CheckConfig(),
) -> dict[str, Any]:
    """Scores a submission on the hidden periods inside `runtime`.

    The grader runs with the task image's own `python`, which has this
    package installed (see the repository's Dockerfile).

    Args:
        tools: The toolset's config, for the sandbox user.
        runtime: A runtime of the task image.
        seed: Dataset seed.
        code: Alpha code.
        claimed_sharpe: The claimed net Sharpe on unseen data.
        check_config: Settings of the lookahead checks.

    Returns:
        The fields of `quantenv.scoring.Score`.

    Raises:
        RuntimeError: If the grader itself fails. Bad alpha code does not
            raise; it scores as invalid.
    """
    stem = f"/tmp/vf-grade-{uuid.uuid4().hex}"
    request = {
        "seed": seed,
        "code": code,
        "claimed_sharpe": claimed_sharpe,
        "checks": dataclasses.asdict(check_config),
        "user": tools.sandbox_user,
    }
    await runtime.write(f"{stem}.json", json.dumps(request).encode())
    result = await runtime.run(
        ["python", "-m", "quant_research.grade", f"{stem}.json", f"{stem}.out.json"],
        {},
    )
    if result.exit_code != 0:
        raise RuntimeError(f"grader failed: {(result.stderr or result.stdout)[-2000:]}")
    return json.loads(await runtime.read(f"{stem}.out.json"))


class QuantResearchTaskConfig(vf.TaskConfig):
    """Task settings.

    Attributes:
        tools: The research toolset's settings, including the backtest budget.
        max_turns: Model turns per episode.
        secret: The taskset's secret; the taskset sets it on every task.
        image: Task image, built from the repository's Dockerfile, which
            installs this package; for a remote runtime, a registry
            reference to it.
        check_config: Settings of the lookahead checks run when grading.
    """

    tools: ResearchConfig = ResearchConfig()
    max_turns: int = Field(MAX_TURNS, gt=0)
    secret: SecretStr | None = None
    image: str = IMAGE
    check_config: checks.CheckConfig = checks.CheckConfig()


class QuantResearchConfig(vf.TasksetConfig):
    """Taskset settings.

    Attributes:
        task: Settings passed to every task.
        split: "eval" is a fixed set of `EVAL_TASKS` tasks; "train" is
            infinite. Their datasets never coincide.
        secret: Required; the taskset will not build without it. Dataset
            seeds are drawn from it, see
            `synthetic.dataset_seed`. Runs that must share datasets, such as
            several models and `scripts/run_baselines.py --secret`, set the
            same value. Dumps show it masked; never show it to the agent.
    """

    task: QuantResearchTaskConfig = QuantResearchTaskConfig()
    split: Literal["train", "eval"] = "eval"
    # Optional only because verifiers' CLI builds this config with defaults
    # before it reads the flags; the taskset refuses to build without it.
    secret: SecretStr | None = None

    @field_validator("secret")
    @classmethod
    def _not_masked(cls, secret: SecretStr | None) -> SecretStr | None:
        # A dumped config read back holds the mask, which would silently
        # become everyone's secret.
        if secret is not None and secret.get_secret_value() in ("", _MASK):
            raise ValueError("secret is empty or masked; pass the real secret")
        return secret


class QuantResearchTask(vf.Task[ResearchData, ResearchState, QuantResearchTaskConfig]):
    """One dataset: research with the tools, submit once."""

    # The tools and the grader run agent-written code.
    NEEDS_CONTAINER = True

    @property
    def seed(self) -> int:
        """The dataset seed, from the secret, split and index.

        Raises:
            ValueError: If the task's config has no secret.
        """
        if self.config.secret is None:
            raise ValueError("task config has no secret; build tasks via the taskset")
        secret = self.config.secret.get_secret_value()
        return synthetic.dataset_seed(self.data.index, secret, self.data.split)

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Checks the sandbox user, then hands the toolset the seed through the
        episode's shared state.

        Raises:
            RuntimeError: If the runtime cannot run code as the sandbox user.
        """
        user = self.config.tools.sandbox_user
        if user is not None:
            check = await runtime.run(
                ["python", "-m", "quant_research.preflight", user], {}
            )
            if check.exit_code != 0:
                raise RuntimeError(
                    f"sandbox user {user!r} cannot be used: "
                    f"{(check.stderr or check.stdout).strip()[-500:]}"
                )
        trace.state.seed = f"{self.seed:032x}"

    @classmethod
    def toolsets(cls, config: QuantResearchTaskConfig) -> list[vf.Toolset]:
        return [ResearchToolset(config.tools)]

    @vf.stop
    def submitted(self, trace: vf.Trace) -> bool:
        """Ends the episode once `submit` has run."""
        return trace.state.submitted

    @vf.stop
    def max_turns(self, trace: vf.Trace) -> bool:
        """Ends the episode after `max_turns` model turns. The name makes
        verifiers count it as a truncation, like its own turn limit."""
        return trace.num_turns >= self.config.max_turns

    async def raw_score(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, Any]:
        """The submission's raw score, graded once per trace and kept on it.

        A `scoring.Score` as a dict: performance, flags, claim gap, error;
        no reward. Also stored as `trace.info["score"]`. A reward plugged in
        place of `hidden_sharpe` (`--env.taskset.task.rewards.hidden_sharpe.fn
        module:function`) gets it with `await task.raw_score(trace, runtime)`.

        Args:
            trace: The episode's trace.
            runtime: A runtime of the task image, for the grader.

        Returns:
            The raw score.
        """
        if _SCORE not in trace.info:
            state = trace.state
            if state.submitted:
                score = await grade(
                    self.config.tools,
                    runtime,
                    self.seed,
                    state.code,
                    state.claimed_sharpe,
                    self.config.check_config,
                )
            else:
                score = dataclasses.asdict(scoring.invalid_score("no submission"))
            trace.info[_SCORE] = score
        return trace.info[_SCORE]

    @vf.metric
    async def outcome(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Checks, performance, claim gap, budget use and shifted ICs.

        Performance, shifted ICs and the claim gap are NaN for an invalid
        submission. `raw_ic` is the IC whether or not a check fired.
        """
        score = await self.raw_score(trace, runtime)
        shifted = score["shifted_ics"]
        nan = float("nan")
        performance = score["performance"] or {}
        return {
            "used_leak": score["used_leak"],
            "lookahead": score["lookahead"],
            "implausible_ic": score["implausible_ic"],
            "invalid": score["invalid"],
            "raw_ic": performance.get("ic", nan),
            "sharpe": performance.get("sharpe", nan),
            "robust_sharpe": performance.get("robust_sharpe", nan),
            "gross_sharpe": performance.get("gross_sharpe", nan),
            "turnover": performance.get("turnover", nan),
            "year_sharpe_min": performance.get("year_sharpe_min", nan),
            "year_sharpe_median": performance.get("year_sharpe_median", nan),
            "year_sharpe_max": performance.get("year_sharpe_max", nan),
            "max_drawdown": performance.get("max_drawdown", nan),
            "claim_gap": score["claim_gap"],
            "backtests_used": trace.state.backtests_used,
            "shifted_ic_m1": shifted.get("-1", nan),
            "shifted_ic_p1": shifted.get("1", nan),
            "shifted_ic_p2": shifted.get("2", nan),
        }

    @vf.reward
    async def hidden_sharpe(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """tanh(S / 3) of the robust net Sharpe S on the hidden periods; -1.5
        if a check fired, -1 if the submission is missing or invalid (see
        `quantenv.reward`)."""
        return reward.reward(await self.raw_score(trace, runtime))

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Grades the oracle alpha: it must score above 0 and pass the checks."""
        weights = synthetic.ground_truth(self.seed).weights
        score = await grade(
            self.config.tools,
            runtime,
            self.seed,
            baselines.weighted_code(weights),
            0.0,
            self.config.check_config,
        )
        return reward.reward(score) > 0


class QuantResearchTaskset(vf.Taskset[QuantResearchTask, QuantResearchConfig]):
    """Task i of a split uses `synthetic.dataset_seed(i, secret, split)`. The
    eval split is tasks 0 to `EVAL_TASKS` - 1; the train split is infinite."""

    def __init__(self, config: QuantResearchConfig) -> None:
        if config.secret is None:
            raise ValueError(
                "the taskset needs a secret: set --env.taskset.secret "
                "(--taskset.secret for vf-validate)"
            )
        super().__init__(config)

    @property
    def INFINITE(self) -> bool:  # noqa: N802 - the name verifiers reads.
        return self.config.split == "train"

    def load(self) -> Iterator[QuantResearchTask]:
        task_config = self.config.task.model_copy(update={"secret": self.config.secret})
        indices: Iterable[int] = (
            itertools.count() if self.INFINITE else range(EVAL_TASKS)
        )
        for index in indices:
            task = QuantResearchTask(
                ResearchData(split=self.config.split, index=index), task_config
            )
            description = describe_panel(synthetic.visible_data(task.seed).panel)
            data = task.data.model_copy(
                update=dict(
                    name=f"{self.config.split}-{index}",
                    prompt=prompt(
                        description,
                        task_config.tools.budget,
                        task_config.max_turns,
                        task_config.check_config,
                    ),
                    image=task_config.image,
                    # Grade in a fresh container of the image, which agent
                    # code never touched; the inputs come from the trace.
                    scoring=vf.ScoringConfig(mode="isolated"),
                    timeout=vf.TaskTimeout(
                        scoring=scoring_timeout(task_config.check_config)
                    ),
                )
            )
            yield QuantResearchTask(data, task_config)
