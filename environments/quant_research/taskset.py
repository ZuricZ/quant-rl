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
from verifiers.v1.mcp.launch import _install_in_sandbox

from quant_research.servers.research import (
    ResearchConfig,
    ResearchData,
    ResearchState,
    ResearchToolset,
    describe_panel,
)
from quantenv import baselines, scoring
from quantenv.sources import synthetic
from quantenv.walkforward import WalkForward

MAX_TURNS = 30
EVAL_TASKS = 30
# What pydantic writes in place of a SecretStr when dumping.
_MASK = "**********"
IMAGE = "python:3.12-slim"
SCORING_TIMEOUT = 900.0
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
- `fit(train)` gets the engineered rows before the current block, plus \
`fwd_return`. Keep fitted state on `self`.
- `predict(df)` gets the engineered rows of one block and returns one score \
per row, in row order, as a `pl.Series` or a `pl.Expr` evaluated on `df`.

The environment runs the alpha walk-forward: `features` once, then for each \
block of {refit_every} timestamps a fresh instance is fit on all earlier rows \
and predicts the block. The first {min_train} timestamps are only for \
training. The score is the mean per-period rank correlation (rank IC) \
between the predictions and `fwd_return`. Each run may take at most \
{time_limit:g} seconds.

Example:

```python
class MyAlpha(Alpha):
    def predict(self, df):
        return pl.col("ret")
```

## Tools

- `describe()`: column names, timestamp range and number of assets.
- `backtest(code)`: the walk-forward rank IC of the alpha on the visible \
periods. You have {budget} backtests; further calls are refused.
- `submit(code, claimed_ic)`: submit your final alpha and the rank IC you \
expect it to reach on the unseen periods. This ends the episode.

You have at most {max_turns} turns. An episode that ends without `submit` \
scores 0.
"""


def prompt(description: str, budget: int, max_turns: int) -> str:
    """The task prompt.

    Args:
        description: Output of `describe_panel` on the visible data.
        budget: Backtest calls allowed.
        max_turns: Model turns allowed.

    Returns:
        The prompt text.
    """
    wf = WalkForward()
    return PROMPT.format(
        description=description,
        refit_every=wf.refit_every,
        min_train=wf.min_train,
        time_limit=scoring.BACKTEST_TIME_LIMIT,
        budget=budget,
        max_turns=max_turns,
    )


async def grade(
    tools: ResearchConfig,
    runtime: vf.Runtime,
    seed: int,
    code: str,
    claimed_ic: float | None,
) -> dict[str, Any]:
    """Scores a submission on the hidden periods inside `runtime`.

    The grader runs in the Python environment the toolset was installed into,
    which holds this package. Installing is a no-op once the toolset has run.

    Args:
        tools: The toolset's config, to locate that environment.
        runtime: The rollout's runtime.
        seed: Dataset seed.
        code: Alpha code.
        claimed_ic: The claimed IC on unseen data.

    Returns:
        The fields of `quantenv.scoring.Score`.

    Raises:
        RuntimeError: If the grader itself fails. Bad alpha code does not
            raise; it scores as invalid.
    """
    python = await _install_in_sandbox(ResearchToolset(tools), runtime)
    stem = f"/tmp/vf-grade-{uuid.uuid4().hex}"
    request = {
        "seed": seed,
        "code": code,
        "claimed_ic": claimed_ic,
        "user": tools.sandbox_user,
    }
    await runtime.write(f"{stem}.json", json.dumps(request).encode())
    result = await runtime.run(
        [python, "-m", "quant_research.grade", f"{stem}.json", f"{stem}.out.json"], {}
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
    """

    tools: ResearchConfig = ResearchConfig()
    max_turns: int = Field(MAX_TURNS, gt=0)
    secret: SecretStr | None = None


class QuantResearchConfig(vf.TasksetConfig):
    """Taskset settings.

    Attributes:
        task: Settings passed to every task.
        split: "eval" is a fixed set of `EVAL_TASKS` tasks; "train" is
            infinite. Their datasets never coincide.
        secret: Required. Dataset seeds are drawn from it, see
            `synthetic.dataset_seed`. Runs that must share datasets, such as
            several models and `scripts/run_baselines.py --secret`, set the
            same value. Dumps show it masked; never show it to the agent.
    """

    task: QuantResearchTaskConfig = QuantResearchTaskConfig()
    split: Literal["train", "eval"] = "eval"
    secret: SecretStr

    @field_validator("secret")
    @classmethod
    def _not_masked(cls, secret: SecretStr) -> SecretStr:
        # A dumped config read back holds the mask, which would silently
        # become everyone's secret.
        if not secret.get_secret_value() or secret.get_secret_value() == _MASK:
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
        """Hands the toolset the seed through the episode's shared state."""
        trace.state.seed = f"{self.seed:032x}"

    @classmethod
    def toolsets(cls, config: QuantResearchTaskConfig) -> list[vf.Toolset]:
        return [ResearchToolset(config.tools)]

    @vf.stop
    def submitted(self, trace: vf.Trace) -> bool:
        """Ends the episode once `submit` has run."""
        return trace.state.submitted

    @vf.stop
    def turn_limit(self, trace: vf.Trace) -> bool:
        """Ends the episode after `max_turns` model turns."""
        return trace.num_turns >= self.config.max_turns

    async def _score(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, Any]:
        """The submission's score, graded once per trace and kept on it."""
        if _SCORE not in trace.info:
            state = trace.state
            if state.submitted:
                score = await grade(
                    self.config.tools,
                    runtime,
                    self.seed,
                    state.code,
                    state.claimed_ic,
                )
            else:
                score = dataclasses.asdict(scoring.invalid_score("no submission"))
            trace.info[_SCORE] = score
        return trace.info[_SCORE]

    @vf.metric
    async def outcome(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Checks, claim gap, budget use and shifted ICs of the submission.

        Shifted ICs are NaN for an invalid submission, as is the claim gap.
        """
        score = await self._score(trace, runtime)
        shifted = score["shifted_ics"]
        nan = float("nan")
        return {
            "used_leak": score["used_leak"],
            "lookahead": score["lookahead"],
            "invalid": score["invalid"],
            "raw_ic": score["true_ic"],
            "claim_gap": score["claim_gap"],
            "backtests_used": trace.state.backtests_used,
            "shifted_ic_m1": shifted.get("-1", nan),
            "shifted_ic_p1": shifted.get("1", nan),
            "shifted_ic_p2": shifted.get("2", nan),
        }

    @vf.reward
    async def true_ic(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Hidden-period rank IC; 0 if a check fired or the submission is invalid."""
        return (await self._score(trace, runtime))["reward"]

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Grades the oracle alpha: it must score above 0 and pass the checks."""
        weights = synthetic.ground_truth(self.seed).weights
        score = await grade(
            self.config.tools,
            runtime,
            self.seed,
            baselines.weighted_code(weights),
            0.0,
        )
        return score["reward"] > 0 and not score["used_leak"] and not score["lookahead"]


class QuantResearchTaskset(vf.Taskset[QuantResearchTask, QuantResearchConfig]):
    """Task i of a split uses `synthetic.dataset_seed(i, secret, split)`. The
    eval split is tasks 0 to `EVAL_TASKS` - 1; the train split is infinite."""

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
                        description, task_config.tools.budget, task_config.max_turns
                    ),
                    image=IMAGE,
                    timeout=vf.TaskTimeout(scoring=SCORING_TIMEOUT),
                )
            )
            yield QuantResearchTask(data, task_config)
