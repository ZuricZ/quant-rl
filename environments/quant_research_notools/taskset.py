"""The no-tools variant of the quant research taskset, for the prime-rl
smoke run.

One reply per episode. The prompt describes the visible periods and lists
each feature's score on them, since the model cannot run code; the model
answers with alpha code in a Python block. The reward is the alpha's rank IC
on the hidden periods, scaled by the IC ceiling, with the main taskset's
penalties for invalid and flagged submissions. Datasets, seeds, the sandbox
and the grader are the main taskset's (`quant_research.taskset`).
"""

import dataclasses
import itertools
import re
from collections.abc import Iterable, Iterator
from typing import Any

import polars as pl
import verifiers.v1 as vf

from quant_research.servers.research import SANDBOX_USER, ResearchData, describe_panel
from quant_research.taskset import (
    ALPHA_SECTION,
    DATA_SECTION,
    EVAL_TASKS,
    IMAGE,
    PORTFOLIO,
    RESOURCES,
    Secret,
    SplitConfig,
    check_sandbox_user,
    check_secret,
    grade,
    prompt_choices,
    prompt_values,
    scoring_timeout,
)
from quantenv import baselines, checks, metrics, reward, scoring
from quantenv.panel import EvalData
from quantenv.sources import synthetic

_SCORE = "score"
# The last fenced Python block of a reply.
_CODE_BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
# What `grade` is told the agent claimed; this variant asks for no claim.
_NO_CLAIM = 0.0

# Polars 1 APIs that polars 2 removed and that base models still write (19 of
# 60 Qwen3.5-9B replies failed on them). Only this taskset lists them: the
# tools taskset can read the docs instead. Part of a format string: no braces.
POLARS_2_NOTES = """\
Polars 2 removed some polars 1 APIs:

- `rolling_*` and `ewm_*` take `min_samples`, not `min_periods`.
- There is no `Expr.corr` or `pl.spearman_corr`: use \
`pl.corr(a, b, method="spearman")`, with `.over("timestamp")` for one value \
per period.

"""

PROMPT = (
    "Write a trading alpha for a simulated panel of asset returns. It is "
    "scored on the periods that directly follow the ones described here. "
    "You cannot run code: answer in one reply.\n\n"
    + DATA_SECTION
    + "{request}"
    + """\
## Features

Each feature, and `ret`, used alone as the prediction on the visible \
periods: its mean rank IC (per period, the Spearman correlation across assets with \
`fwd_return`, averaged over periods) and the annualised net Sharpe ratio of \
the portfolio described below.

{feature_table}

"""
    + ALPHA_SECTION
    + "Vectorised polars expressions finish well inside this; Python loops "
    "over rows do not.\n\n"
    + POLARS_2_NOTES
    + "## Portfolio\n\n"
    + PORTFOLIO
    + "\n\n"
    + "{example}"
    + """## Answer

Reply with your alpha in one ```python block; if there are several, the last \
one is used. Define your own class that subclasses `Alpha`, as the example \
does, rather than redefining `Alpha`. The alpha is run on the {n_hidden:,} \
unseen periods that directly follow the visible ones, walk-forward with the \
same {window}-timestamp training window, and judged by its mean rank IC \
against `fwd_return` there.
"""
)


def feature_table(data: EvalData) -> str:
    """Each feature's score on its own, and that of `ret`, as a Markdown table.

    Args:
        data: The visible data.

    Returns:
        One row per column, features first in column order: mean rank IC
        against the target and the net Sharpe of trading the column as the
        prediction.
    """
    s = data.panel.schema
    frame = data.panel.frame
    columns = [*s.features, s.ret]
    ics = metrics.mean_rank_ics(frame, {c: (c, s.target) for c in columns}, s.timestamp)
    lines = ["| Column | Rank IC | Net Sharpe |", "| --- | --- | --- |"]
    for f in columns:
        sharpe = metrics.performance(
            frame, f, s.target, s.timestamp, s.asset, data.cost, data.periods_per_year
        ).sharpe
        lines.append(f"| {f} | {ics[f]:.4f} | {sharpe:.2f} |")
    return "\n".join(lines)


def prompt(
    description: str,
    table: str,
    config: synthetic.Config = synthetic.DEFAULT_CONFIG,
    variation: int | None = None,
) -> str:
    """The task prompt.

    Args:
        description: Output of `describe_panel` on the visible data.
        table: Output of `feature_table` on the visible data.
        config: The task's generator parameters.
        variation: The task's variation seed, or None for a task without
            variation. Varies the style request and the example, not the
            opening, which states the one-reply rule.

    Returns:
        The prompt text.
    """
    choices = prompt_choices(variation, config.style)
    return PROMPT.format(
        description=description,
        feature_table=table,
        request=choices["request"],
        example=choices["example"],
        **prompt_values(config),
    )


def extract_code(reply: str) -> str | None:
    """The last fenced Python block of a reply's answer.

    Inline reasoning is not part of the answer: only the text after the last
    `</think>` is searched, and a reply whose `<think>` never closes has no
    answer. This matches verifiers' own scoring helpers.

    Args:
        reply: The model's reply text.

    Returns:
        The code of the block, or None if the answer has none.
    """
    if "<think>" in reply and "</think>" not in reply:
        return None
    blocks = _CODE_BLOCK.findall(reply.rsplit("</think>", 1)[-1])
    return blocks[-1] if blocks else None


class NoToolsTaskConfig(vf.TaskConfig):
    """Task settings.

    Attributes:
        secret: The taskset's secret; the taskset sets it on every task.
        image: Task image; the main taskset's, which installs this package.
        check_config: Settings of the lookahead checks run when grading.
        sandbox_user: User that runs the alpha when grading; see
            `quant_research.servers.research.ResearchConfig`.
        vary: Vary the market and the prompt per task (spec, Decision 26).
    """

    secret: Secret = None
    image: str = IMAGE
    check_config: checks.CheckConfig = checks.CheckConfig()
    sandbox_user: str | None = SANDBOX_USER
    vary: bool = True


class NoToolsConfig(SplitConfig):
    """Taskset settings: the split and secret (see `SplitConfig`), and the
    settings passed to every task."""

    task: NoToolsTaskConfig = NoToolsTaskConfig()


class NoToolsTask(vf.Task[ResearchData, vf.State, NoToolsTaskConfig]):
    """One dataset: write an alpha in one reply."""

    # The grader runs agent-written code.
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

    @property
    def variation(self) -> int | None:
        """The variation seed (`synthetic.variation_seed`), or None if the
        task does not vary."""
        if not self.config.vary:
            return None
        return synthetic.variation_seed(self.data.index, self.data.split, self.seed)

    @property
    def market(self) -> synthetic.Config:
        """The task's generator parameters."""
        return synthetic.task_config(self.variation)

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Checks the sandbox user.

        Raises:
            RuntimeError: If the runtime cannot run code as the sandbox user.
        """
        await check_sandbox_user(runtime, self.config.sandbox_user)

    @vf.stop
    def answered(self, trace: vf.Trace) -> bool:
        """Ends the episode after the model's first reply."""
        return trace.num_turns >= 1

    async def raw_score(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, Any]:
        """The reply's raw score, graded once per trace and kept on it as
        `trace.info["score"]`.

        Args:
            trace: The episode's trace.
            runtime: A runtime of the task image, for the grader.

        Returns:
            A `scoring.Score` as a dict; invalid if the reply has no Python
            block.
        """
        if _SCORE not in trace.info:
            code = extract_code(trace.last_reply)
            if code is None:
                score = dataclasses.asdict(scoring.invalid_score("no python block"))
            else:
                score = await grade(
                    self.config.sandbox_user,
                    runtime,
                    self.seed,
                    code,
                    _NO_CLAIM,
                    self.config.check_config,
                    self.variation,
                )
            trace.info[_SCORE] = score
        return trace.info[_SCORE]

    @vf.metric
    async def outcome(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Checks and performance; performance is NaN for an invalid reply."""
        score = await self.raw_score(trace, runtime)
        nan = float("nan")
        performance = score["performance"] or {}
        return {
            "no_code": float(score["error"] == "no python block"),
            "used_leak": score["used_leak"],
            "lookahead": score["lookahead"],
            "implausible_ic": score["implausible_ic"],
            "invalid": score["invalid"],
            "raw_ic": performance.get("ic", nan),
            "sharpe": performance.get("sharpe", nan),
            "turnover": performance.get("turnover", nan),
        }

    @vf.reward
    async def hidden_ic(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Hidden-period rank IC over the IC ceiling; -1.5 if a check fired,
        -1 if the reply has no code or the alpha is invalid."""
        score = await self.raw_score(trace, runtime)
        return reward.ic_reward(score, self.config.check_config.max_ic)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Grades the oracle alpha: it must score above 0 and pass the checks."""
        market = self.market
        truth = synthetic.ground_truth(self.seed, market)
        score = await grade(
            self.config.sandbox_user,
            runtime,
            self.seed,
            baselines.oracle_code(truth, market.schema),
            _NO_CLAIM,
            self.config.check_config,
            self.variation,
        )
        return reward.ic_reward(score, self.config.check_config.max_ic) > 0


class NoToolsTaskset(vf.Taskset[NoToolsTask, NoToolsConfig]):
    """Task i of a split uses `synthetic.dataset_seed(i, secret, split)`, the
    same dataset as task i of the main taskset with the same secret."""

    def __init__(self, config: NoToolsConfig) -> None:
        check_secret(config)
        super().__init__(config)

    @property
    def INFINITE(self) -> bool:  # noqa: N802 - the name verifiers reads.
        return self.config.split == "train"

    def load(self) -> Iterator[NoToolsTask]:
        task_config = self.config.task
        indices: Iterable[int] = (
            itertools.count() if self.INFINITE else range(EVAL_TASKS)
        )
        for index in indices:
            task = NoToolsTask(
                ResearchData(split=self.config.split, index=index), task_config
            )
            market = task.market
            visible = synthetic.visible_data(task.seed, market)
            data = task.data.model_copy(
                update=dict(
                    name=f"{self.config.split}-{index}",
                    prompt=prompt(
                        describe_panel(visible.panel),
                        feature_table(visible),
                        market,
                        task.variation,
                    ),
                    image=task_config.image,
                    resources=RESOURCES,
                    scoring=vf.ScoringConfig(mode="isolated"),
                    timeout=vf.TaskTimeout(
                        scoring=scoring_timeout(task_config.check_config, market)
                    ),
                )
            )
            yield NoToolsTask(data, task_config)
