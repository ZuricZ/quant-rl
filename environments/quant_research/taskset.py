"""The quant research taskset: one simulated dataset per task, scored on
periods the agent never sees.

The agent gets the visible periods through the research toolset, which runs
in the rollout's container. After the episode, the reward runs
`quantenv.grade` in that same runtime; it rebuilds the dataset with
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
from collections.abc import Iterable, Iterator
from typing import Any

import verifiers.v1 as vf
from pydantic import Field, SecretStr

from quant_research.servers.research import (
    ResearchConfig,
    ResearchState,
    ResearchToolset,
)
from quantenv import baselines, checks, reward, scoring
from quantenv.grading import check_sandbox_user, grade, scoring_timeout
from quantenv.sources import synthetic
from quantenv.taskset import (
    ALPHA_SECTION,
    DATA_SECTION,
    EVAL_TASKS,
    IMAGE,
    PORTFOLIO,
    SECRET_VAR,
    ResearchData,
    SplitConfig,
    describe_panel,
    prompt_choices,
    prompt_values,
)

MAX_TURNS = 30
_SCORE = "score"

PROMPT = (
    "{opening}\n\n"
    + DATA_SECTION
    + "{request}"
    + ALPHA_SECTION
    + "A backtest run hands over {backtest_rows:.2f}M rows, so it may take "
    "{backtest_limit:.1f} seconds.\n\n"
    "The score is the annualised net Sharpe ratio of a portfolio traded on "
    "your predictions. "
    + PORTFOLIO
    + "\n\n"
    + "{example}"
    + """## Tools

- `describe()`: column names, timestamp range and number of assets.
- `backtest(code)`: the alpha walk-forward on the visible periods: net \
Sharpe with its standard error, gross Sharpe, turnover, mean rank IC with \
its standard error, and the range of net Sharpe across calendar years. You \
have {budget} backtests; further calls are refused.
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
)


def prompt(
    description: str,
    budget: int,
    max_turns: int,
    check_config: checks.CheckConfig = checks.CheckConfig(),
    config: synthetic.Config = synthetic.DEFAULT_CONFIG,
    variation: int | None = None,
) -> str:
    """The task prompt.

    Args:
        description: Output of `describe_panel` on the visible data.
        budget: Backtest calls allowed.
        max_turns: Model turns allowed.
        check_config: Settings of the checks run when grading; the prompt
            states the IC ceiling.
        config: The task's generator parameters.
        variation: The task's variation seed, or None for a task without
            variation; see `prompt_choices`.

    Returns:
        The prompt text.
    """
    return PROMPT.format(
        description=description,
        budget=budget,
        max_turns=max_turns,
        **prompt_values(check_config, config),
        **prompt_choices(variation, config.style),
    )


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
        vary: Vary the market and the prompt per task (spec, Decision 26).
            False gives every task the default market and the fixed prompt,
            as in the runs of 17 to 19 November 2026.
    """

    tools: ResearchConfig = ResearchConfig()
    max_turns: int = Field(MAX_TURNS, gt=0)
    secret: SecretStr | None = None
    image: str = IMAGE
    check_config: checks.CheckConfig = checks.CheckConfig()
    vary: bool = True


class QuantResearchConfig(SplitConfig):
    """Taskset settings.

    Attributes:
        task: Settings passed to every task.
    """

    task: QuantResearchTaskConfig = QuantResearchTaskConfig()


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
        """Checks the sandbox user, then hands the toolset the seed and the
        variation seed through the episode's shared state.

        Raises:
            RuntimeError: If the runtime cannot run code as the sandbox user.
        """
        await check_sandbox_user(runtime, self.config.tools.sandbox_user)
        trace.state.seed = f"{self.seed:032x}"
        variation = self.variation
        trace.state.variation = "" if variation is None else f"{variation:032x}"

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
                    self.config.tools.sandbox_user,
                    runtime,
                    self.seed,
                    state.code,
                    state.claimed_sharpe,
                    self.config.check_config,
                    self.variation,
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
        market = self.market
        truth = synthetic.ground_truth(self.seed, market)
        score = await grade(
            self.config.tools.sandbox_user,
            runtime,
            self.seed,
            baselines.oracle_code(truth, market.schema),
            0.0,
            self.config.check_config,
            self.variation,
        )
        return reward.reward(score) > 0


class QuantResearchTaskset(vf.Taskset[QuantResearchTask, QuantResearchConfig]):
    """Task i of a split uses `synthetic.dataset_seed(i, secret, split)`. The
    eval split is tasks 0 to `EVAL_TASKS` - 1; the train split is infinite."""

    def __init__(self, config: QuantResearchConfig) -> None:
        if config.secret is None:
            raise ValueError(
                "the taskset needs a secret: set --env.taskset.secret "
                f"(--taskset.secret for vf-validate) or {SECRET_VAR}"
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
            market = task.market
            visible = synthetic.visible_data(task.seed, market)
            data = task.data.model_copy(
                update=dict(
                    name=f"{self.config.split}-{index}",
                    prompt=prompt(
                        describe_panel(visible.panel),
                        task_config.tools.budget,
                        task_config.max_turns,
                        task_config.check_config,
                        market,
                        task.variation,
                    ),
                    image=task_config.image,
                    # Grade in a fresh container of the image, which agent
                    # code never touched; the inputs come from the trace.
                    scoring=vf.ScoringConfig(mode="isolated"),
                    timeout=vf.TaskTimeout(
                        scoring=scoring_timeout(task_config.check_config, market)
                    ),
                )
            )
            yield QuantResearchTask(data, task_config)
