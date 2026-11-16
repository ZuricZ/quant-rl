"""The research toolset: describe, backtest and submit, served over MCP.

The server runs colocated with the harness, inside the rollout's container,
and only ever builds the visible periods of the task's dataset. Each backtest
rebuilds them in a fresh scoring worker, which runs the agent's code in
`quantenv.sandbox` as the unprivileged sandbox user.

Run as `python -m quant_research.servers.research`; verifiers launches it.
"""

import asyncio
import functools
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from quantenv import scoring
from quantenv.alpha import AlphaError
from quantenv.panel import Panel
from quantenv.sources import synthetic

BUDGET = 10
SANDBOX_USER = "nobody"


class ResearchData(vf.TaskData):
    """Task data. Published with every trace, so it holds no seed.

    Attributes:
        split: "train" or "eval".
        index: Dataset number within the split.
    """

    split: Literal["train", "eval"]
    index: int = Field(ge=0)


class ResearchState(vf.State):
    """Episode state shared by the toolset and the task. Never serialised
    into traces.

    Attributes:
        seed: Dataset seed as hex, set by the task's setup before the agent
            starts. The agent must not learn it.
        backtests_used: Backtest calls that ran; refused calls do not count.
        submitted: Whether `submit` has run.
        code: The submitted alpha code.
        claimed_sharpe: The submitted claim of the net Sharpe on unseen data.
    """

    seed: str = ""
    backtests_used: int = 0
    submitted: bool = False
    code: str = ""
    claimed_sharpe: float | None = None


# What the toolset records; it only reads the seed.
_EPISODE_FIELDS = ("backtests_used", "submitted", "code", "claimed_sharpe")


class ResearchConfig(vf.ToolsetConfig):
    """Toolset settings.

    Attributes:
        colocated: Serve the tools inside the harness's runtime.
        budget: Backtest calls allowed per episode.
        sandbox_user: User that runs the agent's code, in backtests and in
            the final score. Every process of this user is killed after each
            pipeline run, so it must not be used for anything else. None runs
            the code as the server's own user, without that isolation.
    """

    colocated: bool = True
    budget: int = Field(BUDGET, ge=0)
    sandbox_user: str | None = SANDBOX_USER


def describe_panel(panel: Panel) -> str:
    """Columns the alpha sees, timestamp range and number of assets.

    Args:
        panel: The data.

    Returns:
        A short plain-text description.
    """
    stamps = panel.timestamps
    n_assets = panel.frame.get_column(panel.schema.asset).n_unique()
    return (
        f"columns: {', '.join(panel.observable.columns)}\n"
        f"timestamps: {stamps.len()}, from {stamps.first()} to {stamps.last()}\n"
        f"assets: {n_assets}"
    )


class ResearchToolset(vf.Toolset[ResearchConfig, ResearchState]):
    """Tools for one episode.

    The toolset keeps its own record of the episode and copies it into the
    shared state at the end of every call. State is pulled before a call and
    pushed after it, so a call that started from an older copy would
    otherwise overwrite what a concurrent call recorded. Backtests run one at
    a time, since each one ends by killing every process of the sandbox user.
    """

    TOOL_PREFIX = None  # Advertise the tools as describe, backtest and submit.

    def __init__(self, config: ResearchConfig) -> None:
        super().__init__(config)
        self._episode = ResearchState()
        self._backtesting = asyncio.Lock()
        self._description: str | None = None

    def _visible(self) -> functools.partial:
        """The loader of the visible periods, from the seed in the state."""
        if not self.state.seed:
            raise ValueError("the task is not set up: no dataset seed")
        return functools.partial(synthetic.visible_data, int(self.state.seed, 16))

    def _publish(self) -> None:
        for name in _EPISODE_FIELDS:
            setattr(self.state, name, getattr(self._episode, name))

    @vf.tool
    def describe(self) -> str:
        """Describes the visible data.

        Returns:
            Column names, timestamp range and number of assets.
        """
        if self._description is None:
            self._description = describe_panel(self._visible()().panel)
        return self._description

    @vf.tool
    async def backtest(self, code: str) -> str:
        """Scores alpha code walk-forward on the visible periods.

        Each call counts against the backtest budget, including calls whose
        code fails. Calls past the budget or after `submit` are refused.

        Args:
            code: Python source defining one subclass of `Alpha`.

        Returns:
            The annualised net Sharpe ratio and mean rank IC, or why the
            code failed, and the number of backtests left.

        Raises:
            ValueError: If the call is refused.
        """
        episode = self._episode
        if episode.submitted:
            raise ValueError("refused: the alpha is already submitted")
        if episode.backtests_used >= self.config.budget:
            raise ValueError(
                f"refused: all {self.config.budget} backtests are used; "
                "call submit to finish"
            )
        episode.backtests_used += 1
        left = self.config.budget - episode.backtests_used
        try:
            async with self._backtesting:
                performance = await asyncio.to_thread(
                    scoring.backtest,
                    code,
                    self._visible(),
                    user=self.config.sandbox_user,
                )
        except AlphaError as e:
            return f"error: {e}\nbacktests left: {left}"
        finally:
            self._publish()
        return (
            f"net Sharpe: {performance.sharpe:.2f}\n"
            f"rank IC: {performance.ic:.4f}\n"
            f"backtests left: {left}"
        )

    @vf.tool
    def submit(self, code: str, claimed_sharpe: float) -> str:
        """Submits the final alpha and ends the episode.

        Args:
            code: Python source defining one subclass of `Alpha`.
            claimed_sharpe: The annualised net Sharpe ratio you expect on the
                unseen periods.

        Returns:
            A confirmation.

        Raises:
            ValueError: If an alpha is already submitted.
        """
        episode = self._episode
        if episode.submitted:
            raise ValueError("refused: the alpha is already submitted")
        episode.submitted = True
        episode.code = code
        episode.claimed_sharpe = claimed_sharpe
        self._publish()
        return "submitted; the episode is over"


if __name__ == "__main__":
    ResearchToolset.run()
