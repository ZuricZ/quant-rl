"""The research toolset: describe, backtest and submit, served over MCP.

The server runs colocated with the harness, inside the rollout's container,
and only ever builds the visible periods of the task's dataset. Each backtest
rebuilds them in a fresh scoring worker, which also runs the agent's code.

Run as `python -m quant_research.servers.research`; verifiers launches it.
"""

import asyncio
import functools

import verifiers.v1 as vf
from pydantic import Field

from quantenv import scoring
from quantenv.alpha import AlphaError
from quantenv.panel import Panel
from quantenv.sources import synthetic

BUDGET = 10


class ResearchData(vf.TaskData):
    """Task data shared by the task and its toolset.

    Attributes:
        seed: Dataset seed.
    """

    seed: int = Field(ge=0)


class ResearchState(vf.State):
    """Episode state shared by the toolset and the task.

    Attributes:
        backtests_used: Backtest calls that ran; refused calls do not count.
        submitted: Whether `submit` has run.
        code: The submitted alpha code.
        claimed_ic: The submitted claim of the IC on unseen data.
    """

    backtests_used: int = 0
    submitted: bool = False
    code: str = ""
    claimed_ic: float | None = None


class ResearchConfig(vf.ToolsetConfig):
    """Toolset settings.

    Attributes:
        colocated: Serve the tools inside the harness's runtime.
        budget: Backtest calls allowed per episode.
    """

    colocated: bool = True
    budget: int = Field(BUDGET, ge=0)


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
    otherwise overwrite what a concurrent call recorded.
    """

    TOOL_PREFIX = None  # Advertise the tools as describe, backtest and submit.

    async def setup_task(self, task: ResearchData) -> None:
        """Builds the visible periods once, for `describe`.

        Args:
            task: The rollout's task data.
        """
        self._load = functools.partial(synthetic.visible_data, task.seed)
        self._description = describe_panel(self._load().panel)
        self._episode = ResearchState()

    def _publish(self) -> None:
        for name in ResearchState.model_fields.keys() - vf.State.model_fields.keys():
            setattr(self.state, name, getattr(self._episode, name))

    @vf.tool
    def describe(self) -> str:
        """Describes the visible data.

        Returns:
            Column names, timestamp range and number of assets.
        """
        return self._description

    @vf.tool
    async def backtest(self, code: str) -> str:
        """Scores alpha code walk-forward on the visible periods.

        Each call counts against the backtest budget, including calls whose
        code fails. Calls past the budget or after `submit` are refused.

        Args:
            code: Python source defining one subclass of `Alpha`.

        Returns:
            The mean per-period rank IC, or why the code failed, and the
            number of backtests left.

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
            ic = await asyncio.to_thread(scoring.backtest, code, self._load)
        except AlphaError as e:
            return f"error: {e}\nbacktests left: {left}"
        finally:
            self._publish()
        return f"rank IC: {ic:.4f}\nbacktests left: {left}"

    @vf.tool
    def submit(self, code: str, claimed_ic: float) -> str:
        """Submits the final alpha and ends the episode.

        Args:
            code: Python source defining one subclass of `Alpha`.
            claimed_ic: The mean rank IC you expect on the unseen periods.

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
        episode.claimed_ic = claimed_ic
        self._publish()
        return "submitted; the episode is over"


if __name__ == "__main__":
    ResearchToolset.run()
