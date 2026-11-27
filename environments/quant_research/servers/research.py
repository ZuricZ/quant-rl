"""The research toolset: describe, docs, check, backtest and submit, served
over MCP.

The server runs colocated with the harness, inside the rollout's container,
and only ever builds the visible periods of the task's dataset. Each backtest
rebuilds them in a fresh scoring worker, which runs the agent's code in
`quantenv.sandbox` as the unprivileged sandbox user; so does each check, on
a scrambled copy.

Run as `python -m quant_research.servers.research`; verifiers launches it.
"""

import asyncio
import functools
from typing import Literal

import verifiers.v1 as vf
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from quantenv import docs as libdocs
from quantenv import metrics, scoring, walkforward
from quantenv.alpha import AlphaError
from quantenv.panel import EvalData, Panel
from quantenv.sources import synthetic
from quantenv.walkforward import WalkForward

BUDGET = 10
SANDBOX_USER = "nobody"
# The libraries the alpha may use, by the names a `docs` lookup may start
# with: the prompt's aliases and the import names.
DOC_PACKAGES = {
    "pl": "polars",
    "polars": "polars",
    "pds": "polars_ds",
    "polars_ds": "polars_ds",
    "sklearn": "sklearn",
    "scipy": "scipy",
    "np": "numpy",
    "numpy": "numpy",
}
# The most characters a `docs` lookup returns, as for a backtest's error.
DOC_LIMIT = 2000
# Tools offered only when their `ResearchConfig` flag is on.
OPTIONAL_TOOLS = ("docs", "check")


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
        variation: The task's variation seed as hex
            (`synthetic.variation_seed`), or "" for the default market.
        backtests_used: Backtest calls that ran; refused calls do not count.
        submitted: Whether `submit` has run.
        code: The submitted alpha code.
        claimed_sharpe: The submitted claim of the net Sharpe on unseen data.
    """

    seed: str = ""
    variation: str = ""
    backtests_used: int = 0
    submitted: bool = False
    code: str = ""
    claimed_sharpe: float | None = None


# What the toolset records; it only reads the seed and the variation.
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
        docs: Offer the `docs` tool. Off by default: the untrained model
            did worse with it (results/day10).
        check: Offer the `check` tool. Off by default: untrained models did
            not gain from it (results/day11).
    """

    colocated: bool = True
    budget: int = Field(BUDGET, ge=0)
    sandbox_user: str | None = SANDBOX_USER
    docs: bool = False
    check: bool = False

    @property
    def optional_tools(self) -> tuple[str, ...]:
        """The optional tools that are on, by name."""
        return tuple(name for name in OPTIONAL_TOOLS if getattr(self, name))


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


def backtest_scope(data: EvalData, wf: WalkForward = WalkForward()) -> str:
    """What a backtest measures on the visible data, in one sentence.

    Args:
        data: The visible data.
        wf: Walk-forward settings.

    Returns:
        The predicted timestamps, the window and refit cadence, and how
        Sharpe ratios are annualised and costed.
    """
    stamps = data.panel.timestamps
    start = walkforward.start_timestamp(data.panel, wf)
    n_scored = (stamps >= start).sum()
    return (
        f"walk-forward on the last {n_scored} of {stamps.len()} visible "
        f"timestamps ({start} to {stamps.last()}); each block of "
        f"{wf.refit_every} is predicted by a fresh fit on the {wf.window} "
        f"timestamps before it. Sharpe ratios are annualised with "
        f"{data.periods_per_year:g} periods per year; net Sharpe pays "
        f"{data.cost * 1e4:g} bp per unit of turnover."
    )


def backtest_report(scope: str, performance: metrics.Performance, left: int) -> str:
    """A backtest's reply: what was measured and how precise it is.

    States numbers only, never advice, since advice would coach what the
    eval measures (spec, Decision 27).

    Args:
        scope: Output of `backtest_scope`.
        performance: The backtest's performance.
        left: Backtests left.

    Returns:
        The reply text.
    """
    p = performance
    if p.years == 0:
        by_year = "fewer periods than one year"
    else:
        by_year = (
            f"min {p.year_sharpe_min:.2f}, median {p.year_sharpe_median:.2f}, "
            f"max {p.year_sharpe_max:.2f} over {p.years} consecutive blocks of "
            "about one year of periods each"
        )
    return (
        f"scope: {scope}\n"
        f"net Sharpe: {p.sharpe:.2f} ± {p.sharpe_se:.2f} (standard error)\n"
        f"gross Sharpe: {p.gross_sharpe:.2f}\n"
        f"turnover: {p.turnover:.2f} per period (sum of absolute weight changes)\n"
        f"rank IC: {p.ic:.4f} ± {p.ic_se:.4f} (standard error, "
        f"{p.periods} periods)\n"
        f"net Sharpe by year: {by_year}\n"
        f"backtests left: {left}"
    )


class ResearchToolset(vf.Toolset[ResearchConfig, ResearchState]):
    """Tools for one episode.

    The toolset keeps its own record of the episode and copies it into the
    shared state at the end of every call. State is pulled before a call and
    pushed after it, so a call that started from an older copy would
    otherwise overwrite what a concurrent call recorded. Backtests and checks
    run one at a time, since each one ends by killing every process of the
    sandbox user.
    """

    TOOL_PREFIX = None  # Advertise the tools by their bare names.

    def __init__(self, config: ResearchConfig) -> None:
        super().__init__(config)
        self._episode = ResearchState()
        self._running = asyncio.Lock()
        # What describe and the backtest scope say about the visible data,
        # built once and off the event loop.
        self._summary: tuple[str, str] | None = None

    def _visible(self) -> functools.partial:
        """The loader of the visible periods, from the seed in the state."""
        if not self.state.seed:
            raise ValueError("the task is not set up: no dataset seed")
        variation = int(self.state.variation, 16) if self.state.variation else None
        return functools.partial(
            synthetic.visible_data,
            int(self.state.seed, 16),
            synthetic.task_config(variation),
        )

    async def _summarise(self) -> tuple[str, str]:
        """The description of the visible data and the backtest scope."""
        if self._summary is None:
            data = await asyncio.to_thread(self._visible())
            self._summary = (describe_panel(data.panel), backtest_scope(data))
        return self._summary

    def register(self, mcp: MCPServer) -> None:
        """Registers the tools, leaving out optional tools that are off.

        Args:
            mcp: The server to register them on.
        """
        super().register(mcp)
        for name in OPTIONAL_TOOLS:
            if name not in self.config.optional_tools:
                mcp.remove_tool(name)

    def _publish(self) -> None:
        for name in _EPISODE_FIELDS:
            setattr(self.state, name, getattr(self._episode, name))

    @vf.tool
    async def describe(self) -> str:
        """Describes the visible data.

        Returns:
            Column names, timestamp range and number of assets.
        """
        description, _ = await self._summarise()
        return description

    @vf.tool
    async def docs(self, name: str) -> str:
        """Looks up the documentation of a polars, polars-ds, scikit-learn,
        scipy or numpy object. Does not use a backtest.

        Args:
            name: Dotted name, such as "polars.Expr.rolling_mean",
                "pds.lin_reg" or "sklearn.linear_model.Ridge".

        Returns:
            The signature and docstring of a function; for a class or module,
            its summary and public members. Cut at a fixed length.

        Raises:
            ValueError: If the name is refused or does not exist.
        """
        try:
            return await asyncio.to_thread(
                libdocs.lookup, name, DOC_PACKAGES, DOC_LIMIT
            )
        except LookupError as e:
            raise ValueError(str(e)) from None

    @vf.tool
    async def check(self, code: str) -> str:
        """Runs alpha code walk-forward, as `backtest` does, on scrambled
        visible data: every column shuffled across rows on its own, so it
        has the real columns, values and size but no signal.

        Does not count against the backtest budget and reports no scores.
        Refused after `submit`.

        Args:
            code: Python source defining one subclass of `Alpha`.

        Returns:
            That the code ran, or why it failed.

        Raises:
            ValueError: If the call is refused.
        """
        if self._episode.submitted:
            raise ValueError("refused: the alpha is already submitted")
        try:
            async with self._running:
                await asyncio.to_thread(
                    scoring.check, code, self._visible(), user=self.config.sandbox_user
                )
        except AlphaError as e:
            return f"error: {e}"
        return "ok: the alpha ran on every block"

    @vf.tool
    async def backtest(self, code: str) -> str:
        """Scores alpha code walk-forward on the visible periods.

        Each call counts against the backtest budget, including calls whose
        code fails. Calls past the budget or after `submit` are refused.

        Args:
            code: Python source defining one subclass of `Alpha`.

        Returns:
            What was measured; net Sharpe with its standard error, gross
            Sharpe and turnover; mean rank IC with its standard error; the
            range of per-year net Sharpe; or why the code failed. Then the
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
            async with self._running:
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
        _, scope = await self._summarise()
        return backtest_report(scope, performance, left)

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
