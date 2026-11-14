"""Quant research verifiers environment: research an alpha, score it on unseen data.

`NullHarness` is exported so that runs default to the `null` harness, a plain
chat loop with tool calls.
"""

from verifiers.v1.harnesses.null import NullHarness

from quant_research.servers.research import ResearchToolset
from quant_research.taskset import QuantResearchTask, QuantResearchTaskset

__all__ = ["NullHarness", "QuantResearchTask", "QuantResearchTaskset", "ResearchToolset"]
