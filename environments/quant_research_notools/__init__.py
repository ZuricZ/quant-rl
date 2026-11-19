"""Quant research without tools: write an alpha in one reply, scored by IC.

The prime-rl smoke run's taskset. `NullHarness` is exported so that runs
default to the `null` harness, a plain chat loop; with no tools it ends after
the model's first reply.
"""

from verifiers.v1.harnesses.null import NullHarness

from quant_research_notools.taskset import NoToolsTask, NoToolsTaskset

__all__ = ["NoToolsTask", "NoToolsTaskset", "NullHarness"]
