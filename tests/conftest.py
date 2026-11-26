"""Shared fixtures."""

import dataclasses

import pytest

from quantenv.sources import synthetic

# The smallest market the default walk-forward and checks run on: a
# 1,000-timestamp training window, then two 250-timestamp blocks.
SMALL_MARKET = dataclasses.replace(
    synthetic.DEFAULT_CONFIG, n_assets=50, n_visible=1_200, n_hidden=500
)


@pytest.fixture(scope="module")
def small_market():
    """Makes every task's market small, varied or not, wherever it is drawn:
    the tasksets, the tools and the grader all call `synthetic.task_config`.
    Module-scoped, so module-scoped task fixtures get it too.

    Yields:
        The base market; a task without variation gets exactly this one.
    """
    task_config = synthetic.task_config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(synthetic, "_ASSETS", (SMALL_MARKET.n_assets,))
        patch.setattr(synthetic, "_VISIBLE", (SMALL_MARKET.n_visible,))
        patch.setattr(
            synthetic,
            "task_config",
            lambda variation, base=SMALL_MARKET: task_config(variation, base),
        )
        yield SMALL_MARKET
