"""Scripted baselines. Each returns signal code and a claimed IC, the same
submission an agent makes."""

from quantenv import engine
from quantenv.generator import Dataset, GroundTruth


def column_code(name: str, sign: float = 1.0) -> str:
    """Signal code that returns one feature column, optionally negated."""
    expr = f"pl.col({name!r})" if sign >= 0 else f"-pl.col({name!r})"
    return f"def signal(df):\n    return {expr}\n"


def naive(visible: Dataset) -> tuple[str, float]:
    """The single feature with the best visible-period IC.

    A feature with a negative IC is negated. On a seed with a leaky feature
    this picks the leak almost always.

    Args:
        visible: The visible split.

    Returns:
        Signal code and its visible-period IC as the claim.
    """
    ics = engine.feature_ics(visible)
    best = max(ics, key=lambda k: abs(ics[k]))
    return column_code(best, ics[best]), abs(ics[best])


def oracle_code(truth: GroundTruth) -> str:
    """Signal code for the optimal linear mix of the real features."""
    terms = ", ".join(f"{w!r} * pl.col({k!r})" for k, w in truth.weights.items())
    return f"def signal(df):\n    return pl.sum_horizontal({terms})\n"


def oracle(visible: Dataset) -> tuple[str, float]:
    """The real features weighted by their true weights.

    Args:
        visible: The visible split; its ground truth gives the weights.

    Returns:
        Signal code and its visible-period IC as the claim.
    """
    code = oracle_code(visible.truth)
    return code, engine.score_code(code, visible)
