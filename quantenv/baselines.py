"""Scripted baselines. Each returns alpha code and a claimed IC, the same
submission an agent makes."""

import polars as pl

from quantenv import metrics
from quantenv.panel import Panel

_COMBINED = "_combined"


def column_code(name: str, sign: float = 1.0) -> str:
    """Alpha code that predicts with one column, optionally negated."""
    expr = f"pl.col({name!r})" if sign >= 0 else f"-pl.col({name!r})"
    return (
        "class ColumnAlpha(Alpha):\n"
        "    def predict(self, df):\n"
        f"        return {expr}\n"
    )


def weighted_code(
    weights: dict[str, float],
    interactions: dict[tuple[str, str], float] | None = None,
) -> str:
    """Alpha code for a fixed mix of columns and products of column pairs.

    Args:
        weights: Column to weight.
        interactions: Column pair to the weight of their product.

    Returns:
        The alpha code.
    """
    terms = [f"{w!r} * pl.col({k!r})" for k, w in weights.items()]
    terms += [
        f"{w!r} * pl.col({a!r}) * pl.col({b!r})"
        for (a, b), w in (interactions or {}).items()
    ]
    return (
        "class WeightedAlpha(Alpha):\n"
        "    def predict(self, df):\n"
        f"        return pl.sum_horizontal({', '.join(terms)})\n"
    )


def template_code(features: list[str], cap: float = 0.1) -> str:
    """The textbook alpha: correlate, drop outliers, weight.

    Each refit, every feature's mean rank IC on the training rows is
    measured; features at or above `cap` are dropped as too good to be true
    (a leak), and the rest are weighted by their IC.

    Args:
        features: Candidate columns.
        cap: |IC| at or above which a feature is dropped.

    Returns:
        The alpha code.
    """
    return (
        "class Template(Alpha):\n"
        f"    names = {features!r}\n"
        "    def fit(self, train):\n"
        "        ics = (train.group_by('timestamp')\n"
        "               .agg(pl.corr(c, 'fwd_return', method='spearman') for c in self.names)\n"
        "               .select(pl.col(self.names).fill_nan(0.0).mean()).row(0, named=True))\n"
        f"        self.w = {{c: v for c, v in ics.items() if abs(v) < {cap!r}}}\n"
        "    def predict(self, df):\n"
        "        return pl.sum_horizontal(w * pl.col(c) for c, w in self.w.items())\n"
    )


def naive(panel: Panel) -> tuple[str, float]:
    """The single feature with the best in-sample IC.

    A feature with a negative IC is negated. On a seed with a leaky feature
    this picks the leak almost always.

    Args:
        panel: The visible panel.

    Returns:
        Alpha code and its in-sample IC as the claim.
    """
    s = panel.schema
    ics = metrics.mean_rank_ics(
        panel.frame, {f: (f, s.target) for f in s.features}, s.timestamp
    )
    best = max(ics, key=lambda k: abs(ics[k]))
    return column_code(best, ics[best]), abs(ics[best])


def oracle(weights: dict[str, float], panel: Panel) -> tuple[str, float]:
    """The real features weighted by their true weights.

    Args:
        weights: True weights, from the simulator's ground truth.
        panel: The visible panel, for the claim.

    Returns:
        Alpha code and its in-sample IC as the claim.
    """
    s = panel.schema
    combined = pl.sum_horizontal(w * pl.col(k) for k, w in weights.items())
    claim = metrics.rank_ic(
        panel.frame.with_columns(combined.alias(_COMBINED)),
        _COMBINED,
        s.target,
        s.timestamp,
    )
    return weighted_code(weights), claim
