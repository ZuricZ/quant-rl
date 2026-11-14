"""Scores of predictions against returns. Generic over any long frame."""

import polars as pl


def mean_rank_ics(
    frame: pl.DataFrame, pairs: dict[str, tuple[str, str]], timestamp: str
) -> dict[str, float]:
    """Mean per-period Spearman correlation for several column pairs, in one pass.

    Null pairs are ignored. A period with fewer than two usable rows or a
    constant column counts as IC 0 and stays in the average, so every result
    is finite.

    TODO: This estimate is biased towards 0. Revisit before trusting small
    ICs or comparing signals with different coverage. Sources:
      - Periods with no usable rows or a constant signal count as 0 and stay
        in the denominator, so a signal that covers fewer periods is shrunk.
      - Nulls are dropped per period, so a sparse signal is scored on fewer
        assets, which gives noisier per-period correlations.
      - Spearman is about 3/pi (~0.955) of the Pearson correlation for
        Gaussian data; the generator corrects for this, real data will not.
      - Each period's sample correlation is itself slightly biased towards 0
        (about rho * (1 - rho**2) / (2 * n) for n assets).
    Options: report coverage next to IC, average over valid periods only,
    weight periods by asset count, or Fisher-z average the per-period ICs.

    Args:
        frame: Long frame with the timestamp and every named column.
        pairs: Output name to (x column, y column).
        timestamp: Time column to group by.

    Returns:
        Output name to mean rank IC.
    """
    return (
        frame.lazy()
        .group_by(timestamp)
        .agg(pl.corr(x, y, method="spearman").alias(k) for k, (x, y) in pairs.items())
        .select(pl.col(list(pairs)).fill_nan(0.0).fill_null(0.0).mean())
        .collect()
        .row(0, named=True)
    )


def rank_ic(frame: pl.DataFrame, x: str, y: str, timestamp: str) -> float:
    """Mean per-period Spearman correlation of `x` with `y`. See mean_rank_ics."""
    return mean_rank_ics(frame, {"ic": (x, y)}, timestamp)["ic"]
