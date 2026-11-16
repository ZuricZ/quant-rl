"""Scores of predictions against returns. Generic over any long frame."""

import dataclasses
import math

import polars as pl

_WEIGHT = "_weight"
_PNL = "_pnl"
# Robust Sharpe: P&L is winsorised at the median plus or minus this many
# robust standard deviations (1.4826 x MAD, the std for Gaussian data).
WINSOR_SIGMAS = 5.0
_MAD_TO_SD = 1.4826


def period_rank_ics(
    frame: pl.DataFrame, pairs: dict[str, tuple[str, str]], timestamp: str
) -> pl.DataFrame:
    """Cross-sectional rank IC per period, for several column pairs at once.

    For each timestamp, the Spearman correlation across assets of x with y.
    The pairs only batch several such correlations into one pass over the
    frame, e.g. predictions against returns at three shifts, or every
    feature against the target. Null pairs are ignored. A period with fewer
    than two usable rows or a constant column gets IC 0, so every value is
    finite.

    Args:
        frame: Long frame with the timestamp and every named column.
        pairs: Output name to (x column, y column).
        timestamp: Time column to group by.

    Returns:
        One row per period, in time order: the timestamp and one IC column
        per pair.
    """
    return (
        frame.lazy()
        .group_by(timestamp, maintain_order=True)
        .agg(pl.corr(x, y, method="spearman").alias(k) for k, (x, y) in pairs.items())
        .with_columns(pl.col(list(pairs)).fill_nan(0.0).fill_null(0.0))
        .collect()
    )


def mean_rank_ics(
    frame: pl.DataFrame, pairs: dict[str, tuple[str, str]], timestamp: str
) -> dict[str, float]:
    """Mean per-period Spearman correlation for several column pairs, in one pass.

    Null pairs are ignored. A period with fewer than two usable rows or a
    constant column counts as IC 0 and stays in the average, so every result
    is finite.

    TODO: This estimate is biased towards 0. Revisit before trusting small
    ICs or comparing alphas with different coverage. Sources:
      - Periods with no usable rows or constant predictions count as 0 and stay
        in the denominator, so an alpha that covers fewer periods is shrunk.
      - Nulls are dropped per period, so sparse predictions are scored on fewer
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
        period_rank_ics(frame, pairs, timestamp)
        .drop(timestamp)
        .mean()
        .row(0, named=True)
    )


def rank_ic(frame: pl.DataFrame, x: str, y: str, timestamp: str) -> float:
    """Mean per-period Spearman correlation of `x` with `y`. See mean_rank_ics."""
    return mean_rank_ics(frame, {"ic": (x, y)}, timestamp)["ic"]


@dataclasses.dataclass(frozen=True)
class Performance:
    """How a dollar-neutral portfolio built from predictions performed.

    Sharpe ratios are annualised and are 0 when P&L does not vary.

    Attributes:
        ic: Mean per-period rank IC (see `mean_rank_ics`).
        sharpe: Net of costs.
        robust_sharpe: Net of costs, on winsorised P&L (see `robust_sharpe`).
        gross_sharpe: Before costs.
        turnover: Mean per-period turnover, sum of |weight change|; 2 is a
            full reversal of a unit-gross book.
        year_sharpe_min: Lowest net Sharpe of a calendar year.
        year_sharpe_median: Median net Sharpe of the calendar years.
        year_sharpe_max: Highest net Sharpe of a calendar year.
        max_drawdown: Largest fall of cumulative net P&L from its running
            peak (starting at 0), in return units of a unit-gross book.
    """

    ic: float
    sharpe: float
    robust_sharpe: float
    gross_sharpe: float
    turnover: float
    year_sharpe_min: float
    year_sharpe_median: float
    year_sharpe_max: float
    max_drawdown: float


def rank_weight(prediction: str, timestamp: str) -> pl.Expr:
    """Dollar-neutral rank weights with unit gross exposure, per period.

    Within each period, predictions are ranked, the ranks centred on their
    mean and scaled so the absolute weights sum to 1. Null predictions, and
    periods whose predictions are all equal, get weight 0.

    Args:
        prediction: Prediction column.
        timestamp: Time column.

    Returns:
        An expression giving one weight per row.
    """
    rank = pl.col(prediction).rank().over(timestamp)
    centred = rank - rank.mean().over(timestamp)
    weight = centred / centred.abs().sum().over(timestamp)
    return weight.fill_nan(0.0).fill_null(0.0)


def sharpe(pnl: pl.Expr, periods_per_year: float) -> pl.Expr:
    """Annualised Sharpe ratio of per-period P&L; 0 if it does not vary.

    NOTE: Biased towards 0 by how nulls are handled upstream: null
    predictions get weight 0 (`rank_weight`), so periods or assets an alpha
    does not cover add flat P&L, and P&L that never varies scores 0.
    """
    ratio = pnl.mean() / pnl.std() * math.sqrt(periods_per_year)
    return ratio.fill_nan(0.0).fill_null(0.0)


def robust_sharpe(pnl: pl.Expr, periods_per_year: float) -> pl.Expr:
    """Annualised Sharpe ratio of winsorised per-period P&L.

    P&L is clipped to the median plus or minus `WINSOR_SIGMAS` robust
    standard deviations (1.4826 x the median absolute deviation, or the plain
    standard deviation when that is 0, as for mostly flat P&L). For Gaussian
    P&L this clips almost nothing and equals `sharpe`; with fat tails, a few
    extreme periods cannot carry the mean or the spread.

    Args:
        pnl: Per-period P&L.
        periods_per_year: For annualising.

    Returns:
        The robust Sharpe ratio; 0 if the clipped P&L does not vary.
    """
    median = pnl.median()
    mad = (pnl - median).abs().median()
    scale = pl.when(mad > 0).then(_MAD_TO_SD * mad).otherwise(pnl.std())
    bound = WINSOR_SIGMAS * scale
    return sharpe(pnl.clip(median - bound, median + bound), periods_per_year)


def performance(
    frame: pl.DataFrame,
    prediction: str,
    target: str,
    timestamp: str,
    asset: str,
    cost: float,
    periods_per_year: float,
) -> Performance:
    """Trades `rank_weights` of the predictions each period.

    The P&L of period t is the weights times the target (the return from t
    to t + 1), minus `cost` times turnover, the sum over assets of the
    absolute weight change from t - 1 (flat before the first period).

    Args:
        frame: Dense long frame sorted by timestamp then asset, Date or
            Datetime timestamps.
        prediction: Prediction column.
        target: Next-period return column.
        timestamp: Time column.
        asset: Asset column.
        cost: Cost per unit of turnover, in return units (0.0005 is 5 bp).
        periods_per_year: For annualising, e.g. 252 for daily bars.

    Returns:
        The performance.
    """
    weight = pl.col(_WEIGHT)
    periods = (
        frame.lazy()
        .select(timestamp, asset, target, prediction)
        .with_columns(rank_weight(prediction, timestamp).alias(_WEIGHT))
        .with_columns(
            (weight - weight.shift(1).over(asset).fill_null(0.0)).abs().alias("_trade")
        )
        .group_by(timestamp, maintain_order=True)
        .agg(
            (weight * pl.col(target)).sum().alias("gross"),
            pl.col("_trade").sum().alias("turnover"),
            pl.corr(prediction, target, method="spearman")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("ic"),
        )
        .with_columns((pl.col("gross") - cost * pl.col("turnover")).alias(_PNL))
    )
    years = (
        periods.group_by(pl.col(timestamp).dt.year())
        .agg(sharpe(pl.col(_PNL), periods_per_year).alias("sharpe"), pl.len())
        .filter(pl.col("len") > 1)
        .select(
            pl.col("sharpe").min().alias("year_sharpe_min"),
            pl.col("sharpe").median().alias("year_sharpe_median"),
            pl.col("sharpe").max().alias("year_sharpe_max"),
        )
    )
    cumulative = pl.col(_PNL).cum_sum()
    totals = periods.select(
        pl.col("ic").mean(),
        sharpe(pl.col(_PNL), periods_per_year).alias("sharpe"),
        robust_sharpe(pl.col(_PNL), periods_per_year).alias("robust_sharpe"),
        sharpe(pl.col("gross"), periods_per_year).alias("gross_sharpe"),
        pl.col("turnover").mean(),
        (cumulative.cum_max().clip(lower_bound=0.0) - cumulative)
        .max()
        .alias("max_drawdown"),
    )
    row = pl.concat([totals, years], how="horizontal").collect().row(0, named=True)
    return Performance(**{k: float(v) for k, v in row.items()})
