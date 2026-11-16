"""The results table: one row per agent, means over its scored episodes.

Generic over who submitted: scripted baselines and model runs give the same
per-episode rows, so one table compares them.
"""

import polars as pl

# Per-episode columns `table` reads; Float64, null where they do not apply.
COLUMNS = (
    "reward",
    "sharpe",
    "gross_sharpe",
    "turnover",
    "ic",
    "claim_gap",
    "used_leak",
    "lookahead",
    "invalid",
    "backtests_used",
)
MARKDOWN = dict(
    tbl_formatting="MARKDOWN",
    tbl_hide_column_data_types=True,
    tbl_hide_dataframe_shape=True,
    tbl_cols=-1,
    tbl_rows=-1,
    tbl_width_chars=250,
    float_precision=3,
)


def table(rows: pl.DataFrame, agent: str = "agent") -> pl.DataFrame:
    """Means per agent over its episodes.

    Performance means skip nulls, so they cover the valid submissions only;
    the rates and the reward cover every episode.

    Args:
        rows: One row per scored episode, with `agent`, "task" and `COLUMNS`.
        agent: Column naming who submitted.

    Returns:
        One row per agent, sorted by name.
    """
    return (
        rows.lazy()
        .with_columns(pl.col(COLUMNS).cast(pl.Float64).fill_nan(None))
        .group_by(agent)
        .agg(
            pl.col("task").n_unique().alias("Tasks"),
            pl.len().alias("Episodes"),
            pl.col("used_leak").mean().alias("Leak rate"),
            pl.col("lookahead").mean().alias("Lookahead rate"),
            pl.col("invalid").mean().alias("Invalid rate"),
            pl.col("reward").mean().alias("Reward"),
            pl.col("sharpe").mean().alias("Net Sharpe"),
            pl.col("gross_sharpe").mean().alias("Gross Sharpe"),
            pl.col("turnover").mean().alias("Turnover"),
            pl.col("ic").mean().alias("IC"),
            pl.col("claim_gap").mean().alias("Claim gap"),
            pl.col("backtests_used").mean().alias("Backtests used"),
        )
        .sort(agent)
        .rename({agent: "Agent"})
        .collect()
    )
