"""Results table and claim-gap plot from model runs and baselines.

Reads the `traces.jsonl` of each run directory written by
`scripts/run_models.py` (one agent per run, named after the run), and
optionally the per-task rows of `scripts/run_baselines.py --out`. Episodes
that ended in an infrastructure error (the trace is not `ok`) are left out of
the table and counted separately: the model is not to blame.

Usage:
    uv run python scripts/results.py outputs/models/qwen3_5-9b [...]
        [--baselines baselines.csv] [--rows rows.csv] [--plot claim_gap.png]
"""

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from quantenv import report

REWARD = "hidden_sharpe"
# Run names of one model at several budgets end like this (run_models.py).
BUDGET_SUFFIX = r"-budget\d+$"
# Trace metric names, by the column `report.table` reads.
METRICS = {
    "sharpe": "sharpe",
    "gross_sharpe": "gross_sharpe",
    "turnover": "turnover",
    "ic": "raw_ic",
    "claim_gap": "claim_gap",
    "used_leak": "used_leak",
    "lookahead": "lookahead",
    "implausible_ic": "implausible_ic",
    "invalid": "invalid",
    "backtests_used": "backtests_used",
}


def episode_row(agent: str, trace: dict[str, Any]) -> dict[str, Any]:
    """One table row from one episode's trace.

    Args:
        agent: Who ran the episode.
        trace: An element of a `traces.jsonl` record's "traces".

    Returns:
        The agent, the task, `report.COLUMNS`, and the episode's status.
    """
    metrics = trace.get("metrics") or {}
    reward = (trace.get("rewards") or {}).get(REWARD) or {}
    return {
        "agent": agent,
        "task": trace["task"]["data"]["index"],
        "reward": reward.get("score"),
        **{col: metrics.get(name) for col, name in METRICS.items()},
        "ok": trace["ok"],
        "stop": trace.get("stop_condition"),
        "error": "; ".join(e.get("message", str(e)) for e in trace.get("errors", [])),
        "output_tokens": trace.get("num_output_tokens"),
    }


def run_rows(run_dir: Path) -> pl.DataFrame:
    """Every episode under a run directory, named after it: one run, or a
    model's chunks as `run_models.py` writes them."""
    rows = []
    for traces in sorted(run_dir.rglob("traces.jsonl")):
        with traces.open() as f:
            for line in f:
                for trace in json.loads(line)["traces"]:
                    rows.append(episode_row(run_dir.name, trace))
    return pl.DataFrame(rows, infer_schema_length=None)


def claim_gap_by_backtests(rows: pl.DataFrame) -> pl.DataFrame:
    """Mean claim gap per model and number of backtests used.

    Runs of one model at several budgets are pooled under the model's name.
    Only valid submissions have a claim gap.

    Args:
        rows: Scored episodes with "agent", "backtests_used" and "claim_gap".

    Returns:
        One row per model and backtest count: episodes, mean claim gap and
        its standard error.
    """
    return (
        rows.lazy()
        .with_columns(pl.col("claim_gap").cast(pl.Float64).fill_nan(None))
        .filter(pl.col("claim_gap").is_not_null())
        .with_columns(model=pl.col("agent").str.replace(BUDGET_SUFFIX, ""))
        .group_by("model", "backtests_used")
        .agg(
            pl.len().alias("episodes"),
            pl.col("claim_gap").mean().alias("claim_gap"),
            (pl.col("claim_gap").std() / pl.len().sqrt()).alias("se"),
        )
        .sort("model", "backtests_used")
        .collect()
    )


def plot(rows: pl.DataFrame, path: Path) -> None:
    """Claim gap against backtests used: episodes as dots, per-model means
    with standard errors as lines. Budgets of one model share a colour.

    Args:
        rows: Scored episodes with "agent", "backtests_used" and "claim_gap".
        path: Image file to write.
    """
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = rows.with_columns(
        pl.col("claim_gap").cast(pl.Float64).fill_nan(None),
        model=pl.col("agent").str.replace(BUDGET_SUFFIX, ""),
    ).filter(pl.col("claim_gap").is_not_null())
    means = claim_gap_by_backtests(rows)
    models = valid.get_column("model").unique().sort().to_list()
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for i, model in enumerate(models):
        mine = valid.filter(pl.col("model") == model)
        # Spread models sideways so equal backtest counts do not overlap.
        offset = (i - (len(models) - 1) / 2) * 0.08
        ax.scatter(
            mine.get_column("backtests_used") + offset,
            mine.get_column("claim_gap"),
            s=14,
            alpha=0.4,
            color=f"C{i}",
        )
        line = means.filter(pl.col("model") == model)
        ax.errorbar(
            line.get_column("backtests_used") + offset,
            line.get_column("claim_gap"),
            yerr=line.get_column("se").fill_null(0.0),
            marker="o",
            capsize=3,
            color=f"C{i}",
            label=model,
        )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Backtests used")
    ax.set_ylabel("Claim gap (claimed − realised net Sharpe)")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)


def main() -> None:
    """Prints the table; writes the rows and the plot if asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", type=Path, help="run directories")
    parser.add_argument("--baselines", type=Path, help="run_baselines.py --out CSV")
    parser.add_argument("--rows", type=Path, help="write per-episode rows here")
    parser.add_argument("--plot", type=Path, help="write the claim-gap plot here")
    args = parser.parse_args()

    frames = [run_rows(d) for d in args.runs]
    if args.baselines:
        frames.append(
            pl.read_csv(args.baselines)
            .rename({"baseline": "agent"})
            .with_columns(ok=pl.lit(True), stop=pl.lit("baseline"))
        )
    if not frames:
        parser.error("give run directories, --baselines, or both")
    columns = ["agent", "task", *report.COLUMNS, "ok", "stop"]
    rows = pl.concat([f.select(columns) for f in frames], how="vertical_relaxed")
    if args.rows:
        rows.write_csv(args.rows)

    errored = rows.filter(~pl.col("ok"))
    scored = rows.filter(pl.col("ok"))
    with pl.Config(**report.MARKDOWN):
        print(report.table(scored))
        stops = scored.group_by("agent", "stop").len().sort("agent", "stop")
        print("\nHow episodes ended:")
        print(stops)
    with pl.Config(**report.MARKDOWN):
        print("\nClaim gap by backtests used (valid submissions):")
        print(claim_gap_by_backtests(scored))
    if errored.height:
        print(f"\nLeft out, infrastructure errors: {errored.height} episodes")
        print(errored.group_by("agent").len().sort("agent"))
    if args.plot:
        plot(scored, args.plot)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()
