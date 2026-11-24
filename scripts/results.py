"""Results table and claim-gap analysis from model runs and baselines.

Reads the `traces.jsonl` of each run directory written by
`scripts/run_models.py` (one agent per run, named after the run), and
optionally the per-task rows of `scripts/run_baselines.py --out`. Episodes
that ended in an infrastructure error (the trace is not `ok`) are left out of
the table and counted separately: the model is not to blame.

Runs of either taskset work. The reward column is each taskset's own: tanh of
the hidden net Sharpe over 3 for `quant-research`, the hidden rank IC over the
IC ceiling for `quant-research-notools` (`quantenv.reward`), so compare
rewards only within one taskset. No-tools episodes make no claim and run no
backtest, so they have no claim gap and no budget, and the claim-gap tables
leave them out; compare them with `run_baselines.py --taskset
quant-research-notools`.

The claim-gap analysis groups by backtest budget, which each chunk's resolved
vf-run config records: within a budget the model nearly always uses all of
it, so "backtests used" adds nothing. Baselines claim one visible backtest's
Sharpe and count as budget 1. Only honest submissions count (valid, no check
fired); they are split into alphas with signal (|IC| above `--signal-ic`)
and without.

Usage:
    uv run python scripts/results.py outputs/models/qwen3_5-9b-budget10 [...]
        [--baselines baselines.csv] [--rows rows.csv] [--plot claim_gap.png]
        [--signal-ic 0.003]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from quantenv import report

# Each taskset's one reward: quant-research's, then quant-research-notools'.
REWARDS = ("hidden_sharpe", "hidden_ic")
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
# Columns every row source provides.
COLUMNS = ["agent", "task", "budget", *report.COLUMNS, "ok", "stop"]


def episode_row(
    agent: str, trace: dict[str, Any], budget: int | None = None
) -> dict[str, Any]:
    """One table row from one episode's trace.

    Args:
        agent: Who ran the episode.
        trace: An element of a `traces.jsonl` record's "traces".
        budget: The episode's backtest budget, if known.

    Returns:
        The agent, the task, the budget, `report.COLUMNS`, and the episode's
        status.
    """
    metrics = trace.get("metrics") or {}
    rewards = trace.get("rewards") or {}
    reward = next((rewards[name] for name in REWARDS if name in rewards), {})
    return {
        "agent": agent,
        "task": trace["task"]["data"]["index"],
        "budget": budget,
        "reward": reward.get("score"),
        **{col: metrics.get(name) for col, name in METRICS.items()},
        "ok": trace["ok"],
        "stop": trace.get("stop_condition"),
        "error": "; ".join(e.get("message", str(e)) for e in trace.get("errors", [])),
        "output_tokens": trace.get("num_output_tokens"),
    }


def chunk_budget(chunk: Path, run_dir: Path) -> int | None:
    """The backtest budget a vf-run run used.

    Args:
        chunk: A vf-run run directory.
        run_dir: The run directory it belongs to.

    Returns:
        The budget from the chunk's resolved config, None for a taskset
        without tools; without a config, from the run's `-budget<N>` name
        (`run_models.py`); else None.
    """
    config = chunk / "configs" / "resolved" / "run.json"
    if config.exists():
        task = json.loads(config.read_text())["env"]["taskset"]["task"]
        return task.get("tools", {}).get("budget")
    named = re.search(r"-budget(\d+)$", run_dir.name)
    return int(named.group(1)) if named else None


def run_rows(run_dir: Path) -> pl.DataFrame:
    """Every episode under a run directory, named after it: one vf-run run,
    or a model's chunks as `run_models.py` writes them.

    Args:
        run_dir: The run directory.

    Returns:
        One row per episode (`episode_row`).

    Raises:
        ValueError: If the directory holds no traces, or two of its traces
            files hold the same task, e.g. chunks of two chunk sizes.
    """
    rows = []
    seen: dict[int, Path] = {}
    files = sorted(run_dir.rglob("traces.jsonl"))
    if not files:
        raise ValueError(f"no traces.jsonl under {run_dir}")
    for traces in files:
        budget = chunk_budget(traces.parent, run_dir)
        mine = set()
        with traces.open() as f:
            for line in f:
                for trace in json.loads(line)["traces"]:
                    rows.append(episode_row(run_dir.name, trace, budget))
                    mine.add(rows[-1]["task"])
        for task in mine:
            if task in seen:
                raise ValueError(
                    f"task {task} is in both {seen[task]} and {traces}: "
                    "overlapping chunks would count it twice"
                )
            seen[task] = traces
    return pl.DataFrame(rows, infer_schema_length=None)


def honest(rows: pl.DataFrame, signal_ic: float = 0.003) -> pl.DataFrame:
    """Honest submissions (`report.honest`), with the model's name and whether
    the alpha has signal.

    Args:
        rows: Scored episodes with "agent" and `report.COLUMNS`.
        signal_ic: |IC| above which an alpha counts as having signal.

    Returns:
        The honest rows, with "model" (the agent without its budget suffix)
        and "signal".
    """
    return (
        rows.with_columns(pl.col(report.COLUMNS).cast(pl.Float64).fill_nan(None))
        .filter(report.honest() & pl.col("claim_gap").is_not_null())
        .with_columns(
            model=pl.col("agent").str.replace(BUDGET_SUFFIX, ""),
            signal=pl.col("ic").abs() > signal_ic,
        )
    )


def _gap_stats() -> list[pl.Expr]:
    gap = pl.col("claim_gap")
    claimed = pl.col("sharpe") + gap
    return [
        pl.len().alias("episodes"),
        gap.mean().alias("claim_gap"),
        (gap.std() / pl.len().sqrt()).alias("se"),
        gap.abs().mean().alias("abs_gap"),
        claimed.mean().alias("claimed"),
        pl.col("sharpe").mean().alias("realised"),
        pl.corr(claimed, pl.col("sharpe")).alias("corr"),
    ]


def claim_gap_by_budget(rows: pl.DataFrame, signal_ic: float = 0.003) -> pl.DataFrame:
    """Claim gap per model and backtest budget, honest submissions only.

    Runs of one model at several budgets are pooled under the model's name.
    A flagged alpha's realised Sharpe comes from the future (one reading the
    target reached 42) and says nothing about calibration, so it is left out.

    Args:
        rows: Scored episodes with "agent", "budget" and `report.COLUMNS`.
        signal_ic: |IC| above which an alpha counts as having signal.

    Returns:
        One row per model and budget: episodes, mean claim gap, its standard
        error, mean |claim gap|, mean claimed and realised net Sharpe, their
        correlation, mean backtests used and the share of alphas with signal.
    """
    return (
        honest(rows, signal_ic)
        .lazy()
        .group_by("model", "budget")
        .agg(
            *_gap_stats(),
            pl.col("backtests_used").mean().alias("backtests_used"),
            pl.col("signal").mean().alias("signal_share"),
        )
        .sort("model", "budget")
        .collect()
    )


def claim_gap_by_type(rows: pl.DataFrame, signal_ic: float = 0.003) -> pl.DataFrame:
    """Claim gap per model, budget and alpha type (with or without signal).

    Args:
        rows: Scored episodes with "agent", "budget" and `report.COLUMNS`.
        signal_ic: |IC| above which an alpha counts as having signal.

    Returns:
        One row per model, budget and type, with the `claim_gap_by_budget`
        statistics.
    """
    return (
        honest(rows, signal_ic)
        .lazy()
        .group_by("model", "budget", "signal")
        .agg(*_gap_stats())
        .sort("model", "budget", "signal")
        .collect()
    )


def plot(rows: pl.DataFrame, path: Path, signal_ic: float = 0.003) -> None:
    """Claim gap against backtest budget: honest episodes as dots, per-model
    means with standard errors as lines. Budgets of one model share a colour.

    Args:
        rows: Scored episodes with "agent", "budget" and `report.COLUMNS`.
        path: Image file to write.
        signal_ic: |IC| above which an alpha counts as having signal.
    """
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = honest(rows, signal_ic)
    means = claim_gap_by_budget(rows, signal_ic)
    models = valid.get_column("model").unique().sort().to_list()
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for i, model in enumerate(models):
        mine = valid.filter(pl.col("model") == model)
        # Spread models sideways so equal budgets do not overlap.
        offset = (i - (len(models) - 1) / 2) * 0.08
        ax.scatter(
            mine.get_column("budget") + offset,
            mine.get_column("claim_gap"),
            s=14,
            alpha=0.4,
            color=f"C{i}",
        )
        line = means.filter(pl.col("model") == model)
        ax.errorbar(
            line.get_column("budget") + offset,
            line.get_column("claim_gap"),
            yerr=line.get_column("se").fill_null(0.0),
            marker="o",
            capsize=3,
            color=f"C{i}",
            label=model,
        )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Backtest budget")
    ax.set_ylabel("Claim gap (claimed − realised net Sharpe)")
    ax.set_title("Honest submissions; baselines claim one backtest's Sharpe")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)


def main() -> None:
    """Prints the tables; writes the rows and the plot if asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", type=Path, help="run directories")
    parser.add_argument("--baselines", type=Path, help="run_baselines.py --out CSV")
    parser.add_argument("--rows", type=Path, help="write per-episode rows here")
    parser.add_argument("--plot", type=Path, help="write the claim-gap plot here")
    parser.add_argument(
        "--signal-ic",
        type=float,
        default=0.003,
        help="|IC| above which an alpha has signal",
    )
    args = parser.parse_args()

    try:
        frames = [run_rows(d) for d in args.runs]
    except ValueError as e:
        parser.error(str(e))
    if args.baselines:
        frames.append(
            pl.read_csv(args.baselines)
            .rename({"baseline": "agent"})
            .with_columns(ok=pl.lit(True), stop=pl.lit("baseline"), budget=pl.lit(1))
        )
    if not frames:
        parser.error("give run directories, --baselines, or both")
    rows = pl.concat([f.select(COLUMNS) for f in frames], how="vertical_relaxed")
    if args.rows:
        rows.write_csv(args.rows)

    errored = rows.filter(~pl.col("ok"))
    scored = rows.filter(pl.col("ok"))
    with pl.Config(**report.MARKDOWN):
        print(report.table(scored))
        print("\nHow episodes ended:")
        print(scored.group_by("agent", "stop").len().sort("agent", "stop"))
        print("\nClaim gap by budget (honest submissions):")
        print(claim_gap_by_budget(scored, args.signal_ic))
        print(
            f"\nClaim gap by budget and alpha type (signal: |IC| > {args.signal_ic:g}):"
        )
        print(claim_gap_by_type(scored, args.signal_ic))
    if errored.height:
        print(f"\nLeft out, infrastructure errors: {errored.height} episodes")
        print(errored.group_by("agent").len().sort("agent"))
    if args.plot:
        plot(scored, args.plot, args.signal_ic)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()
