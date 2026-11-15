"""Scores the scripted baselines on a split's tasks and prints the results table.

Baselines:
- naive: the single feature with the best visible-period IC, claiming that IC.
- oracle: the real features at their true weights.
- leaky: the leaky feature alone, claiming its visible-period backtest IC.
  Only datasets that have a leaky feature are scored.

Task i uses dataset `synthetic.dataset_seed(i, secret, split)`, as in the
taskset; pass the taskset's secret and split to score the same datasets as a
model run.
Every submission goes through `quantenv.scoring.score_submission` on the
hidden periods, the same path as the environment's reward. It imports no
verifiers code, so it also runs on Windows.

Usage:
    uv run python scripts/run_baselines.py --secret S [--split eval] [--tasks 30]
        [--workers 4]
        [--out rows.csv]
"""

import argparse
import concurrent.futures
import functools

import polars as pl

from quantenv import baselines, scoring
from quantenv.sources import synthetic


def submissions(seed: int) -> dict[str, tuple[str, float, int]]:
    """The baselines' submissions for one seed.

    Args:
        seed: Dataset seed.

    Returns:
        Baseline name to alpha code, claimed IC and backtests used.
    """
    syn = synthetic.generate(seed)
    naive_code, naive_claim = baselines.naive(syn.panel)
    oracle_code, oracle_claim = baselines.oracle(syn.truth.weights, syn.panel)
    out = {
        "naive": (naive_code, naive_claim, 0),
        "oracle": (oracle_code, oracle_claim, 0),
    }
    if syn.truth.leaky is not None:
        code = baselines.column_code(syn.truth.leaky)
        visible = functools.partial(synthetic.visible_data, seed)
        out["leaky"] = (code, scoring.backtest(code, visible), 1)
    return out


def score_task(index: int, secret: str, split: str) -> list[dict]:
    """Scores every baseline on one task.

    Args:
        index: Task number.
        secret: The taskset's secret.
        split: The taskset's split.

    Returns:
        One row per baseline.
    """
    seed = synthetic.dataset_seed(index, secret, split)
    hidden = functools.partial(synthetic.hidden_data, seed)
    rows = []
    for name, (code, claim, used) in submissions(seed).items():
        score = scoring.score_submission(code, claim, hidden)
        rows.append(
            {
                "baseline": name,
                "task": index,
                "reward": score.reward,
                "claim_gap": score.claim_gap,
                "used_leak": score.used_leak,
                "lookahead": score.lookahead,
                "invalid": score.invalid,
                "backtests_used": used,
            }
        )
    return rows


def table(rows: pl.DataFrame) -> pl.DataFrame:
    """The spec's results table, one row per baseline.

    Args:
        rows: Output rows of `score_task`.

    Returns:
        Means per baseline.
    """
    return (
        rows.lazy()
        .group_by("baseline")
        .agg(
            pl.len().alias("Tasks"),
            pl.col("used_leak").mean().alias("Leak rate"),
            pl.col("lookahead").mean().alias("Lookahead rate"),
            pl.col("reward").mean().alias("True IC"),
            pl.col("claim_gap").mean().alias("Claim gap"),
            pl.col("backtests_used").mean().alias("Backtests used"),
        )
        .sort("baseline")
        .rename({"baseline": "Baseline"})
        .collect()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--secret", required=True, help="the taskset's secret")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--tasks", type=int, default=30, help="score tasks 0..N-1")
    parser.add_argument("--workers", type=int, default=4, help="tasks in parallel")
    parser.add_argument("--out", help="write the per-task rows to this CSV")
    args = parser.parse_args()

    # Each scoring call runs in its own process, so threads are enough here.
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        per_task = []
        score = functools.partial(score_task, secret=args.secret, split=args.split)
        for task_rows in pool.map(score, range(args.tasks)):
            per_task.extend(task_rows)
            print(f"scored task {task_rows[0]['task']}", flush=True)
    rows = pl.DataFrame(per_task)
    if args.out:
        rows.write_csv(args.out)
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        float_precision=4,
    ):
        print(table(rows))


if __name__ == "__main__":
    main()
