"""Scores the scripted baselines on a split's tasks and prints the results table.

Baselines, each claiming the net Sharpe of one visible-period backtest, as an
agent would:
- naive: the single feature with the best visible-period IC.
- template: the textbook alpha, refit each block: measure every feature's
  IC, drop those too good to be true, weight the rest by IC.
- oracle: the real features and their interaction at their true weights,
  from the generator's ground truth. A baseline only; never in the reward.
- leaky: the leaky feature alone. Only datasets that have one are scored.

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
import dataclasses
import functools

import polars as pl

from quantenv import baselines, metrics, report, reward, scoring
from quantenv.sources import synthetic


def submissions(seed: int) -> dict[str, tuple[str, float]]:
    """The baselines' submissions for one seed.

    Args:
        seed: Dataset seed.

    Returns:
        Baseline name to alpha code and claimed net Sharpe.
    """
    syn = synthetic.generate(seed)
    truth = syn.truth
    interactions = (
        {truth.interaction: truth.interaction_weight} if truth.interaction else {}
    )
    codes = {
        "naive": baselines.naive(syn.panel)[0],
        "template": baselines.template_code(list(syn.panel.schema.features)),
        "oracle": baselines.weighted_code(truth.weights, interactions),
    }
    if truth.leaky is not None:
        codes["leaky"] = baselines.column_code(truth.leaky)
    visible = functools.partial(synthetic.visible_data, seed)
    return {
        name: (code, scoring.backtest(code, visible).sharpe)
        for name, code in codes.items()
    }


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
    for name, (code, claim) in submissions(seed).items():
        score = scoring.score_submission(code, claim, hidden)
        if score.performance is None:
            print(f"task {index} {name}: invalid: {score.error}", flush=True)
            performance = {
                f.name: None for f in dataclasses.fields(metrics.Performance)
            }
        else:
            performance = dataclasses.asdict(score.performance)
        rows.append(
            {
                "baseline": name,
                "task": index,
                "reward": reward.reward(dataclasses.asdict(score)),
                **performance,
                "claim_gap": score.claim_gap,
                "used_leak": score.used_leak,
                "lookahead": score.lookahead,
                "invalid": score.invalid,
                "error": score.error,
                "backtests_used": 1,
            }
        )
    return rows


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
    with pl.Config(**report.MARKDOWN):
        print(report.table(rows, "baseline"))


if __name__ == "__main__":
    main()
