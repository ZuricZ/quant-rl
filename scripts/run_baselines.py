"""Scores the scripted baselines on a split's tasks and prints the results table.

Baselines, each claiming the net Sharpe of one visible-period backtest, as an
agent would:
- naive: the single feature with the best visible-period IC.
- template: the textbook alpha, refit each block: measure every feature's
  IC, drop those too good to be true, weight the rest by IC.
- oracle: the real features, their interaction and any past-return signal
  at their true weights, from the generator's ground truth. A baseline only;
  never in the reward.
- leaky: the leaky feature alone. Only datasets that have one are scored.

With `--taskset quant-research-notools` the baselines are scored as that
taskset scores a reply: no claim and no backtest, and the reward is the hidden
rank IC over the IC ceiling (`reward.ic_reward`), so the table compares with
no-tools model runs. The datasets are the same either way.

Task i uses dataset `synthetic.dataset_seed(i, secret, split)` and, unless
`--no-vary`, the varied market of `synthetic.variation_seed`, as in the
taskset; pass the taskset's secret and split to score the same datasets as a
model run.
Every submission goes through `quantenv.scoring.score_submission` on the
hidden periods, the same path as the environment's reward. It imports no
verifiers code, so it also runs on Windows.

Usage:
    uv run python scripts/run_baselines.py --secret S [--split eval] [--tasks 30]
        [--taskset quant-research] [--workers 4] [--no-vary]
        [--out rows.csv]
"""

import argparse
import concurrent.futures
import dataclasses
import functools

import polars as pl

from quantenv import baselines, checks, metrics, report, reward, scoring
from quantenv.sources import synthetic

# The main taskset first: the default.
TASKSETS = ("quant-research", "quant-research-notools")


def codes(seed: int, market: synthetic.Config) -> dict[str, str]:
    """The baselines' alphas for one seed.

    Args:
        seed: Dataset seed.
        market: The task's generator parameters.

    Returns:
        Baseline name to alpha code.
    """
    syn = synthetic.generate(seed, market)
    truth = syn.truth
    alphas = {
        "naive": baselines.naive(syn.panel)[0],
        "template": baselines.template_code(list(syn.panel.schema.features)),
        "oracle": baselines.oracle_code(truth, syn.panel.schema),
    }
    if truth.leaky is not None:
        alphas["leaky"] = baselines.column_code(truth.leaky)
    return alphas


def score_task(
    index: int, secret: str, split: str, vary: bool, taskset: str = TASKSETS[0]
) -> list[dict]:
    """Scores every baseline on one task.

    On the main taskset each baseline claims the net Sharpe of one
    visible-period backtest, as an agent would. On the no-tools taskset it
    claims nothing and runs no backtest, as a reply does there, and earns
    that taskset's reward.

    Args:
        index: Task number.
        secret: The taskset's secret.
        split: The taskset's split.
        vary: Use the task's varied market, as the taskset does by default.
        taskset: The taskset whose scoring to follow.

    Returns:
        One row per baseline, with the task's style and bar.
    """
    seed = synthetic.dataset_seed(index, secret, split)
    variation = synthetic.variation_seed(index, split, seed) if vary else None
    market = synthetic.task_config(variation)
    visible = functools.partial(synthetic.visible_data, seed, market)
    hidden = functools.partial(synthetic.hidden_data, seed, market)
    tools = taskset == TASKSETS[0]
    max_ic = checks.CheckConfig().max_ic
    rows = []
    for name, code in codes(seed, market).items():
        claim = scoring.backtest(code, visible).sharpe if tools else 0.0
        score = scoring.score_submission(code, claim, hidden)
        raw = dataclasses.asdict(score)
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
                "style": market.style,
                "bar": "daily" if market.bar is None else str(market.bar),
                "reward": reward.reward(raw) if tools else reward.ic_reward(raw, max_ic),
                **performance,
                "claim_gap": score.claim_gap if tools else None,
                "used_leak": score.used_leak,
                "lookahead": score.lookahead,
                "implausible_ic": score.implausible_ic,
                "invalid": score.invalid,
                "error": score.error,
                "backtests_used": 1 if tools else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--secret", required=True, help="the taskset's secret")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--tasks", type=int, default=30, help="score tasks 0..N-1")
    parser.add_argument("--taskset", choices=TASKSETS, default=TASKSETS[0])
    parser.add_argument("--workers", type=int, default=4, help="tasks in parallel")
    parser.add_argument("--out", help="write the per-task rows to this CSV")
    parser.add_argument(
        "--no-vary",
        dest="vary",
        action="store_false",
        help="the default market for every task, as before Decision 26",
    )
    args = parser.parse_args()

    # Each scoring call runs in its own process, so threads are enough here.
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        per_task = []
        score = functools.partial(
            score_task,
            secret=args.secret,
            split=args.split,
            vary=args.vary,
            taskset=args.taskset,
        )
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
