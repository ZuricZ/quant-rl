"""Measures the IC decay checks on honest alphas and on known exploits.

The calibration behind `CheckConfig.decay_z`, `memory_z` and `decay_share`
(spec, Decision 29). On eval tasks 0..N-1 of `--secret`, every alpha runs
walk-forward on the hidden periods, in process, and both decay checks score
it with the task's check seed. Honest set:
- naive, template and oracle (see `scripts/run_baselines.py`);
- momentum20: the 20-period past return; reversal5 and reversal1: minus the
  5-period and 1-period past return;
- blends: the oracle z-scored per period plus 0.5 or 1.0 times momentum20,
  reversal5 or reversal1, z-scored.
On the first three tasks, the exploits of `tests/test_exploits.py`, built
on the oracle-weighted signal: sparse_next, block_tail, sparse_after and
sparse_plateau. A flag on an honest alpha is a false positive.

Prints one row per task and alpha (flag, largest z and share per check,
seconds per check), then flags per alpha. It imports no verifiers code, so
it also runs on Windows.

Usage:
    uv run python scripts/measure_decay_checks.py [--secret profile-measure]
        [--tasks 10] [--out rows.csv]
"""

import argparse
import dataclasses
import time

import polars as pl

from quantenv import baselines, checks, walkforward
from quantenv.alpha import compile_alpha
from quantenv.sources import synthetic

WF = walkforward.WalkForward()
CONFIG = checks.CheckConfig()
# About 2% of periods, chosen by the timestamp alone.
GATE = "(pl.col('timestamp').hash(7) % 50 == 0)"
NEXT = "pl.col('ret').shift(-1).over('asset')"
AFTER = "pl.col('ret').shift(-2).over('asset')"
PLATEAU = "pl.col('ret').shift(-10).rolling_mean(10).over('asset')"
PAST_RETURN = {
    "momentum20": "pl.col('ret').rolling_sum(20).over('asset')",
    "reversal5": "-pl.col('ret').rolling_sum(5).over('asset')",
    "reversal1": "-pl.col('ret')",
}
EXPLOIT_TASKS = 3


def _predict(body: str) -> str:
    return f"class S(Alpha):\n    def predict(self, df):\n        return {body}\n"


def _gated(honest: str, leak: str) -> str:
    return (
        "class S(Alpha):\n"
        "    def features(self, df):\n"
        f"        return df.with_columns(x=pl.when({GATE}).then({leak}).otherwise({honest}))\n"
        "    def predict(self, df):\n"
        "        return pl.col('x')\n"
    )


def _block_tail(honest: str) -> str:
    return (
        "class S(Alpha):\n"
        "    def predict(self, df):\n"
        "        stamps = df.get_column('timestamp').unique().sort()\n"
        f"        if stamps.len() < {WF.refit_every}:\n"
        f"            return {honest}\n"
        "        tail = stamps.slice(-6, 5).to_list()\n"
        f"        return pl.when(pl.col('timestamp').is_in(tail)).then({NEXT}).otherwise({honest})\n"
    )


def _zscore(timestamps: pl.Series, prediction: pl.Series) -> pl.Series:
    """The prediction standardised within each period."""
    p = pl.col("p")
    return (
        pl.DataFrame({"t": timestamps, "p": prediction})
        .select(((p - p.mean().over("t")) / p.std().over("t")).alias("z"))
        .get_column("z")
    )


def _score(name: str, task: int, style: str, data, run: walkforward.Run) -> dict:
    """Both decay checks on one run, with their margins and timings."""
    started = time.perf_counter()
    b = checks.smooth_decay(
        checks.ic_profile(data.panel, run, CONFIG.profile_lags),
        CONFIG,
        seed=data.check_seed,
    )
    b_seconds = time.perf_counter() - started
    started = time.perf_counter()
    c = checks.memory_decay(
        checks.delay_profile(data.panel, run, CONFIG.profile_lags),
        CONFIG,
        seed=data.check_seed,
    )
    c_seconds = time.perf_counter() - started
    return {
        "task": task,
        "style": style,
        "alpha": name,
        "B": b.flagged,
        "B_z": b.z,
        "B_share": b.share,
        "C": c.flagged,
        "C_z": c.z,
        "C_share": c.share,
        "B_s": b_seconds,
        "C_s": c_seconds,
    }


def measure_task(index: int, secret: str) -> list[dict]:
    """Runs the honest set, and on the first tasks the exploits, on one task.

    Args:
        index: Eval task number.
        secret: Secret the dataset seeds are hashed from.

    Returns:
        One row per alpha.
    """
    seed = synthetic.dataset_seed(index, secret, "eval")
    market = synthetic.task_config(synthetic.variation_seed(index, "eval", seed))
    visible = synthetic.generate(seed, market)
    hidden = synthetic.hidden_data(seed, market)
    truth = visible.truth
    honest = " + ".join(f"{w!r} * pl.col({n!r})" for n, w in truth.weights.items())
    codes = {
        "naive": baselines.naive(visible.panel)[0],
        "template": baselines.template_code(list(visible.panel.schema.features)),
        "oracle": baselines.oracle_code(truth, visible.panel.schema),
        **{name: _predict(body) for name, body in PAST_RETURN.items()},
    }
    if index < EXPLOIT_TASKS:
        codes |= {
            "exploit sparse_next": _gated(honest, NEXT),
            "exploit block_tail": _block_tail(honest),
            "exploit sparse_after": _gated(honest, AFTER),
            "exploit sparse_plateau": _gated(honest, PLATEAU),
        }
    runs = {
        name: walkforward.run(
            walkforward.Local(compile_alpha(code)), hidden.panel, WF, hidden.start
        )
        for name, code in codes.items()
    }
    s = hidden.panel.schema
    oracle = runs["oracle"]
    timestamps = (
        hidden.panel.frame.lazy()
        .filter(pl.col(s.timestamp) >= oracle.start)
        .select(s.timestamp)
        .collect()
        .get_column(s.timestamp)
    )
    z_oracle = _zscore(timestamps, oracle.predictions.get_column(walkforward.PREDICTION))
    for name in PAST_RETURN:
        z_other = _zscore(timestamps, runs[name].predictions.get_column(walkforward.PREDICTION))
        for weight in (0.5, 1.0):
            blend = (z_oracle + weight * z_other).alias(walkforward.PREDICTION)
            runs[f"oracle+{weight}*{name}"] = dataclasses.replace(
                oracle, predictions=oracle.predictions.with_columns(blend)
            )
    return [_score(name, index, market.style, hidden, run) for name, run in runs.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--secret", default="profile-measure")
    parser.add_argument("--tasks", type=int, default=10, help="eval tasks 0..N-1")
    parser.add_argument("--out", help="write the per-alpha rows to this CSV")
    args = parser.parse_args()
    rows = []
    for index in range(args.tasks):
        rows.extend(measure_task(index, args.secret))
        print(f"measured task {index}", flush=True)
    frame = pl.DataFrame(rows)
    if args.out:
        frame.write_csv(args.out)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(frame)
        print(
            frame.group_by("alpha", maintain_order=True).agg(
                pl.len().alias("runs"),
                pl.col("B").sum(),
                pl.col("C").sum(),
                pl.col("B_z").max(),
                pl.col("C_z").max(),
                pl.col("B_s").max(),
                pl.col("C_s").max(),
            )
        )


if __name__ == "__main__":
    main()
