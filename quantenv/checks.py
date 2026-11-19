"""Lookahead guards. Generic over any panel; no knowledge of the simulator.

- truncation: does a prediction at t change when rows after t are removed?
  Catches alpha code that reads future rows.
- column_dependence: does performance fall when columns are permuted across
  assets within each timestamp? Catches use of columns known or suspected to
  hold future information.
- shifted_ics: rank IC against past, next and next-but-one returns.
  `implausible_ic` flags an IC against the next return too high for honest
  work: a backstop for lookahead the other checks miss, e.g. a leak built
  into the data. A statistical flag, not proof.

Truncation compares per-period ranks, not raw values, and forgives rank swaps
of near-ties too small to be anything but floating-point noise. Column
dependence compares rank IC, because predictions may change without the
column being used: a data-driven screen that drops the suspect column as
implausibly strong may admit it once it is permuted into noise. Their random
choices (cut dates, permutation) come from a seed that agent code must not
know: an alpha that knew the cut dates could be honest on exactly those
dates. Their thresholds are a `CheckConfig`.
"""

import dataclasses
from typing import Any

import polars as pl

from quantenv import metrics, walkforward
from quantenv.panel import Panel
from quantenv.walkforward import PREDICTION, Run, Runner, WalkForward

# Scratch columns, dropped before results are returned.
_KEY = "_shuffle_key"
_STRATUM = "_stratum"


@dataclasses.dataclass(frozen=True)
class CheckConfig:
    """Settings of the lookahead checks. See the spec, Decisions 7 and 8,
    for the measurements behind the defaults.

    Attributes:
        n_cuts: Truncation cuts; each costs one pipeline run.
        rank_noise: Truncation forgives a changed rank as floating-point
            noise when the rank moved by at most 1 and the value by at most
            this times (|a| + |b| + the period's median absolute deviation);
            see `same_period_ranks`. 0 for exact ranks.
        rank_noise_share: ...but only while forgiven rows are at most this
            share of the rows compared in a cut.
        dependence_z: Column dependence flags a drop in mean rank IC larger
            than this many standard errors of the paired per-period
            difference...
        dependence_min_drop: ...and larger than this share of the base
            run's |IC|.
        max_ic: `implausible_ic` flags a mean rank IC against the next
            return with absolute value at or above this. Honest alphas on
            the default synthetic data reach about 0.03 at most (spec,
            Decision 24); peeking at the target gives close to 1. Set it
            per data source: real daily equity alphas rarely reach 0.1.
    """

    n_cuts: int = 8
    rank_noise: float = 1e-4
    rank_noise_share: float = 0.01
    dependence_z: float = 4.0
    dependence_min_drop: float = 0.05
    max_ic: float = 0.05


def cut_points(timestamps: pl.Series, n: int, seed: int) -> list[Any]:
    """Draws `n` cut dates, one at random from each of `n` equal strata.

    Strata spread the cuts over the whole period; drawing within each keeps
    them unpredictable without the seed.

    Args:
        timestamps: Candidate timestamps, ascending.
        n: Number of cuts; at most `len(timestamps)`.
        seed: Seed for the draw.

    Returns:
        The cut timestamps, ascending.
    """
    ts = timestamps.name
    return (
        timestamps.to_frame()
        .lazy()
        .with_columns((pl.int_range(pl.len()) * n // pl.len()).alias(_STRATUM))
        .group_by(_STRATUM, maintain_order=True)
        .agg(pl.col(ts).shuffle(seed).first())
        .collect()
        .get_column(ts)
        .to_list()
    )


def same_period_ranks(
    timestamps: pl.Series,
    a: pl.Series,
    b: pl.Series,
    noise: float = 0.0,
    max_share: float = 0.0,
) -> bool:
    """Whether two outputs rank the assets identically in every period, up to
    floating-point noise.

    A row agrees if its rank within its period is the same in both. A row
    whose rank changed is forgiven as noise if the rank moved by at most 1
    and the value by at most `noise` times (|a| + |b| + the period's median
    absolute deviation of `a`), as long as forgiven rows are at most
    `max_share` of all rows. Reruns of the same alpha on the same rows can
    differ in the last bits, e.g. when polars sums in parallel, and swap
    near-ties; a real difference moves ranks further or values more.

    The scale ignores the period's spread, which agent code controls: one
    extreme asset would inflate a standard deviation and hide every other
    row's change. The median absolute deviation resists such outliers and
    cannot overflow, and the share cap stops forgiveness at scale.

    Args:
        timestamps: Timestamp of each row.
        a: First output, one value per row.
        b: Second output, same rows.
        noise: Relative value change still treated as noise; 0 for exact.
        max_share: Largest share of rows that may be forgiven.

    Returns:
        True if every row agrees or is forgiven, nulls included.
    """
    ts = timestamps.name
    va, vb = pl.col("a"), pl.col("b")
    ra, rb = va.rank().over(ts), vb.rank().over(ts)
    same = ra.eq_missing(rb)
    mad = (va - va.median()).abs().median().over(ts)
    scale = va.abs() + vb.abs() + mad
    forgiven = (
        ~same & ((ra - rb).abs() <= 1) & ((va - vb).abs() <= noise * scale)
    ).fill_null(False)
    counts = (
        pl.DataFrame([timestamps, a.alias("a"), b.alias("b")])
        .select(
            changed=(~same & ~forgiven).sum(),
            forgiven=forgiven.sum(),
            rows=pl.len(),
        )
        .row(0, named=True)
    )
    return counts["changed"] == 0 and counts["forgiven"] <= max_share * counts["rows"]


def truncation(
    runner: Runner,
    panel: Panel,
    wf: WalkForward,
    start: Any = None,
    base: Run | None = None,
    *,
    seed: int,
    config: CheckConfig = CheckConfig(),
) -> bool:
    """Reruns on the panel cut after a few secret timestamps and compares.

    Each cut is one session. `features` reruns on the truncated panel. If its
    output up to the cut matches the full run, every block before the cut is
    identical by construction, so only the block containing the cut is refit
    and predicted. Otherwise the whole truncated pipeline reruns, since a
    feature that changed may still not affect the predictions.

    Args:
        runner: Runs the alpha.
        panel: The data.
        wf: Walk-forward settings.
        start: First predicted timestamp, as for `walkforward.run`.
        base: The full run, if already computed.
        seed: Secret seed for the cut dates (see `cut_points`).
        config: Check settings; uses `n_cuts`.

    Returns:
        True if any prediction at or before a cut changed: lookahead.

    Raises:
        AlphaError: If any run fails.
    """
    s = panel.schema
    if base is None:
        base = walkforward.run(runner, panel, wf, start)
    predicted = panel.timestamps.filter(panel.timestamps >= base.start)
    cuts = cut_points(predicted, min(config.n_cuts, len(predicted)), seed)
    for cut in cuts:
        short = panel.until(cut)
        until_cut = base.engineered.lazy().filter(pl.col(s.timestamp) <= cut)
        todo = walkforward.blocks(short, wf, base.start)
        with runner.session(s.timestamp) as session:
            engineered = walkforward.engineer(session, short)
            if engineered.equals(until_cut.collect()):
                # Features agree up to the cut: only the cut's block can differ.
                todo = todo[-1:]
            out = walkforward.predict_blocks(session, short, todo)
        expected = (
            base.predictions.lazy()
            .filter(pl.col(s.timestamp).is_between(todo[0].start, cut))
            .collect()
        )
        stamps = expected.get_column(s.timestamp)
        expected_out = expected.get_column(PREDICTION)
        if not same_period_ranks(
            stamps, expected_out, out, config.rank_noise, config.rank_noise_share
        ):
            return True
    return False


def shuffle_within_periods(
    frame: pl.DataFrame, columns: list[str], timestamp: str, seed: int
) -> pl.DataFrame:
    """Permutes columns across assets within each timestamp.

    Sorting each period by a globally shuffled key gives every period its own
    permutation. Every per-period statistic of a column is kept; only which
    asset has which value changes. Other columns are shared, not copied.

    Args:
        frame: Long frame sorted by timestamp.
        columns: Columns to permute, all with the same permutation.
        timestamp: Time column.
        seed: Seed for the permutation.

    Returns:
        The frame with `columns` permuted.
    """
    return (
        frame.lazy()
        .with_columns(pl.int_range(pl.len()).shuffle(seed).alias(_KEY))
        .with_columns(pl.col(columns).sort_by(_KEY).over(timestamp))
        .drop(_KEY)
        .collect()
    )


def column_dependence(
    runner: Runner,
    panel: Panel,
    columns: list[str],
    wf: WalkForward,
    start: Any = None,
    base: Run | None = None,
    *,
    seed: int,
    config: CheckConfig = CheckConfig(),
) -> bool:
    """Reruns with `columns` permuted within each timestamp and compares IC.

    Permuting also changes what `fit` trains on, so a model that learned to
    use a column is caught as well. Flags when the permuted run's mean rank
    IC is lower by more than `config.dependence_z` standard errors of the
    paired per-period difference and by more than
    `config.dependence_min_drop` of the base run's |IC|.

    Args:
        runner: Runs the alpha.
        panel: The data.
        columns: Columns to test, e.g. those suspected of holding future
            information.
        wf: Walk-forward settings.
        start: First predicted timestamp, as for `walkforward.run`.
        base: The unpermuted run, if already computed.
        seed: Secret seed for the permutation.
        config: Check settings; uses the dependence thresholds.

    Returns:
        True if performance depends on any of `columns`. False if `columns`
        is empty.

    Raises:
        AlphaError: If any run fails.
    """
    if not columns:
        return False
    if base is None:
        base = walkforward.run(runner, panel, wf, start)
    shuffled = panel.with_frame(
        shuffle_within_periods(panel.frame, columns, panel.schema.timestamp, seed)
    )
    out = walkforward.run(runner, shuffled, wf, base.start).predictions
    s = panel.schema
    ics = metrics.period_rank_ics(
        panel.frame.lazy()
        .filter(pl.col(s.timestamp) >= base.start)
        .select(s.timestamp, s.target)
        .collect()
        .with_columns(
            base.predictions.get_column(PREDICTION).alias("_base"),
            out.get_column(PREDICTION).alias("_permuted"),
        ),
        {"base": ("_base", s.target), "permuted": ("_permuted", s.target)},
        s.timestamp,
    )
    diff = pl.col("base") - pl.col("permuted")
    drop, se, base_ic = ics.select(
        diff.mean().alias("drop"),
        (diff.std() / pl.len().sqrt()).alias("se"),
        pl.col("base").mean().alias("base_ic"),
    ).row(0)
    material = drop > config.dependence_min_drop * abs(base_ic)
    return drop > config.dependence_z * (se or 0.0) and material


def shifted_ics(panel: Panel, run: Run) -> dict[int, float]:
    """Rank IC of the predictions against returns at several shifts.

    Args:
        panel: The data the predictions were made on.
        run: The walk-forward run.

    Returns:
        Shift to IC: -1 is the return realised at t (`ret`), 1 the target,
        2 the return one period after the target.
    """
    s = panel.schema
    frame = (
        panel.frame.lazy()
        .select(
            s.timestamp,
            s.ret,
            s.target,
            pl.col(s.target).shift(-1).over(s.asset).alias("_next"),
        )
        .filter(pl.col(s.timestamp) >= run.start)
        .collect()
        .with_columns(run.predictions.get_column(PREDICTION))
    )
    ics = metrics.mean_rank_ics(
        frame,
        {
            "-1": (PREDICTION, s.ret),
            "1": (PREDICTION, s.target),
            "2": (PREDICTION, "_next"),
        },
        s.timestamp,
    )
    return {int(k): v for k, v in ics.items()}


def implausible_ic(
    shifted: dict[int, float], config: CheckConfig = CheckConfig()
) -> bool:
    """Whether the IC against the next return is too high to be honest.

    Only shift 1 is judged. Shift -1 is the return already realised at t,
    which an honest alpha may track. Shift 2 says nothing on its own:
    persistent honest signals predict it too.

    Args:
        shifted: Output of `shifted_ics`.
        config: Check settings; uses `max_ic`.

    Returns:
        True if |IC at shift 1| >= `config.max_ic`. False for a NaN IC.
    """
    return abs(shifted[1]) >= config.max_ic
