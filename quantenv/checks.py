"""Lookahead guards. Generic over any panel; no knowledge of the simulator.

- truncation: does a prediction at t change when rows after t are removed?
  Catches alpha code that reads future rows.
- column_dependence: does performance fall when columns are permuted across
  assets within each timestamp? Catches use of columns known or suspected to
  hold future information.
- ic_profile: rank IC of the fixed predictions against the returns realised
  before t and those to come after it, period by period. Two flags read it,
  both backstops for lookahead the other checks miss, e.g. a leak built into
  the data, and both statistical flags, not proof. Neither judges the past
  side: an alpha may be built from past returns, so its IC there can reach 1.
  `implausible_ic` flags an IC against any future return too high for honest
  work. `late_ic_peak` flags a profile that rises after the target: an honest
  alpha knows later returns only through what persists, so its IC fades with
  distance, while a leak too far ahead peaks where it leaks. Unlike the
  ceiling, it does not depend on the IC's scale.

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
        max_ic: `implausible_ic` flags a mean rank IC against any future
            return with absolute value at or above this. Honest alphas on
            the default synthetic data reach about 0.03 at most (spec,
            Decision 24); peeking at the target gives close to 1. Set it
            per data source: real daily equity alphas rarely reach 0.1.
        profile_lags: Lags of the IC profile on each side of t, so
            `shifted_ics` covers -this to -1 and 1 to this; also the lags of
            `smooth_decay` and the delays of `memory_decay`. At least 3.
        peak_z: `late_ic_peak` flags a later lag whose mean |IC| exceeds
            the one at +1 by more than this many standard errors of the
            paired per-period difference...
        peak_ratio: ...and is at least this many times the |IC| at +1.
        decay_z: `smooth_decay` flags a lag whose IC stands above the
            exponential decay of the other lags by more than this many
            bootstrap standard errors...
        memory_z: `memory_decay` flags a delay whose IC falls below what the
            signal's own autocorrelation keeps by more than this many
            bootstrap standard errors. Stricter than `decay_z`: slow honest
            signals, such as momentum, keep more memory than edge (spec,
            Decision 29)...
        decay_share: ...and, for both, by more than this share of the IC
            expected there.
        tail_z: `smooth_decay` judges a lag only where the decay predicted
            from the other lags is this many standard errors from 0: a fast
            or useless signal has no tail to stand above.
        bootstrap_block: Consecutive periods per block of the moving-block
            bootstrap that gives the decay checks their standard errors.
        bootstrap_draws: Resamples of that bootstrap.

    Raises:
        ValueError: If `profile_lags` is below 3.
    """

    n_cuts: int = 8
    rank_noise: float = 1e-4
    rank_noise_share: float = 0.01
    dependence_z: float = 4.0
    dependence_min_drop: float = 0.05
    max_ic: float = 0.05
    profile_lags: int = 10
    peak_z: float = 4.0
    peak_ratio: float = 2.0
    decay_z: float = 4.0
    memory_z: float = 6.0
    decay_share: float = 0.5
    tail_z: float = 2.0
    bootstrap_block: int = 20
    bootstrap_draws: int = 200

    def __post_init__(self) -> None:
        # Leaving one lag out of `smooth_decay` must leave two to fit.
        if self.profile_lags < 3:
            raise ValueError(f"profile_lags must be at least 3, got {self.profile_lags}")


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


def ic_profile(panel: Panel, run: Run, lags: int) -> pl.DataFrame:
    """Per-period rank IC of the predictions against returns at each lag.

    The predictions stay fixed while the return slides one period at a time.
    Lag 1 is the target, lag k the return k - 1 periods after it; lag -1 is
    the return realised at t (`ret`), lag -k the one k - 1 periods before.
    There is no lag 0. Periods too close to an end of the panel to have a
    lag's return get IC 0 there.

    Args:
        panel: The data the predictions were made on.
        run: The walk-forward run.
        lags: Lags on each side.

    Returns:
        One row per scored period: the timestamp, then one IC column per lag,
        "-`lags`" to "-1" and "1" to "`lags`".
    """
    s = panel.schema
    # The target shifted forward by `-shift` periods is the return at the lag.
    shifts = {
        **{str(-k): k for k in range(lags, 0, -1)},
        **{str(k): 1 - k for k in range(1, lags + 1)},
    }
    frame = (
        panel.frame.lazy()
        .select(
            s.timestamp,
            *(
                pl.col(s.target).shift(n).over(s.asset).alias(f"_r{lag}")
                for lag, n in shifts.items()
            ),
        )
        .filter(pl.col(s.timestamp) >= run.start)
        .collect()
        .with_columns(run.predictions.get_column(PREDICTION))
    )
    return metrics.period_rank_ics(
        frame, {lag: (PREDICTION, f"_r{lag}") for lag in shifts}, s.timestamp
    )


def shifted_ics(profile: pl.DataFrame) -> dict[int, float]:
    """Mean rank IC at each lag of an IC profile.

    Args:
        profile: Output of `ic_profile`.

    Returns:
        Lag to mean IC, e.g. -1 for the return realised at t (`ret`), 1 for
        the target.
    """
    means = profile.drop(profile.columns[0]).mean().row(0, named=True)
    return {int(lag): ic for lag, ic in means.items()}


def bootstrap_rows(n: int, block: int, draws: int, seed: int) -> list[pl.Series]:
    """Row indices of a moving-block bootstrap over `n` consecutive periods.

    Each draw concatenates blocks of `block` consecutive rows, each starting
    at a row drawn uniformly at random, and keeps the first `n`. Blocks keep
    the serial correlation that a persistent signal gives its per-period ICs,
    which resampling single periods would destroy.

    Args:
        n: Rows to resample, one per period.
        block: Rows per block; at most `n` are used.
        draws: Number of resamples.
        seed: Seed for the block starts.

    Returns:
        `draws` Series of `n` UInt32 row indices.
    """
    block = max(1, min(block, n))
    per_draw = -(-n // block)
    starts = pl.int_range(n - block + 1, eager=True).sample(
        per_draw * draws, with_replacement=True, seed=seed % 2**64
    )
    return [
        part.get_column("row")
        for part in pl.DataFrame({"start": starts})
        .lazy()
        .select(
            (pl.int_range(pl.len()) // per_draw).alias("draw"),
            pl.int_ranges("start", pl.col("start") + block, dtype=pl.UInt32).alias("row"),
        )
        .explode("row")
        .filter(pl.int_range(pl.len()).over("draw") < n)
        .collect()
        .partition_by("draw", maintain_order=True, include_key=False)
    ]


def implausible_ic(
    shifted: dict[int, float], config: CheckConfig = CheckConfig()
) -> bool:
    """Whether the IC against any future return is too high to be honest.

    Only future lags are judged: the past ones are returns already realised
    at t, which an honest alpha may track.

    Args:
        shifted: Output of `shifted_ics`.
        config: Check settings; uses `max_ic`.

    Returns:
        True if |IC| >= `config.max_ic` at some lag of 1 or more. NaN ICs
        never flag.
    """
    return any(abs(ic) >= config.max_ic for lag, ic in shifted.items() if lag > 0)


def late_ic_peak(profile: pl.DataFrame, config: CheckConfig = CheckConfig()) -> bool:
    """Whether the IC profile peaks after the target.

    Compares the |IC| at each lag after 1 with the |IC| at lag 1, period by
    period, each signed by its own mean so a sign flip cannot hide a peak.
    Flags a lag whose excess is larger than `config.peak_z` standard errors
    of the paired per-period difference and whose |IC| is at least
    `config.peak_ratio` times the one at lag 1. Past lags are ignored.

    Args:
        profile: Output of `ic_profile`.
        config: Check settings; uses the peak thresholds.

    Returns:
        True if some later lag beats lag 1 on both counts.
    """
    first, *later = [c for c in profile.columns[1:] if int(c) > 0]
    means = profile.select(pl.col(first, *later).mean()).row(0, named=True)

    def oriented(lag: str) -> pl.Expr:
        return pl.col(lag) * (-1.0 if means[lag] < 0 else 1.0)

    excess = profile.select(
        pl.struct(
            mean=(oriented(k) - oriented(first)).mean(),
            se=(oriented(k) - oriented(first)).std() / pl.len().sqrt(),
        ).alias(k)
        for k in later
    ).row(0, named=True)
    return any(
        e["mean"] > config.peak_z * (e["se"] or 0.0)
        and abs(means[k]) >= config.peak_ratio * abs(means[first])
        for k, e in excess.items()
    )
