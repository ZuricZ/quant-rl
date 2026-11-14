"""Simulated asset panels with planted, noise and leaky features.

A seed fixes everything: which feature names are real, noise or leaky, the
planted ICs, and the data itself. The visible and hidden periods come from
independent random streams, so the visible panel can be built without ever
drawing the hidden one.

numpy is used only to draw random numbers, which polars cannot do. All
draws for a part come from one call; everything else is polars expressions.
"""

import dataclasses
import datetime as dt
import hashlib
import math
from typing import Any

import numpy as np
import polars as pl

from quantenv.panel import EvalData, Panel, Schema

_START = dt.datetime(2000, 1, 3)
_RETURN_VOL = 0.02
_RETURN_NOISE = "_return_noise"
_LEAK_NOISE = "_leak_noise"


@dataclasses.dataclass(frozen=True)
class Config:
    """Generator parameters. The defaults follow the build spec.

    Attributes:
        n_assets: Assets per period.
        n_visible: Periods the agent can backtest on.
        n_hidden: Periods used only for scoring, directly after the visible.
        bar: Length of an intraday period. None gives daily periods on
            business days with a Date timestamp; a timedelta gives
            back-to-back periods of that length with a Datetime timestamp.
        n_features: Total feature columns, named f00, f01, ...
        n_real: Real features on a seed that has a leaky feature.
        real_ic_range: Range the planted IC of each real feature is drawn from.
        leaky_ic: Rank IC of the leaky feature.
        no_leak_prob: Share of seeds whose leaky slot becomes another real
            feature.
        replacement_ic: Planted IC of that extra real feature.
    """

    n_assets: int = 200
    n_visible: int = 1_000
    n_hidden: int = 10_000
    bar: dt.timedelta | None = None
    n_features: int = 20
    n_real: int = 3
    real_ic_range: tuple[float, float] = (0.02, 0.04)
    leaky_ic: float = 0.3
    no_leak_prob: float = 0.2
    replacement_ic: float = 0.06

    @property
    def feature_names(self) -> list[str]:
        return [f"f{i:02d}" for i in range(self.n_features)]

    @property
    def schema(self) -> Schema:
        return Schema(features=tuple(self.feature_names))


DEFAULT_CONFIG = Config()


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    """The role of every feature for one seed. Only the simulator has this.

    Attributes:
        real: Real feature names, sorted.
        noise: Noise feature names, sorted.
        leaky: The leaky feature name, or None on a seed without one.
        planted_ic: Target rank IC of each real feature.
        weights: Weight of each real feature in the standardized next-period
            return. Each equals that feature's Pearson correlation with it.
    """

    real: tuple[str, ...]
    noise: tuple[str, ...]
    leaky: str | None
    planted_ic: dict[str, float]
    weights: dict[str, float]


@dataclasses.dataclass(frozen=True)
class Synthetic:
    """A simulated panel with its ground truth.

    Attributes:
        seed: The seed it was built from.
        panel: The data. Visible periods, then hidden ones if requested.
        truth: Ground truth for the seed.
        hidden_start: First hidden timestamp, or None if the panel holds only
            the visible periods.
    """

    seed: int
    panel: Panel
    truth: GroundTruth
    hidden_start: Any = None


def dataset_seed(index: int, secret: str) -> int:
    """The seed of dataset `index` under `secret`.

    Code that can run the generator can rebuild any dataset whose seed it can
    guess, hidden periods and ground truth included. Seeds drawn from a
    secret it does not know cannot be guessed.

    Args:
        index: Dataset number, e.g. a task's position in the taskset.
        secret: Known only to whoever builds and scores the datasets.

    Returns:
        A 128-bit seed.
    """
    digest = hashlib.sha256(f"{secret}\0{index}".encode()).digest()
    return int.from_bytes(digest[:16], "big")


def spearman_to_pearson(rho_s: float) -> float:
    """Pearson correlation of a bivariate normal with a given Spearman one."""
    return 2.0 * math.sin(math.pi * rho_s / 6.0)


def ground_truth(seed: int, config: Config = DEFAULT_CONFIG) -> GroundTruth:
    """Draws the feature roles and planted ICs for a seed.

    Args:
        seed: Non-negative integer seed.
        config: Generator parameters.

    Returns:
        The seed's ground truth.
    """
    rng = np.random.default_rng(_streams(seed)[0])
    shuffled = rng.permutation(config.feature_names).tolist()
    has_leak = rng.random() >= config.no_leak_prob
    ics = rng.uniform(*config.real_ic_range, size=config.n_real).tolist()

    k = config.n_real
    real, slot, noise = shuffled[:k], shuffled[k], shuffled[k + 1 :]
    if not has_leak:
        real.append(slot)
        ics.append(config.replacement_ic)

    planted = dict(sorted(zip(real, ics)))
    return GroundTruth(
        real=tuple(planted),
        noise=tuple(sorted(noise)),
        leaky=slot if has_leak else None,
        planted_ic=planted,
        weights={k: spearman_to_pearson(v) for k, v in planted.items()},
    )


def generate(
    seed: int, config: Config = DEFAULT_CONFIG, hidden: bool = False
) -> Synthetic:
    """Builds the panel for a seed.

    Features are independent standard normals. The standardized next-period
    return is a weighted sum of the real features plus noise, scaled to a
    volatility of 2% per period. The leaky feature is the standardized return
    plus enough noise to give it the configured rank IC. `ret` at t is the
    same asset's target at the previous timestamp.

    Args:
        seed: Non-negative integer seed.
        config: Generator parameters.
        hidden: Whether to append the hidden periods after the visible ones.

    Returns:
        The panel and its ground truth.
    """
    truth = ground_truth(seed, config)
    streams = _streams(seed)
    parts = [_part(streams[1], config, truth, 0, config.n_visible)]
    if hidden:
        parts.append(
            _part(streams[2], config, truth, config.n_visible, config.n_hidden)
        )
    schema = config.schema
    frame = (
        pl.concat(parts, rechunk=False)
        .lazy()
        .with_columns(
            pl.col(schema.timestamp).set_sorted(),
            # Rows are timestamp-major with every asset in every period, so
            # the previous period's row is exactly n_assets rows back.
            pl.col(schema.target).shift(config.n_assets).alias(schema.ret),
        )
        .select(
            schema.timestamp, schema.asset, *schema.features, schema.ret, schema.target
        )
        .collect()
    )
    panel = Panel(frame, schema)
    hidden_start = panel.timestamps[config.n_visible] if hidden else None
    return Synthetic(seed, panel, truth, hidden_start)


def _part(
    stream: np.random.SeedSequence,
    config: Config,
    truth: GroundTruth,
    offset: int,
    n_periods: int,
) -> pl.DataFrame:
    """One contiguous run of periods: keys, features and target."""
    schema = config.schema
    # Every draw in one call, one contiguous row per column.
    draws = np.random.default_rng(stream).standard_normal(
        (config.n_features + 2, n_periods * config.n_assets), dtype=np.float32
    )
    columns = [*schema.features, _RETURN_NOISE, _LEAK_NOISE]

    weights = truth.weights
    z = pl.sum_horizontal(
        math.sqrt(1.0 - sum(w * w for w in weights.values())) * pl.col(_RETURN_NOISE),
        *(w * pl.col(k) for k, w in weights.items()),
    )
    if truth.leaky is None:
        leak = []
    else:
        r = spearman_to_pearson(config.leaky_ic)
        noise = math.sqrt(1.0 / r**2 - 1.0) * pl.col(_LEAK_NOISE)
        leak = [(r * (z + noise)).alias(truth.leaky)]

    return (
        _keys(config, offset, n_periods)
        .hstack(pl.from_numpy(draws, schema=columns, orient="col"))
        .lazy()
        .with_columns(*leak, (_RETURN_VOL * z).alias(schema.target))
        .drop(_RETURN_NOISE, _LEAK_NOISE)
        .collect()
    )


def _keys(config: Config, offset: int, n_periods: int) -> pl.DataFrame:
    """The (timestamp, asset) columns, timestamp-major and sorted."""
    schema = config.schema
    period = pl.int_range(offset, offset + n_periods)
    if config.bar is None:
        stamp = pl.lit(_START.date()).dt.add_business_days(period)
    else:
        stamp = pl.lit(_START) + period * pl.lit(config.bar)
    asset_names = [f"A{i:03d}" for i in range(config.n_assets)]
    assets = pl.DataFrame(
        {schema.asset: pl.Series(asset_names, dtype=pl.Enum(asset_names))}
    )
    return pl.select(stamp.alias(schema.timestamp)).join(
        assets, how="cross", maintain_order="left_right"
    )


def _streams(seed: int) -> list[np.random.SeedSequence]:
    """Independent streams for metadata, visible periods and hidden periods."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.SeedSequence(seed).spawn(3)


def visible_data(seed: int, config: Config = DEFAULT_CONFIG) -> EvalData:
    """Backtest data: the visible periods only.

    Args:
        seed: Dataset seed.
        config: Generator parameters.

    Returns:
        The visible panel, scored after the walk-forward warm-up.
    """
    return EvalData(generate(seed, config).panel)


def hidden_data(seed: int, config: Config = DEFAULT_CONFIG) -> EvalData:
    """Scoring data: visible then hidden periods, scored on the hidden ones.

    The leaky column, if any, is the suspect for the column dependence check.

    Args:
        seed: Dataset seed.
        config: Generator parameters.

    Returns:
        The full panel, the first hidden timestamp and the suspect columns.
    """
    syn = generate(seed, config, hidden=True)
    suspects = () if syn.truth.leaky is None else (syn.truth.leaky,)
    return EvalData(syn.panel, syn.hidden_start, suspects)
