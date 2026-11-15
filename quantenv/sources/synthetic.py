"""Simulated asset panels with planted, noise and leaky features.

A seed fixes everything: which feature names are real, noise or leaky, the
planted ICs, the structure of the market, and the data itself. The visible
and hidden periods come from independent random streams, and every recursion
over time is causal, so the visible panel is an exact prefix of the full one
and can be built without ever drawing the hidden periods.

The market, with each mechanism controlled by `Config`:

- Features: Gaussian, correlated through a few common feature factors, and
  persistent over time (AR(1) per asset, the coefficient drawn per feature).
- Signal: a weighted sum of the real features plus one interaction of two of
  them. Its strength switches between an "on" and a weaker "off" regime, a
  Markov chain over periods.
- Returns: the signal plus fat-tailed (Student t) idiosyncratic noise, scaled
  per asset, plus common return factors with per-asset loadings, all scaled
  by a market volatility that moves over time (stochastic log-volatility).
- Leak: the cross-sectionally standardised next-period return plus noise.

numpy is used only to draw random numbers, which polars cannot do; every
transformation is a polars expression.
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
_IDIO = "_idio"
_LEAK_NOISE = "_leak_noise"
_VOL_SHOCK = "_vol_shock"
_SWITCH = "_switch"
_VOL = "_vol"
_STRENGTH = "_strength"
_FIRST = "_first"


@dataclasses.dataclass(frozen=True)
class Config:
    """Generator parameters. The defaults give the realistic market; see
    `PLAIN` for one with every mechanism off.

    Attributes:
        n_assets: Assets per period.
        n_visible: Periods the agent can backtest on.
        n_hidden: Periods used only for scoring, directly after the visible.
        bar: Length of an intraday period. None gives daily periods on
            business days with a Date timestamp; a timedelta gives
            back-to-back periods of that length with a Datetime timestamp.
        n_features: Total feature columns, named f00, f01, ...
        n_real: Real features on a seed that has a leaky feature.
        real_ic_range: Range each real feature's planted IC is drawn from:
            its Spearman correlation with the standardised return component
            in the "on" regime, before factor and volatility effects.
        leaky_ic: Rank IC of the leaky feature.
        no_leak_prob: Share of seeds whose leaky slot becomes another real
            feature.
        replacement_ic: Planted IC of that extra real feature.
        interaction_ic: Planted IC of the product of two real features;
            0 for none.
        n_feature_factors: Common factors behind the features.
        feature_factor_share: Range each feature's variance share from the
            feature factors is drawn from.
        persistence_range: Range each feature's AR(1) coefficient is drawn
            from; the feature factors draw one too. (0, 0) for none.
        n_return_factors: Common factors in returns.
        return_factor_share: Share of return variance from those factors.
        tail_df: Degrees of freedom of the Student t return noise; None for
            Gaussian.
        idio_vol_dispersion: Log-normal spread of per-asset volatility.
        vol_persistence: AR(1) coefficient of market log-volatility.
        vol_of_vol: Standard deviation of market log-volatility; 0 for
            constant volatility.
        regime_switch_prob: Chance per period that the signal regime flips.
        regime_off_strength: Signal strength in the "off" regime, relative to
            "on"; 1 for no regimes.
        cost: Trading cost per unit of turnover, in return units.
    """

    n_assets: int = 200
    n_visible: int = 1_000
    n_hidden: int = 10_000
    bar: dt.timedelta | None = None
    n_features: int = 20
    n_real: int = 3
    real_ic_range: tuple[float, float] = (0.01, 0.018)
    leaky_ic: float = 0.3
    no_leak_prob: float = 0.2
    replacement_ic: float = 0.018
    interaction_ic: float = 0.012
    n_feature_factors: int = 3
    feature_factor_share: tuple[float, float] = (0.1, 0.6)
    persistence_range: tuple[float, float] = (0.8, 0.98)
    n_return_factors: int = 3
    return_factor_share: float = 0.3
    tail_df: float | None = 4.0
    idio_vol_dispersion: float = 0.4
    vol_persistence: float = 0.98
    vol_of_vol: float = 0.3
    regime_switch_prob: float = 0.004
    regime_off_strength: float = 0.25
    cost: float = 0.0002

    @property
    def feature_names(self) -> list[str]:
        return [f"f{i:02d}" for i in range(self.n_features)]

    @property
    def periods_per_year(self) -> float:
        """252 trading days, or as many back-to-back intraday bars in them."""
        if self.bar is None:
            return 252.0
        return 252.0 * (dt.timedelta(days=1) / self.bar)

    @property
    def schema(self) -> Schema:
        return Schema(features=tuple(self.feature_names))


DEFAULT_CONFIG = Config()

# Every mechanism off: independent Gaussian features and returns, constant
# signal and volatility, no costs, and the original planted ICs. For tests of
# the machinery that need exact planted ICs.
PLAIN = Config(
    real_ic_range=(0.02, 0.04),
    replacement_ic=0.06,
    interaction_ic=0.0,
    n_feature_factors=0,
    feature_factor_share=(0.0, 0.0),
    persistence_range=(0.0, 0.0),
    n_return_factors=0,
    return_factor_share=0.0,
    tail_df=None,
    idio_vol_dispersion=0.0,
    vol_of_vol=0.0,
    regime_off_strength=1.0,
    cost=0.0,
)


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    """The role of every feature for one seed. Only the simulator has this.

    Attributes:
        real: Real feature names, sorted.
        noise: Noise feature names, sorted.
        leaky: The leaky feature name, or None on a seed without one.
        planted_ic: Planted IC of each real feature (see
            `Config.real_ic_range`).
        weights: Weight of each real feature in the standardised signal.
        interaction: The two real features whose product is in the signal,
            or None.
        interaction_weight: Its weight in the signal.
    """

    real: tuple[str, ...]
    noise: tuple[str, ...]
    leaky: str | None
    planted_ic: dict[str, float]
    weights: dict[str, float]
    interaction: tuple[str, str] | None = None
    interaction_weight: float = 0.0


@dataclasses.dataclass(frozen=True)
class _World:
    """Everything drawn once per seed: the ground truth and market structure.

    Attributes:
        truth: Feature roles and signal weights.
        persistence: AR(1) coefficient of each feature's own component.
        factor_persistence: AR(1) coefficient of the feature factors.
        directions: Each feature's loadings on the feature factors, scaled
            to its factor variance share.
        betas: Per asset, its loadings on the return factors.
        idio_vol: Per asset, its volatility multiplier.
        first_regime: 0 if the signal starts "on", 1 if "off".
        signal_var: Variance of the signal, for scaling the noise.
        interaction_mean: Mean of the interaction product, removed from it.
    """

    truth: GroundTruth
    persistence: dict[str, float]
    factor_persistence: float
    directions: dict[str, tuple[float, ...]]
    betas: tuple[tuple[float, ...], ...]
    idio_vol: tuple[float, ...]
    first_regime: int
    signal_var: float
    interaction_mean: float


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


def dataset_seed(index: int, secret: str, split: str) -> int:
    """The seed of dataset `index` of `split` under `secret`.

    Code that can run the generator can rebuild any dataset whose seed it can
    guess, hidden periods and ground truth included. Seeds drawn from a
    secret it does not know cannot be guessed. The split is part of the
    hash, so train and eval datasets are independent.

    Args:
        index: Dataset number within the split.
        secret: Known only to whoever builds and scores the datasets.
        split: E.g. "train" or "eval".

    Returns:
        A 128-bit seed.
    """
    return _hash_int(16, secret, split, index)


def _hash_int(n_bytes: int, *parts: object) -> int:
    """An integer of `n_bytes` bytes hashed from `parts`, NUL-separated."""
    text = "\0".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:n_bytes], "big")


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
    return _world(seed, config).truth


def _world(seed: int, config: Config) -> _World:
    """Draws the roles, signal and market structure for a seed."""
    rng = np.random.default_rng(_streams(seed)[0])
    names = config.feature_names
    shuffled = rng.permutation(names).tolist()
    has_leak = rng.random() >= config.no_leak_prob
    ics = rng.uniform(*config.real_ic_range, size=config.n_real).tolist()
    k = config.n_real
    real, slot, noise = shuffled[:k], shuffled[k], shuffled[k + 1 :]
    if not has_leak:
        real.append(slot)
        ics.append(config.replacement_ic)
    pair = rng.choice(sorted(real), size=2, replace=False).tolist()

    shares = rng.uniform(*config.feature_factor_share, size=len(names)).tolist()
    phis = rng.uniform(*config.persistence_range, size=len(names) + 1).tolist()
    raw = rng.standard_normal((len(names), config.n_feature_factors)).tolist()
    market = rng.normal(1.0, 0.3, size=config.n_assets).tolist()
    others = rng.standard_normal(
        (config.n_assets, max(config.n_return_factors - 1, 0))
    ).tolist()
    log_vol = rng.standard_normal(config.n_assets).tolist()
    first_regime = int(rng.random() < 0.5)

    directions = {}
    for name, share, row in zip(names, shares, raw):
        norm = math.sqrt(sum(v * v for v in row)) or 1.0
        directions[name] = tuple(math.sqrt(share) * v / norm for v in row)

    def corr(a: str, b: str) -> float:
        if a == b:
            return 1.0
        return sum(x * y for x, y in zip(directions[a], directions[b]))

    planted = dict(sorted(zip(real, ics)))
    weights = {name: spearman_to_pearson(ic) for name, ic in planted.items()}
    interaction = tuple(pair) if config.interaction_ic > 0 else None
    w_int = spearman_to_pearson(config.interaction_ic)
    rho = corr(*pair)
    # Gaussian features: the product of two is uncorrelated with each, with
    # mean rho and variance 1 + rho^2.
    signal_var = sum(weights[a] * weights[b] * corr(a, b) for a in real for b in real)
    if interaction is not None:
        signal_var += w_int**2 * (1.0 + rho**2)
    if config.n_return_factors > 0:
        betas = tuple((m, *o) for m, o in zip(market, others))
    else:
        betas = tuple(() for _ in range(config.n_assets))
    spread = config.idio_vol_dispersion
    return _World(
        truth=GroundTruth(
            real=tuple(planted),
            noise=tuple(sorted(noise)),
            leaky=slot if has_leak else None,
            planted_ic=planted,
            weights=weights,
            interaction=interaction,
            interaction_weight=w_int if interaction is not None else 0.0,
        ),
        persistence=dict(zip(names, phis)),
        factor_persistence=phis[-1],
        directions=directions,
        betas=betas,
        # Mean square 1, so the spread leaves the average variance unchanged.
        idio_vol=tuple(math.exp(spread * v - spread**2) for v in log_vol),
        first_regime=first_regime,
        signal_var=signal_var,
        interaction_mean=rho if interaction is not None else 0.0,
    )


def generate(
    seed: int, config: Config = DEFAULT_CONFIG, hidden: bool = False
) -> Synthetic:
    """Builds the panel for a seed.

    The standardised return of asset i from t to t + 1 is

        z = s_t * signal + sqrt(1 - s_t^2 * var(signal)) * noise,

    where s_t is the regime strength and the signal is the weighted real
    features plus the centred interaction. The target is then

        fwd_return = 2% * vol_t * (sqrt(1 - q) * sigma_i * z
                                   + sqrt(q / K) * beta_i . F_t),

    with market volatility vol_t, per-asset volatility sigma_i, return factor
    share q and K return factors F_t. `ret` at t is the same asset's target
    at the previous timestamp.

    Args:
        seed: Non-negative integer seed.
        config: Generator parameters.
        hidden: Whether to append the hidden periods after the visible ones.

    Returns:
        The panel and its ground truth.
    """
    world = _world(seed, config)
    streams = _streams(seed)
    parts = [_draws(streams[1], config, 0, config.n_visible)]
    if hidden:
        parts.append(_draws(streams[2], config, config.n_visible, config.n_hidden))
    rows = pl.concat([p[0] for p in parts], rechunk=False)
    periods = pl.concat([p[1] for p in parts], rechunk=False)

    schema = config.schema
    ts, asset = schema.timestamp, schema.asset
    frame = (
        rows.lazy()
        .join(_period_states(periods, config, world), on=ts, maintain_order="left")
        .with_columns((pl.col(ts) == pl.col(ts).first()).alias(_FIRST))
        .with_columns(_features(config, world, asset))
        .with_columns(_target(config, world).alias(schema.target))
        .with_columns(_leak(config, world, ts))
        .with_columns(
            pl.col(ts).set_sorted(),
            # Rows are timestamp-major with every asset in every period, so
            # the previous period's row is exactly n_assets rows back.
            pl.col(schema.target).shift(config.n_assets).alias(schema.ret),
        )
        .select(ts, asset, *schema.features, schema.ret, schema.target)
        .collect()
    )
    panel = Panel(frame, schema)
    hidden_start = panel.timestamps[config.n_visible] if hidden else None
    return Synthetic(seed, panel, world.truth, hidden_start)


def _draws(
    stream: np.random.SeedSequence, config: Config, offset: int, n_periods: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The random draws of one contiguous run of periods.

    Returns:
        Per-row draws with the (timestamp, asset) keys, and per-period draws
        with the timestamp.
    """
    rng = np.random.default_rng(stream)
    n_rows = n_periods * config.n_assets
    normal_names = [
        *_factor_names(config.n_feature_factors, "_g"),
        *(f"_e{name}" for name in config.feature_names),
        _LEAK_NOISE,
    ]
    normals = rng.standard_normal((len(normal_names), n_rows), dtype=np.float32)
    idio = _unit_noise(rng, config.tail_df, (1, n_rows))
    factor_names = _factor_names(config.n_return_factors, "_F")
    factors = _unit_noise(rng, config.tail_df, (len(factor_names), n_periods))
    vol_shocks = rng.standard_normal((1, n_periods))
    switches = rng.random((1, n_periods))

    keys = _keys(config, offset, n_periods)
    rows = keys.hstack(_columns(normal_names, normals)).hstack(_columns([_IDIO], idio))
    periods = (
        keys.select(config.schema.timestamp)
        .unique(maintain_order=True)
        .hstack(_columns(factor_names, factors))
        .hstack(_columns([_VOL_SHOCK], vol_shocks))
        .hstack(_columns([_SWITCH], switches))
    )
    return rows, periods


def _columns(names: list[str], draws: np.ndarray) -> list[pl.Series]:
    """One Series per row of `draws`, named by `names`."""
    return [pl.Series(name, row) for name, row in zip(names, draws)]


def _unit_noise(
    rng: np.random.Generator, df: float | None, shape: tuple[int, int]
) -> np.ndarray:
    """Unit-variance noise: Student t with `df` degrees of freedom, or normal."""
    if df is None:
        return rng.standard_normal(shape)
    return rng.standard_t(df, shape) / math.sqrt(df / (df - 2.0))


def _factor_names(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{k}" for k in range(n)]


def _ar1(value: pl.Expr, phi: float, first: pl.Expr) -> pl.Expr:
    """A unit-variance AR(1) process driven by unit-variance shocks `value`.

    x_t = phi x_{t-1} + sqrt(1 - phi^2) e_t, started from its stationary
    distribution. Written as an exponentially weighted mean, which polars
    computes in one pass; group with `.over` for one process per asset.
    """
    if phi == 0.0:
        return value
    alpha = 1.0 - phi
    stationary = math.sqrt(1.0 - phi**2)
    shocks = pl.when(first).then(value * (alpha / stationary)).otherwise(value)
    return shocks.ewm_mean(alpha=alpha, adjust=False) * (stationary / alpha)


def _period_states(
    periods: pl.DataFrame, config: Config, world: _World
) -> pl.LazyFrame:
    """Per period: market volatility, signal strength and return factors."""
    ts = config.schema.timestamp
    first = pl.col(ts) == pl.col(ts).first()
    log_vol = _ar1(pl.col(_VOL_SHOCK), config.vol_persistence, first)
    vov = config.vol_of_vol
    regime = (
        (pl.col(_SWITCH) < config.regime_switch_prob).cum_sum() + world.first_regime
    ) % 2
    return periods.lazy().select(
        ts,
        *_factor_names(config.n_return_factors, "_F"),
        (vov * log_vol - vov**2 / 2.0).exp().alias(_VOL),
        pl.when(regime == 0)
        .then(1.0)
        .otherwise(config.regime_off_strength)
        .alias(_STRENGTH),
    )


def _features(config: Config, world: _World, asset: str) -> list[pl.Expr]:
    """Each feature: its factor loadings times the persistent feature factors,
    plus its own persistent component, with unit variance."""
    first = pl.col(_FIRST)
    factors = [
        _ar1(pl.col(name), world.factor_persistence, first).over(asset)
        for name in _factor_names(config.n_feature_factors, "_g")
    ]
    out = []
    for name in config.feature_names:
        direction = world.directions[name]
        own = _ar1(pl.col(f"_e{name}"), world.persistence[name], first).over(asset)
        share = sum(v * v for v in direction)
        out.append(
            pl.sum_horizontal(
                math.sqrt(1.0 - share) * own,
                *(v * g for v, g in zip(direction, factors)),
            ).alias(name)
        )
    return out


def _per_asset(values: tuple[float, ...], config: Config) -> pl.Expr:
    """A per-asset constant, matched to rows by the asset's position."""
    assets = pl.col(config.schema.asset).to_physical()
    return pl.lit(pl.Series(values, dtype=pl.Float64)).gather(assets)


def _target(config: Config, world: _World) -> pl.Expr:
    """The next-period return (see `generate`)."""
    truth = world.truth
    terms = [w * pl.col(name) for name, w in truth.weights.items()]
    if truth.interaction is not None:
        a, b = truth.interaction
        product = pl.col(a) * pl.col(b) - world.interaction_mean
        terms.append(truth.interaction_weight * product)
    strength = pl.col(_STRENGTH)
    noise = (1.0 - strength**2 * world.signal_var).sqrt() * pl.col(_IDIO)
    z = strength * pl.sum_horizontal(terms) + noise
    q = config.return_factor_share
    idio = math.sqrt(1.0 - q) * _per_asset(world.idio_vol, config) * z
    k = config.n_return_factors
    common = [
        math.sqrt(q / k)
        * _per_asset(tuple(b[j] for b in world.betas), config)
        * pl.col(f"_F{j}")
        for j in range(k)
    ]
    return _RETURN_VOL * pl.col(_VOL) * pl.sum_horizontal(idio, *common)


def _leak(config: Config, world: _World, ts: str) -> list[pl.Expr]:
    """The leaky feature: the target's standardised cross-sectional rank plus
    noise, if any. Ranks rather than values keep its rank IC at
    `Config.leaky_ic` even when returns have fat tails."""
    if world.truth.leaky is None:
        return []
    rank = pl.col(config.schema.target).rank().over(ts)
    standard = (rank - rank.mean().over(ts)) / rank.std().over(ts)
    r = spearman_to_pearson(config.leaky_ic)
    noise = math.sqrt(1.0 / r**2 - 1.0) * pl.col(_LEAK_NOISE)
    return [(r * (standard + noise)).alias(world.truth.leaky)]


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
    return EvalData(
        generate(seed, config).panel,
        cost=config.cost,
        periods_per_year=config.periods_per_year,
    )


def hidden_data(seed: int, config: Config = DEFAULT_CONFIG) -> EvalData:
    """Scoring data: visible then hidden periods, scored on the hidden ones.

    The leaky column, if any, is the suspect for the column dependence check.

    Args:
        seed: Dataset seed.
        config: Generator parameters.

    Returns:
        The full panel, the first hidden timestamp, the suspect columns and
        a seed for the checks derived from `seed`.
    """
    syn = generate(seed, config, hidden=True)
    suspects = () if syn.truth.leaky is None else (syn.truth.leaky,)
    # As secret as the dataset seed it is derived from.
    check_seed = _hash_int(8, "checks", seed)
    return EvalData(
        syn.panel,
        syn.hidden_start,
        suspects,
        check_seed,
        cost=config.cost,
        periods_per_year=config.periods_per_year,
    )
