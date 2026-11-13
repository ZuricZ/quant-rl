"""Simulated asset panels with planted, noise and leaky features.

A seed fixes everything: which feature names are real, noise or leaky, the
planted ICs, and the data itself. The visible and hidden days come from
independent random streams, so either can be built without the other.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import numpy as np
import pandas as pd

Split = Literal["visible", "hidden"]

_START_DATE = "2000-01-03"
_RETURN_VOL = 0.02


@dataclasses.dataclass(frozen=True)
class Config:
    """Generator parameters. The defaults follow the build spec.

    Attributes:
        n_assets: Assets per day.
        n_visible: Days the agent can backtest on.
        n_hidden: Days used only for scoring.
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
    n_features: int = 20
    n_real: int = 3
    real_ic_range: tuple[float, float] = (0.02, 0.04)
    leaky_ic: float = 0.3
    no_leak_prob: float = 0.2
    replacement_ic: float = 0.06


DEFAULT_CONFIG = Config()


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    """The role of every feature for one seed.

    Attributes:
        real: Real feature names, sorted.
        noise: Noise feature names, sorted.
        leaky: The leaky feature name, or None on a seed without one.
        planted_ic: Target rank IC of each real feature.
        weights: Weight of each real feature in the standardized next-day
            return. Each equals that feature's Pearson correlation with it.
    """

    real: tuple[str, ...]
    noise: tuple[str, ...]
    leaky: str | None
    planted_ic: dict[str, float]
    weights: dict[str, float]


@dataclasses.dataclass(frozen=True)
class Dataset:
    """One split of one seed.

    Attributes:
        seed: The seed it was built from.
        split: "visible" or "hidden".
        features: Feature columns indexed by (date, asset).
        fwd_returns: Next-day return of each row, on the same index.
        truth: Ground truth for the seed.
    """

    seed: int
    split: Split
    features: pd.DataFrame
    fwd_returns: pd.Series
    truth: GroundTruth

    @property
    def n_days(self) -> int:
        return len(self.features.index.levels[0])

    @property
    def n_assets(self) -> int:
        return len(self.features.index.levels[1])


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
    names = [f"f{i:02d}" for i in range(config.n_features)]
    order = rng.permutation(config.n_features)
    has_leak = rng.random() >= config.no_leak_prob
    ics = rng.uniform(*config.real_ic_range, size=config.n_real).tolist()

    real = [names[i] for i in order[: config.n_real]]
    slot = names[order[config.n_real]]
    noise = [names[i] for i in order[config.n_real + 1 :]]
    if has_leak:
        leaky = slot
    else:
        leaky = None
        real.append(slot)
        ics.append(config.replacement_ic)

    planted = dict(zip(real, ics))
    weights = {k: spearman_to_pearson(v) for k, v in planted.items()}
    return GroundTruth(
        real=tuple(sorted(real)),
        noise=tuple(sorted(noise)),
        leaky=leaky,
        planted_ic={k: planted[k] for k in sorted(planted)},
        weights={k: weights[k] for k in sorted(weights)},
    )


def generate(
    seed: int, split: Split = "visible", config: Config = DEFAULT_CONFIG
) -> Dataset:
    """Builds one split of the dataset for a seed.

    Features are independent standard normals. The standardized next-day
    return is a weighted sum of the real features plus noise, scaled to a
    daily volatility of 2%. The leaky feature is the standardized return plus
    enough noise to give it the configured rank IC.

    Args:
        seed: Non-negative integer seed.
        split: "visible" for the agent's days, "hidden" for scoring days.
        config: Generator parameters.

    Returns:
        The requested split.

    Raises:
        ValueError: If split is not "visible" or "hidden".
    """
    if split not in ("visible", "hidden"):
        raise ValueError(f"unknown split: {split!r}")
    truth = ground_truth(seed, config)
    stream = _streams(seed)[1 if split == "visible" else 2]
    rng = np.random.default_rng(stream)
    n_days = config.n_visible if split == "visible" else config.n_hidden
    n_assets, n_features = config.n_assets, config.n_features

    x = rng.standard_normal((n_days, n_assets, n_features), dtype=np.float32)
    col = {f"f{i:02d}": i for i in range(n_features)}

    w = np.array(list(truth.weights.values()))
    real_idx = [col[k] for k in truth.weights]
    z = x[:, :, real_idx].astype(np.float64) @ w
    z += math.sqrt(1.0 - float(w @ w)) * rng.standard_normal((n_days, n_assets))
    if truth.leaky is not None:
        r = spearman_to_pearson(config.leaky_ic)
        leak = z + math.sqrt(1.0 / r**2 - 1.0) * rng.standard_normal(z.shape)
        x[:, :, col[truth.leaky]] = leak / math.sqrt(1.0 / r**2)

    offset = 0 if split == "visible" else config.n_visible
    dates = pd.bdate_range(_START_DATE, periods=config.n_visible + config.n_hidden)
    dates = dates[offset : offset + n_days]
    assets = [f"A{i:03d}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    features = pd.DataFrame(
        x.reshape(n_days * n_assets, n_features), index=index, columns=list(col)
    )
    fwd_returns = pd.Series(
        (_RETURN_VOL * z).reshape(-1), index=index, name="fwd_return"
    )
    return Dataset(seed, split, features, fwd_returns, truth)


def _streams(seed: int) -> list[np.random.SeedSequence]:
    """Independent streams for metadata, visible days and hidden days."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.SeedSequence(seed).spawn(3)
