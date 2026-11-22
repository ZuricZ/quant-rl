"""Taskset pieces both quant research environments share.

Splits and the dataset secret, task data, the task image, and the prompt
sections and values. One of the two `quantenv` modules that import verifiers
(with `quantenv.grading`); the rest of the package stays free of it, so the
core runs without the eval stack.
"""

import os
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field, SecretStr, field_validator

from quantenv import checks, reward, scoring, walkforward
from quantenv.panel import Panel
from quantenv.sources import synthetic
from quantenv.walkforward import WalkForward

EVAL_TASKS = 30
# What pydantic writes in place of a SecretStr when dumping.
_MASK = "**********"
# Built from the repository's Dockerfile; holds this package.
IMAGE = "quant-research:0.1.0"
# Environment variable a taskset reads its secret from when none is set.
SECRET_VAR = "QUANT_RESEARCH_SECRET"

# Sections both environments build their prompts from.
DATA_SECTION = """\
## Data

{description}

One row per (timestamp, asset), sorted by timestamp then asset. `ret` is the \
return realised at that timestamp, from the previous period to this one. The \
features are anonymous. The target, `fwd_return` (the return from this \
period to the next), is shown only to `fit`, and only for rows whose target \
was already realised.

"""

ALPHA_SECTION = """\
## Alpha

Write Python that defines exactly one subclass of the predefined class \
`Alpha`. `pl` (polars) and `Alpha` are available without imports.

```python
class Alpha:
    def features(self, df: pl.DataFrame) -> pl.DataFrame:
        return df                    # default: no new features
    def fit(self, train: pl.DataFrame) -> None:
        pass                         # default: nothing to fit
    def predict(self, df: pl.DataFrame) -> pl.Series | pl.Expr:
        raise NotImplementedError    # must be overridden
```

- `features(df)` gets every row and returns the same rows in the same order \
with any added columns. A value at timestamp t may only use rows at or \
before t.
- `fit(train)` gets the engineered rows of the {window} timestamps just \
before the current block, plus `fwd_return`. Keep fitted state on `self`.
- `predict(df)` gets the engineered rows of one block and returns one score \
per row, in row order, as a `pl.Series` or a `pl.Expr` evaluated on `df`.
- For randomness, use `self.rng`, a `numpy.random.Generator` that the \
environment seeds before each call, so reruns draw the same numbers.

The environment runs the alpha walk-forward: `features` once, then for each \
block of {refit_every} timestamps a fresh instance is fit on the {window} \
timestamps before it and predicts the block. The window slides; it never \
grows. The first {window} timestamps are only for training.

Time limits scale with the rows a run hands to your code: `features` gets \
every row, each block's `fit` the rows of its window, and its `predict` the \
block's rows. A run may take {startup:g} seconds plus {per_million_rows:g} \
seconds per million such rows. \
"""

PORTFOLIO = """\
Each period the assets are ranked by prediction and held with \
dollar-neutral weights proportional to the centred rank, scaled so the \
absolute weights sum to 1. Each period's P&L is the weights times \
`fwd_return`, minus {cost_bp:g} bp per unit of turnover (the sum of absolute \
weight changes). At most {max_null:.0%} of the predictions may be null.\
"""

# The example of tasks without variation (spec, Decision 26).
EXAMPLE = """\
Example:

```python
class MyAlpha(Alpha):
    def predict(self, df):
        return pl.col("ret")
```
"""

# The example of varied tasks: the structure of an alpha, not a submission.
# "COLUMN" is not a column, so a copy fails.
PATTERN_EXAMPLE = """\
Example of the structure only; "COLUMN" stands for a real column name, so \
this exact code fails:

```python
class MyAlpha(Alpha):
    def features(self, df):
        # New columns from current and past rows only: here a 10-period
        # rolling mean of one column, per asset.
        smooth = pl.col("COLUMN").rolling_mean(10).over("asset")
        return df.with_columns(smooth.alias("smooth"))

    def fit(self, train):
        # Learn from the training window, which has `fwd_return`: here the
        # sign of the new column's rank correlation with it.
        corr = train.select(pl.corr("smooth", "fwd_return", method="spearman"))
        self.sign = 1.0 if corr.item() >= 0 else -1.0

    def predict(self, df):
        # One score per row of the block; higher means a larger position.
        return self.sign * pl.col("smooth")
```
"""

# What a varied task asks for, by `synthetic.Config.style`. Not checked by
# the reward yet (spec, open decision 4).
STYLE_REQUESTS = {
    "any": "",
    "momentum": "The desk wants a momentum alpha: one that favours assets "
    "whose returns over recent periods were high.",
    "reversal": "The desk wants a short-term reversal alpha: one that favours "
    "assets whose returns over the last few periods were low.",
    "low_turnover": "The desk wants a low-turnover alpha: positions should "
    "change slowly, since trading costs here are high.",
    "combination": "The desk wants an alpha that combines several features "
    "rather than relying on one.",
}


def prompt_choices(variation: int | None, style: str) -> dict[str, str]:
    """The varied parts of a task's prompt that both environments share
    (spec, Decision 26).

    Args:
        variation: The task's variation seed, or None for a task without
            variation: `EXAMPLE` and no request, as in the runs before
            Decision 26.
        style: The style the task asks for (`synthetic.Config.style`).

    Returns:
        The `request` and `example` format values.
    """
    if variation is None:
        return dict(request="", example=EXAMPLE + "\n")
    request = STYLE_REQUESTS[style]
    return dict(
        request=f"## Request\n\n{request}\n\n" if request else "",
        example=PATTERN_EXAMPLE + "\n",
    )


def run_rows(
    hidden: bool,
    config: synthetic.Config = synthetic.DEFAULT_CONFIG,
    wf: WalkForward = WalkForward(),
) -> int:
    """Rows a run on synthetic data hands to agent code.

    From the data's shape (`walkforward.pipeline_rows`), so the prompt can
    state the limits without building the hidden periods.

    Args:
        hidden: True for a scored run on the hidden periods, False for a
            backtest on the visible ones.
        config: The task's generator parameters.
        wf: Walk-forward settings.

    Returns:
        The number of rows.
    """
    if hidden:
        n_timestamps, n_warmup = config.n_visible + config.n_hidden, config.n_visible
    else:
        n_timestamps, n_warmup = config.n_visible, wf.window
    return walkforward.pipeline_rows(n_timestamps, config.n_assets, n_warmup, wf)


def prompt_values(
    check_config: checks.CheckConfig = checks.CheckConfig(),
    config: synthetic.Config = synthetic.DEFAULT_CONFIG,
) -> dict[str, Any]:
    """The values the prompt sections refer to, apart from the description,
    the episode limits and the varied parts.

    Args:
        check_config: Settings of the checks run when grading; gives the IC
            ceiling.
        config: The task's generator parameters.

    Returns:
        Format values by name.
    """
    wf = WalkForward()
    limit = scoring.TimeLimit()
    backtest_rows = run_rows(hidden=False, config=config)
    score_rows = run_rows(hidden=True, config=config)
    return dict(
        refit_every=wf.refit_every,
        window=wf.window,
        max_ic=check_config.max_ic,
        startup=limit.startup,
        per_million_rows=limit.per_million_rows,
        backtest_rows=backtest_rows / 1e6,
        backtest_limit=limit.seconds(backtest_rows),
        score_rows=score_rows / 1e6,
        n_hidden=config.n_hidden,
        invalid_reward=reward.INVALID_REWARD,
        flagged_reward=reward.FLAGGED_REWARD,
        score_limit=limit.seconds(score_rows),
        cost_bp=config.cost * 1e4,
        max_null=walkforward.MAX_NULL_SHARE,
    )


def secret_from_env() -> SecretStr | None:
    """The secret in `SECRET_VAR`, or None if it is unset or empty."""
    value = os.environ.get(SECRET_VAR)
    return SecretStr(value) if value else None


class SplitConfig(vf.TasksetConfig):
    """Settings that pick a taskset's datasets.

    Attributes:
        split: "eval" is a fixed set of `EVAL_TASKS` tasks; "train" is
            infinite. Their datasets never coincide.
        secret: Required; the taskset will not build without it. Dataset
            seeds are drawn from it, see `synthetic.dataset_seed`. Runs that
            must share datasets, such as several models and
            `scripts/run_baselines.py --secret`, set the same value. Unset,
            it is read from the environment variable `SECRET_VAR`, which
            keeps it out of config files, e.g. a prime-rl config. Dumps show
            it masked; never show it to the agent.
    """

    split: Literal["train", "eval"] = "eval"
    # Optional only because verifiers' CLI builds this config with defaults
    # before it reads the flags; the taskset refuses to build without it.
    secret: SecretStr | None = Field(default_factory=secret_from_env)

    @field_validator("secret")
    @classmethod
    def _not_masked(cls, secret: SecretStr | None) -> SecretStr | None:
        # A dumped config read back holds the mask, which would silently
        # become everyone's secret.
        if secret is not None and secret.get_secret_value() in ("", _MASK):
            raise ValueError("secret is empty or masked; pass the real secret")
        return secret


class ResearchData(vf.TaskData):
    """Task data. Published with every trace, so it holds no seed.

    Attributes:
        split: "train" or "eval".
        index: Dataset number within the split.
    """

    split: Literal["train", "eval"]
    index: int = Field(ge=0)


def describe_panel(panel: Panel) -> str:
    """Columns the alpha sees, timestamp range and number of assets.

    Args:
        panel: The data.

    Returns:
        A short plain-text description.
    """
    stamps = panel.timestamps
    n_assets = panel.frame.get_column(panel.schema.asset).n_unique()
    return (
        f"columns: {', '.join(panel.observable.columns)}\n"
        f"timestamps: {stamps.len()}, from {stamps.first()} to {stamps.last()}\n"
        f"assets: {n_assets}"
    )
