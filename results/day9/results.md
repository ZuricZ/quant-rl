# Day 9 results: no-tools eval before training

The base model of the training run on the no-tools eval split (spec,
Decision 29), at 8,192 tokens per reply. Output of:

    uv run python scripts/run_models.py qwen3_5-9b --taskset quant-research-notools \
        --secret S --tasks 30 --rollouts 4
    uv run python scripts/run_baselines.py --taskset quant-research-notools --secret S
    uv run python scripts/results.py outputs/models/qwen3_5-9b-notools --baselines baselines.csv

Qwen3.5-9B on OpenRouter, thinking mode, temperature 0.6, `max_tokens` 8192;
30 tasks × 4 rollouts, graded in Docker with the image of main at e5fc0af
(26 November 2026, after the IC profile checks of Decision 30). The secret is
private and not recorded here. All 120 episodes `ok`
(`traces/qwen3_5-9b-notools.jsonl.gz`).

| Agent              | Tasks | Episodes | Honest | Lookahead rate | Implausible IC rate | Invalid rate | Reward | Net Sharpe | Turnover | IC    |
|--------------------|-------|----------|--------|----------------|---------------------|--------------|--------|------------|----------|-------|
| naive              | 30    | 30       | 30     | 0.000          | 0.000               | 0.000        | 0.250  | 1.088      | 0.461    | 0.012 |
| template           | 30    | 30       | 30     | 0.000          | 0.000               | 0.000        | 0.303  | 1.575      | 0.439    | 0.015 |
| oracle             | 30    | 30       | 30     | 0.000          | 0.000               | 0.000        | 0.433  | 2.364      | 0.464    | 0.022 |
| qwen3_5-9b-notools | 30    | 120      | 16     | 0.008          | 0.025               | 0.833        | -0.862 | 0.768      | 0.369    | 0.008 |

Leak rate is 0 for every agent. Performance columns cover honest submissions
only; rates and reward cover every episode.

Why the 100 invalid episodes failed:

- 73 ran out of tokens while thinking (`finish_reason` length, no closing
  `</think>`), so the reply had no answer. The 47 replies that finished used
  a median of 1,972 completion tokens (p90 5,011, max 7,927): the model either
  answers in a few thousand tokens or keeps thinking past the limit.
- 11 had no `Alpha` subclass.
- 16 raised errors in polars: a column used in the same `with_columns` call
  that creates it (`f00_z`, `momentum`), removed arguments
  (`rolling_mean(min_periods=)`), `Expr.corr`, the truth value of an `Expr`.

Of the 20 alphas that ran, 1 was flagged by lookahead and 3 by the IC profile.
The 16 honest ones reach a mean IC of 0.008, below the naive baseline.

Because of the token cutoffs, the training candidates now sample 16,384
tokens per reply, and the no-tools prompt states the limit
(`reply_tokens`). This table is the 8,192-token baseline; the
before/after comparison uses the rerun at 16,384.
