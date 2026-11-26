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

## What the cut-off replies were thinking

All 73 were cut inside the thinking: none had started the answer. Across 120
replies at temperature 0.6, the same prompt finished in some rollouts and ran
out in others (only 5 of 30 tasks ran out in all 4), so a reply either
answers or falls into a mode it does not leave.

- **Re-verifying rules, not designing.** 42 of the 73 had already drafted the
  full alpha in their thinking (median at 56% of it); the rest was spent
  re-reading the prompt. 22% of their thinking quotes the prompt back,
  against 3% in replies that finished. Typical: a trivial `-ret` reversal
  decided early, then "Wait, one more constraint:" over "features is called
  once", "a fresh instance is fit", "predict gets the block", each confirmed
  ("This is consistent.") and checked again. A median 40 "Wait" per cut
  reply, 0 per finished one.
- **The most re-checked rules** (episodes of 73 with a "Wait" on them):
  `features` once versus per block and state on `self` (69), the run's row
  count and seconds (69; median 10 mentions per cut reply, 0 per finished
  one), whether `predict` sees `fwd_return` (69), feature choice and signs
  (67), whether the IC can reach the 0.05 ceiling (45: the ceiling read as a
  target).
- **Not the style request.** Cut rate by request: none 75%, momentum 75%,
  reversal 67%, low turnover 50%, combine several features 38%.
- **Longer thinking did not pay.** The best rewards came at 500 to 2,300
  reasoning tokens; finished replies over 4,000 tokens were valid 2 times in
  8. Very short ones failed on the interface: 11 replies redefined
  `class Alpha:` instead of subclassing it.

## Changes for the rerun

Replies get 16,384 tokens (`runs/models.toml`); the prompt does not mention
a limit. Neither taskset's prompt states the checks (the IC ceiling, the
decay rules) or the reward and penalties any more: the ceiling told a leaking
alpha how far it could go, and the reward belongs to the trainer. The no-tools
prompt also drops the scored run's rows and seconds (vectorised polars
expressions finish well inside the limit) and says to subclass `Alpha`
rather than redefine it.
