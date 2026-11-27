# Day 11: does the `check(code)` tool help?

Qwen3.5-9B on the tools taskset, 30 eval tasks × 3 rollouts, backtest
budget 10, 16k completion tokens, 30 turns, via OpenRouter. The runs differ
only in the `check(code)` tool and its prompt line:

- **main**: `main` at 24966ad, image `quant-research:0.1.1`; the same run
  as day 10's "before" (`traces/qwen3_5-9b-tools-main.jsonl.gz`);
- **check**: branch `check-tool` at 548166b, image `quant-research:0.1.3`
  (`traces/qwen3_5-9b-tools-check.jsonl.gz`).

Per-episode rows are in `rows.csv`.

## Result: no gain; turns run out

| | main | with `check` |
|---|---|---|
| Mean reward | -0.028 | -0.142 |
| Invalid submissions | 13 / 90 | 21 / 90 |
| No submission (30 turns used) | 0 | 6 |
| Net Sharpe on hidden data (honest) | 0.658 | 0.534 |
| Backtests per episode | 10.7 | 9.8 |
| Backtests returning an error | 46.5% | 42.0% |
| `check` calls per episode | — | 4.8 (49% errors) |
| Tool calls per episode | 12.7 | 16.6 |

Paired by task the check run is lower by 0.113 (16 of 30 tasks worse;
bootstrap 95% interval -0.28 to +0.05): not significant, but no gain.

## Why

- **The model uses `check` while exploring, not on its final edit.** Of 84
  submissions, 54 were still never run as submitted, by `check` or
  `backtest`; only 2 were code that passed a `check` but no backtest.
  Untested submissions failed 13 of 54 times (24%; 14% on main).
- **Checks spend turns.** 8 episodes reached the 30-turn limit and 6 of them
  never submitted, scoring as invalid. Main had none.
- **7 submissions redefined `Alpha`** instead of subclassing it ("found 0"
  subclasses), 4 of them after only one or two tool calls. Main had 1.
- `check` errors themselves look as intended: sklearn NaN input, shape
  mismatches, `NotImplementedError` from an unoverridden `predict`, missing
  imports, all-null predictions, time limits.

## Options

1. Make `submit` run the check first and refuse failing code with its
   error, leaving the episode open. This targets the failure directly
   (untested final edits) without coaching, and makes a standalone `check`
   unnecessary.
2. Keep `check` and raise the turn limit, so it does not crowd out `submit`.
3. Drop it.
