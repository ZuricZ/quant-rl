# Day 10: does the `docs(name)` tool help?

Qwen3.5-9B on the tools taskset (`quant-research`), 30 eval tasks × 3
rollouts, backtest budget 10, 16k completion tokens, 30 turns, via
OpenRouter. The two runs differ only in the `docs(name)` tool and its one
prompt line:

- **before**: `main` at 24966ad, image `quant-research:0.1.1`
  (`traces/qwen3_5-9b-tools-before.jsonl.gz`);
- **after**: branch `docs-tool` at af2d1bc, image `quant-research:0.1.2`
  (`traces/qwen3_5-9b-tools-after.jsonl.gz`).

Per-episode rows of both are in `rows.csv` (`scripts/results.py`).

## Result: `docs` made the model worse

| | before | after (with `docs`) |
|---|---|---|
| Mean reward | **-0.028** | **-0.291** |
| Invalid submissions | 13 / 90 | 25 / 90 |
| Honest submissions | 74 | 59 |
| Net Sharpe on hidden data (honest) | 0.658 | 0.427 |
| Claim gap (honest) | 0.031 | 0.124 |
| Backtests returning an error | 46.5% | 55.7% |
| … of which library misuse (AttributeError, TypeError, …) | 10.7% of backtests | 16.7% of backtests |
| `docs` calls per episode | — | 1.76 |

Paired by task, the after run is lower by 0.263 in mean reward (21 of 30
tasks worse; bootstrap 95% interval -0.43 to -0.09), so this is not
rollout noise. An earlier run that had `docs` in 78 of 90 episodes, by
mistake (the checkout was switched mid-run), scored -0.153: between the two,
in the same direction.

## Where the loss comes from

Most final submissions are code that was never backtested exactly as
submitted: the model edits after its last backtest and submits. That holds
in both runs (58 of 90 before, 66 after), but that untested code fails far
more often with `docs`: 8 of 58 (14%) before, 20 of 66 (30%) after. Code
that passed a backtest and was submitted unchanged is about as reliable in
both (4 of 31 vs 3 of 22 invalid, mostly hidden-data time limits).

The extra failures are library misuse: `AttributeError` on polars
`Expr`/`Series`/`DataFrame`, `SchemaError`, `InvalidOperationError`, an
sklearn shape mismatch, `NameError` for an unimported `Ridge`, and 4
submissions with no subclass of `Alpha`. With docs at hand the model reaches
for unfamiliar API (polars-ds was the most looked-up package, 70 of 158
lookups) and trusts it without running it.

What it looked up: polars-ds 70, polars 46, scikit-learn 18, numpy 1, scipy
0, and our own `Alpha` 23 times (refused: it is not a library). 41 lookups
were refused or named something that does not exist, e.g.
`polars.Expr.corr`, `pds.rank_corr`, `polars.Expr.standardize`.

## Caveats

One model, one run per arm, 90 episodes each. A model trained with RL on
this taskset could learn to use `docs` well; this measures the untrained
model only.

## Next

Do not ship `docs` as is. Unchecked final edits are the main failure in
both runs, which is what `check(code)` (compile and run the alpha on
placeholder data, no scores) targets; measure that next, then `docs` again
on top of it.
