# Day 12: no-tools base checks before the hosted RL run

The hosted RL run's first evals (`runs/prime-rl/lora-9b.toml`) scored almost
every reply -1. Before spending more GPU time, these checks ran the untrained
models on the no-tools taskset through OpenRouter, graded in Prime sandboxes
with the task image `prime/zuricz/quant-research:0.1.4` and the 2 CPU / 8 GB
of `RESOURCES`, as in the hosted run. Each run has 30 eval tasks × 2 rollouts
at temperature 0.6, top_p 0.95 and top_k 20, on 27 November 2026.

| Run (trace in `traces/`) | Model | Prompt | Thinking, tokens | Settings (`runs/openrouter/`) |
|---|---|---|---|---|
| `qwen3_5-9b-nothink-notools` | Qwen3.5-9B | `main` at f790fb7 | mostly off, 12,288 | `nothink-topk.toml` |
| `qwen3_5-9b-think-interface-notools` | Qwen3.5-9B | interface fix (08ecf74) | on, 16,384 | `think-9b.toml` |
| `qwen3_5-9b-think-polars-notools` | Qwen3.5-9B | + polars 2 notes (d371c80) | on, 16,384 | `think-9b.toml` |
| `qwen3_5-35b-a3b-think-polars-notools` | Qwen3.5-35B-A3B | + polars 2 notes (d371c80) | on, 16,384 | `think-35b-a3b.toml` |

Per-episode rows are in `rows.csv` (`scripts/results.py`).

**The tasks are not the hosted run's eval tasks.** Every run used the
placeholder secret `...`, not the training secret: the command that read
the secret from `lora-9b.secret.toml` matched the header comment
(`` `env.taskset.secret = "..."` ``) before the real line. The four runs
share the same 30 tasks, so they compare with each other, but not with the
hosted run's step-0 eval. Rerun with `--secret ...` to reproduce them.

## Result

| | 9B, no thinking | 9B, thinking, interface fix | 9B, thinking, + polars notes | 35B-A3B, thinking, + polars notes |
|---|---|---|---|---|
| Mean reward | -0.550 | -0.505 | **-0.423** | -0.462 |
| Valid | 23 | 26 | 31 | 33 |
| Caught by a check (-1.5) | 0 | 0 | 1 (lookahead) | 4 (2 lookahead, 2 implausible IC) |
| Honest alphas | 23 | 26 | 30 | 29 |
| Honest alphas, median reward | 0.163 | 0.136 | 0.187 | 0.191 |
| Redefined `Alpha` | 17 | 0 | 0 | 0 |
| polars 1 API errors | about 8 | 19 | 9 | 10 |
| Invented columns | 4 | 6 | 4 | 9 |
| Cut off, no code | 4 | 5 | 9 | 7 |
| Median output tokens | 780 | 2,858 | 3,252 | 2,319 |
| OpenRouter cost | $0.03 | $0.10 | $0.09 | $0.38 |

Differences of 3 to 5 episodes are within noise at 60 episodes. The clear
effects are the interface fix (17 to 0), the polars notes (19 to 9) and the
35B-A3B's 4 flagged alphas.

## Findings

- **Showing `class Alpha:` made models redefine it.** The prompt's interface
  block was the base class itself, and 17 of 60 replies copied it and filled
  it in instead of subclassing, though the prompt said not to. The block now
  reads `class MyAlpha(Alpha):` (both tasksets; PR #26).
- **Polars 1 APIs were the largest failure left.** `rolling_*(min_periods=)`
  (now `min_samples`), `Expr.corr` and `pl.spearman_corr` (now
  `pl.corr(a, b, method="spearman")`). Without tools the model cannot look
  them up and only sees -1. The no-tools prompt now names them; the tools
  taskset has `docs` and backtest errors instead (PR #26).
- **Thinking did not produce better alphas here.** It produced a few more
  valid ones (26 against 23) with similar honest rewards, at about four
  times the tokens. The hosted run keeps thinking on, at 16,384 tokens: with
  thinking, 5 of 60 hit 16,384, 9 would have hit 12,288 and 18 would have
  hit 8,192.
- **Without a provider setting, thinking varied by provider.** OpenRouter
  routed the no-thinking run across its providers for the model (bf16, fp8
  and fp4); 46 of 60 replies had no reasoning and 4 thought until cut off.
  The thinking runs turn reasoning on and limit the providers.
- **Qwen3.5-35B-A3B was no better.** Slightly more valid code, but more of it
  cheats (4 flagged against 1) and invents columns (9 against 4), and its
  honest alphas score the same (median 0.191 against 0.187). OpenRouter serves it in fp8 at best. The
  model the registry names for training is Qwen3.6-35B-A3B, which this check
  did not run.
- **OpenRouter does not bound Qwen's thinking.** With `reasoning.max_tokens`
  300 the model still thought for 914 tokens.

## Problems running the checks

- vf-run passed `top_k` from its command line as the text `"20"` and
  OpenRouter refused every call (0 of 60 episodes); it now comes from a TOML
  settings file.
- Pinning one provider (`primeintellect/bf16`) hit its shared rate limit at
  30 concurrent calls; the runs use any bf16 provider at 10.
