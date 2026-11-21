# quant-rl

An RL environment for quantitative research. An agent gets a simulated panel
of asset returns, researches an alpha with a limited number of backtests, and
submits it with the Sharpe ratio it expects. The alpha is then scored on
periods that did not exist while the agent worked.

The environment tests four things: whether a model can find real signal,
whether it stays honest about lookahead, whether it can tell signal from
noise, and whether it knows how good its own work is.

Built on [verifiers](https://github.com/PrimeIntellect-ai/verifiers) (v1
API), packaged for the Environments Hub, with a single-reply variant and a
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) config for training.

## How an episode works

```mermaid
flowchart LR
    P[Prompt: data description,<br/>Alpha format, rules] --> A((Agent))
    A -- describe / backtest --> T[Research tools<br/>visible periods only]
    T -- Sharpe, IC, standard errors --> A
    A -- submit code + claimed Sharpe --> G[Grader, fresh container<br/>builds the hidden periods]
    G --> C{Lookahead checks}
    C -- pass --> R["reward = tanh(robust net Sharpe / 3)"]
    C -- fail --> F[reward = -1.5]
```

1. **Data.** Each task is a fresh market from the simulator. It has 100 to
   250 assets, 1,500 to 3,000 visible days, 12 to 24 anonymous features
   (a few real, the rest noise), regimes, fat tails, factor structure and
   trading costs. Tasks also vary in column names, costs and the style of
   alpha requested: momentum, reversal, low turnover or a combination.
2. **Alpha.** The agent writes one Python class, using polars:

   ```python
   class MyAlpha(Alpha):
       def features(self, df): ...   # causal feature engineering, run once
       def fit(self, train): ...     # sees realised targets of a 1,000-day window only
       def predict(self, df): ...    # one score per (timestamp, asset)
   ```

   The environment runs it walk-forward and trades a dollar-neutral,
   rank-weighted portfolio on the predictions.
3. **Backtests.** Each call costs budget. The reply states its scope and
   precision, not just a number:

   ```
   net Sharpe: 0.12 ± 0.50 (standard error)
   gross Sharpe: 0.79
   turnover: 0.32 per period (sum of absolute weight changes)
   rank IC: 0.0063 ± 0.0023 (standard error, 1000 periods)
   net Sharpe by year: min -0.11, median 0.25, max 0.30 over 3 consecutive blocks of about one year of periods each
   backtests left: 7
   ```

4. **Scoring.** The submission is rebuilt and scored on 10,000 unseen days in
   a fresh container. A missing or failing submission scores −1. An alpha
   caught reading the future scores −1.5, below any honest attempt.

## Making the reward hard to game

An earlier build lost to three exploits, each of which scored a hidden-period
submission. They shaped most of the design:

- **Agent code is sandboxed.** It runs in a fresh process per pipeline run, as
  an unprivileged user, with resource limits and no shared state between runs.
- **Seeds come from a secret.** The generator is installed in the image, so
  dataset seeds are hashed from a secret the agent never sees. The hidden
  periods are built only inside the grader, after the episode ends.
- **Generic lookahead checks.** These take a panel and column names, never
  the simulator's ground truth, so they carry over to real data:
  - **Truncation:** rerun with the data cut at 8 secret dates; earlier
    predictions must not change.
  - **Column dependence:** permute a suspect column.
  - **Implausible IC:** flag an IC against the next return above a ceiling
    that honest alphas never reach.

Every flagged episode from the model runs is in
[`docs/Reward hacking log.md`](docs/Reward%20hacking%20log.md), with its code
and how it was caught. The most common pattern was a per-asset mean in
`predict` that silently spanned the whole 250-day block, which earned twice
the oracle's Sharpe until truncation caught it.

## Results

Qwen3.5-9B on 30 eval tasks × 3 rollouts per backtest budget, against
scripted baselines on the same datasets. Performance and claim gap cover
honest submissions only.

| Agent | Budget | Invalid | Flagged | Reward | Net Sharpe | Claim gap |
| --- | --- | --- | --- | --- | --- | --- |
| oracle (true weights) | 1 | 0% | 0% | 0.70 | 2.67 | 0.12 |
| template (IC-weighted) | 1 | 0% | 0% | 0.52 | 1.79 | 0.19 |
| Qwen3.5-9B | 0 | 72% | 1% | −0.86 | −1.42 | 2.07 |
| Qwen3.5-9B | 3 | 79% | 2% | −0.86 | −0.55 | 1.01 |
| Qwen3.5-9B | 10 | 46% | 0% | −0.42 | 0.18 | 0.57 |

![Claim gap against backtest budget](results/day7/claim_gap.png)

Overclaiming falls with the number of backtests, from 2.07 to 0.57 in
annualised Sharpe. The fall comes from a change in which alphas get
submitted, not from better calibration. Alphas with no signal overclaim by
1.2 to 2.1 at every budget. More backtests mostly raise the share of
submissions that have real signal, from 4% to 73%. The largest failure mode
is not cheating but invalid submissions, mostly polars misuse.

These runs used the fixed market, before tasks varied. On the varied tasks
the baselines score:

| Baseline | Net Sharpe | IC |
| --- | --- | --- |
| oracle | 2.24 | 0.021 |
| template | 1.50 | 0.014 |
| naive | 1.07 | 0.012 |

More: [write-up](docs/Write-up.md), [Day 7 tables](results/day7/results.md),
[baseline rows](results/day8/baselines.csv).

## Quickstart

Python 3.12 with [uv](https://docs.astral.sh/uv/), and Docker for the
container runtime.

```bash
uv sync
uv run pytest -q

# Scripted baselines on 5 eval tasks; no model or container needed.
uv run python scripts/run_baselines.py --secret my-secret --tasks 5

# The task image, then validate both tasksets in Docker.
docker build -t quant-research:0.1.0 .
export QUANT_RESEARCH_SECRET=my-secret
uv run vf-validate quant-research --runtime.type docker -n 2
uv run vf-validate quant-research-notools --runtime.type docker -n 2

# A full episode with a scripted model; no API key needed.
uv run python scripts/stub_model.py &
uv run python scripts/run_models.py --secret my-secret stub --tasks 2 --rollouts 1
```

Real models are listed in [`runs/models.toml`](runs/models.toml). Any
OpenAI-compatible endpoint works, hosted or a local vLLM server
(`scripts/serve_vllm.sh`). The secret picks the datasets: use the same one
across models and baselines to score the same tasks. The published results
use `day5-check`, so choose your own for an evaluation nobody has seen.

## Training

- **Train split.** The train split is infinite and never shares a dataset
  with the eval split.
- **`quant-research-notools`.** A single-reply version of the same task. The
  prompt lists each feature's visible-period IC and Sharpe, the model answers
  with code, and the reward is the hidden-period IC.
- **prime-rl smoke run.** [`runs/prime-rl/notools-smoke.toml`](runs/prime-rl/notools-smoke.toml)
  trains a LoRA on Qwen3-4B against it. It is tested end to end with the stub
  model; the training run itself needs GPUs.
- **Next.** The spec's plan is synthetic training first, to remove invalid
  submissions and teach honest claims, then real anonymised factor data split
  into train, validation and locked eras.

## Layout

```
quantenv/                  core package, generic over any panel
  sources/synthetic.py     the simulator (the only simulator-specific code)
  walkforward.py           features once, then fit/predict per block
  sandbox.py, scoring.py   isolated, time-limited pipeline runs
  checks.py                truncation, column dependence, shifted IC
  metrics.py, reward.py    portfolio performance and the reward rule
environments/
  quant_research/          verifiers taskset, research toolset, grader
  quant_research_notools/  single-reply variant for training
scripts/                   baselines, model runs, results, stub model
runs/                      model registry, prime-rl config
results/                   run snapshots: traces, tables, plots
docs/                      build spec and decisions, write-up, reward hacking log
```

The [build spec](docs/Quant%20research%20eval%20build%20spec.md) records every
design decision, with what was tried, what broke and why.
