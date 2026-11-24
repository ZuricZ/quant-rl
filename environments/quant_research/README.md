# quant-research

A multi-turn research task. The agent gets a simulated panel of asset
returns, tests alpha ideas with a limited number of walk-forward backtests,
and submits one alpha with the net Sharpe ratio it expects. The submission is
scored on unseen periods, which are built only after the episode ends.

## Task

- **Data.** Each task has its own market: 100 to 250 assets, 1,500 to 3,000
  visible daily periods and 12 to 24 anonymous features, a few of them real.
  Tasks vary in column names, cost (1 to 5 bp) and the style of alpha the
  prompt asks for. Momentum and reversal are planted in past returns.
- **Submission.** Python defining one subclass of `Alpha`, with `features`,
  `fit` and `predict`, written in polars. The environment runs it
  walk-forward and trades a dollar-neutral, rank-weighted portfolio on its
  predictions.
- **Tools**, served over MCP and colocated in the rollout's container:

  | Tool | Returns |
  | --- | --- |
  | `describe()` | Columns, timestamp range, number of assets |
  | `backtest(code)` | Net Sharpe ± s.e., gross Sharpe, turnover, rank IC ± s.e., per-year range. Costs one unit of budget. |
  | `submit(code, claimed_sharpe)` | Ends the episode |

## Reward

`hidden_sharpe`: tanh(S / 3) of the robust annualised net Sharpe S on the
hidden periods.
- −1 if there is no submission or the code fails.
- −1.5 if a lookahead check fires (truncation, column dependence, or an
  implausible IC profile: |IC| at or above `max_ic` against any future
  return, or an IC that rises after the next return).

An honest attempt always scores above both penalties.

Each trace also records metrics: the flags, net and gross Sharpe, turnover,
IC, claim gap (claimed minus realised Sharpe), backtests used and shifted
ICs at -1, +1 and +2; the grader's score keeps the whole IC profile.

## Running

The task runs agent code, so it needs a container runtime. Build the image
from the repository root, then:

```bash
docker build -t quant-research:0.1.0 .
export QUANT_RESEARCH_SECRET=<your secret>
uv run vf-validate quant-research --runtime.type docker -n 2
uv run vf-run quant-research --model <model> --client.base-url <url> \
    --env.agent.runtime.type docker -n 30 -r 3
```

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `env.taskset.secret` | `$QUANT_RESEARCH_SECRET` | Required. Dataset seeds are hashed from it, and agent code never sees it. |
| `env.taskset.split` | `eval` | `eval`: 30 fixed tasks, 6 per style. `train`: infinite, never shares a dataset with eval. |
| `env.taskset.task.tools.budget` | 10 | Backtests per episode |
| `env.taskset.task.max_turns` | 30 | Model turns per episode |
| `env.taskset.task.vary` | true | Vary market and prompt per task; false gives the fixed market of the earliest runs |
| `env.taskset.task.check_config.max_ic` | 0.05 | IC ceiling of the implausible-IC check, at every future lag (set it through a TOML file) |
| `env.taskset.task.image` | `quant-research:0.1.0` | Task image; a registry reference for a remote runtime |
