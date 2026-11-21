# quant-research-notools

The single-reply version of `quant-research`, made for RL training. There are
no tools and no backtests. The prompt describes the data and lists each
feature's score on the visible periods; the model answers with alpha code in
one Python block.

## Task

- **Prompt.** The data description, a table of each feature's (and `ret`'s)
  visible-period rank IC and net Sharpe used alone as the prediction, the
  `Alpha` format, the portfolio rules, one explained example and the style
  request.
- **Answer.** The last fenced Python block of the reply, defining one
  `Alpha` subclass.
- **Data.** The same markets, seeds and variation as `quant-research`: task i
  of a split is the same dataset in both.

## Reward

`hidden_ic`: the alpha's rank IC on the hidden periods divided by the IC
ceiling (0.05), so an honest alpha scores between −1 and 1.
- −1 if the reply has no Python block or the code fails.
- −1.5 if a lookahead check fires.

Metrics: `no_code`, the check flags, `invalid`, raw IC, net Sharpe and
turnover.

## Running

Grading runs agent code, so it needs the task image built from the
repository root:

```bash
docker build -t quant-research:0.1.0 .
export QUANT_RESEARCH_SECRET=<your secret>
uv run vf-validate quant-research-notools --runtime.type docker -n 2
```

A prime-rl config for a LoRA smoke run on Qwen3-4B is in the repository at
`runs/prime-rl/notools-smoke.toml`. Settings are as for `quant-research`,
without the tool settings.
