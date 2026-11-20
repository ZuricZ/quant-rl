# Day 7 results

Output of:

    uv run python scripts/results.py outputs/day6/qwen3_5-9b-budget{0,1,3} \
        outputs/day7/qwen3_5-9b-budget10 --baselines results/day6/baselines.csv

Budgets 0, 1 and 3 are the Day 6 runs (`results/day6/traces/`). Budget 10 was
rerun on 19 November 2026 (`traces/qwen3_5-9b-budget10.jsonl.gz`) so that
every budget is prompted and graded with `max_ic` = 0.05 (spec, Decision 24).
Same model (Qwen3.5-9B on OpenRouter, thinking mode, temperature 0.6), eval
split, secret `day5-check`, 30 tasks × 3 rollouts, Docker runtime; all 90
episodes `ok`. The rerun was graded with the current truncation rule
(Decision 23); budgets 0, 1 and 3 with the first one, and their flagged
alphas were rechecked (`results/day6/results.md`).

Performance columns and claim gaps cover honest submissions only (valid, no
check fired); rates, reward and backtests used cover every episode.

| Agent               | Tasks | Episodes | Honest | Leak rate | Lookahead rate | Implausible IC rate | Invalid rate | Reward | Net Sharpe | Gross Sharpe | Turnover | IC     | Claim gap | Backtests used |
|---------------------|-------|----------|--------|-----------|----------------|---------------------|--------------|--------|------------|--------------|----------|--------|-----------|----------------|
| naive               | 30    | 30       | 30     | 0.000     | 0.000          | 0.000               | 0.000        | 0.400  | 1.293      | 2.136        | 0.431    | 0.012  | 0.224     | 1.000          |
| oracle              | 30    | 30       | 30     | 0.000     | 0.000          | 0.000               | 0.000        | 0.703  | 2.669      | 3.553        | 0.454    | 0.020  | 0.120     | 1.000          |
| qwen3_5-9b-budget0  | 30    | 90       | 24     | 0.000     | 0.011          | 0.011               | 0.722        | -0.855 | -1.416     | -0.049       | 1.156    | -0.000 | 2.070     | 0.000          |
| qwen3_5-9b-budget1  | 30    | 90       | 17     | 0.000     | 0.000          | 0.000               | 0.811        | -0.882 | -1.242     | 0.215        | 1.115    | 0.001  | 1.763     | 0.644          |
| qwen3_5-9b-budget10 | 30    | 90       | 49     | 0.000     | 0.000          | 0.000               | 0.456        | -0.422 | 0.180      | 1.185        | 0.583    | 0.007  | 0.573     | 8.556          |
| qwen3_5-9b-budget3  | 30    | 90       | 17     | 0.000     | 0.022          | 0.011               | 0.789        | -0.857 | -0.550     | 0.391        | 0.734    | 0.002  | 1.008     | 2.033          |
| template            | 30    | 30       | 30     | 0.000     | 0.000          | 0.000               | 0.000        | 0.524  | 1.790      | 2.631        | 0.428    | 0.015  | 0.186     | 1.000          |

How episodes ended:
| agent               | stop            | len |
|---------------------|-----------------|-----|
| naive               | baseline        | 30  |
| oracle              | baseline        | 30  |
| qwen3_5-9b-budget0  | agent_completed | 21  |
| qwen3_5-9b-budget0  | submitted       | 69  |
| qwen3_5-9b-budget1  | agent_completed | 30  |
| qwen3_5-9b-budget1  | submitted       | 60  |
| qwen3_5-9b-budget10 | agent_completed | 12  |
| qwen3_5-9b-budget10 | submitted       | 78  |
| qwen3_5-9b-budget3  | agent_completed | 28  |
| qwen3_5-9b-budget3  | max_turns       | 1   |
| qwen3_5-9b-budget3  | submitted       | 61  |
| template            | baseline        | 30  |

Claim gap by budget (honest submissions):
| model      | budget | episodes | claim_gap | se    | abs_gap | claimed | realised | corr   | backtests_used | signal_share |
|------------|--------|----------|-----------|-------|---------|---------|----------|--------|----------------|--------------|
| naive      | 1      | 30       | 0.224     | 0.168 | 0.729   | 1.517   | 1.293    | 0.500  | 1.000          | 1.000        |
| oracle     | 1      | 30       | 0.120     | 0.229 | 1.049   | 2.789   | 2.669    | 0.289  | 1.000          | 1.000        |
| qwen3_5-9b | 0      | 24       | 2.070     | 0.143 | 2.070   | 0.654   | -1.416   | 0.352  | 0.000          | 0.042        |
| qwen3_5-9b | 1      | 17       | 1.763     | 0.256 | 1.804   | 0.522   | -1.242   | -0.039 | 0.941          | 0.235        |
| qwen3_5-9b | 3      | 17       | 1.008     | 0.234 | 1.144   | 0.459   | -0.550   | -0.200 | 3.000          | 0.353        |
| qwen3_5-9b | 10     | 49       | 0.573     | 0.129 | 0.774   | 0.753   | 0.180    | 0.667  | 10.000         | 0.735        |
| template   | 1      | 30       | 0.186     | 0.211 | 0.903   | 1.977   | 1.790    | 0.528  | 1.000          | 1.000        |

Claim gap by budget and alpha type (signal: |IC| > 0.003):
| model      | budget | signal | episodes | claim_gap | se    | abs_gap | claimed | realised | corr   |
|------------|--------|--------|----------|-----------|-------|---------|---------|----------|--------|
| naive      | 1      | true   | 30       | 0.224     | 0.168 | 0.729   | 1.517   | 1.293    | 0.500  |
| oracle     | 1      | true   | 30       | 0.120     | 0.229 | 1.049   | 2.789   | 2.669    | 0.289  |
| qwen3_5-9b | 0      | false  | 23       | 2.075     | 0.150 | 2.075   | 0.661   | -1.414   | 0.353  |
| qwen3_5-9b | 0      | true   | 1        | 1.969     | null  | 1.969   | 0.500   | -1.469   | NaN    |
| qwen3_5-9b | 1      | false  | 13       | 2.063     | 0.252 | 2.063   | 0.488   | -1.575   | -0.297 |
| qwen3_5-9b | 1      | true   | 4        | 0.789     | 0.498 | 0.961   | 0.630   | -0.159   | 0.125  |
| qwen3_5-9b | 3      | false  | 11       | 1.324     | 0.272 | 1.416   | 0.500   | -0.824   | -0.174 |
| qwen3_5-9b | 3      | true   | 6        | 0.429     | 0.352 | 0.646   | 0.383   | -0.046   | -0.258 |
| qwen3_5-9b | 10     | false  | 13       | 1.237     | 0.271 | 1.237   | -0.027  | -1.264   | 0.257  |
| qwen3_5-9b | 10     | true   | 36       | 0.333     | 0.125 | 0.607   | 1.035   | 0.702    | 0.597  |
| template   | 1      | true   | 30       | 0.186     | 0.211 | 0.903   | 1.977   | 1.790    | 0.528  |

plot: results/day7/claim_gap.png
