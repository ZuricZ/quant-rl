| Agent              | Tasks | Episodes | Leak rate | Lookahead rate | Implausible IC rate | Invalid rate | Reward | Net Sharpe | Gross Sharpe | Turnover | IC    | Claim gap | Backtests used |
|--------------------|-------|----------|-----------|----------------|---------------------|--------------|--------|------------|--------------|----------|-------|-----------|----------------|
| naive              | 30    | 30       | 0.000     | 0.000          | 0.000               | 0.000        | 0.400  | 1.293      | 2.136        | 0.431    | 0.012 | 0.224     | 1.000          |
| oracle             | 30    | 30       | 0.000     | 0.000          | 0.000               | 0.000        | 0.703  | 2.669      | 3.553        | 0.454    | 0.020 | 0.120     | 1.000          |
| qwen3_5-9b         | 30    | 90       | 0.000     | 0.044          | 0.000               | 0.467        | -0.559 | -0.037     | 1.142        | 0.640    | 0.007 | 0.954     | 7.556          |
| qwen3_5-9b-budget0 | 30    | 90       | 0.000     | 0.011          | 0.011               | 0.722        | -0.855 | -1.160     | 0.153        | 1.110    | 0.002 | 1.808     | 0.000          |
| qwen3_5-9b-budget1 | 30    | 90       | 0.000     | 0.000          | 0.000               | 0.811        | -0.882 | -1.242     | 0.215        | 1.115    | 0.001 | 1.763     | 0.644          |
| qwen3_5-9b-budget3 | 30    | 90       | 0.000     | 0.022          | 0.011               | 0.789        | -0.857 | 1.594      | 2.617        | 0.796    | 0.054 | -0.894    | 2.033          |
| template           | 30    | 30       | 0.000     | 0.000          | 0.000               | 0.000        | 0.524  | 1.790      | 2.631        | 0.428    | 0.015 | 0.186     | 1.000          |

How episodes ended:
| agent              | stop            | len |
|--------------------|-----------------|-----|
| naive              | baseline        | 30  |
| oracle             | baseline        | 30  |
| qwen3_5-9b         | agent_completed | 21  |
| qwen3_5-9b         | submitted       | 69  |
| qwen3_5-9b-budget0 | agent_completed | 21  |
| qwen3_5-9b-budget0 | submitted       | 69  |
| qwen3_5-9b-budget1 | agent_completed | 30  |
| qwen3_5-9b-budget1 | submitted       | 60  |
| qwen3_5-9b-budget3 | agent_completed | 28  |
| qwen3_5-9b-budget3 | max_turns       | 1   |
| qwen3_5-9b-budget3 | submitted       | 61  |
| template           | baseline        | 30  |

Claim gap by backtests used (valid, unflagged submissions):
| model      | backtests_used | episodes | claim_gap | se    | claimed | realised |
|------------|----------------|----------|-----------|-------|---------|----------|
| naive      | 1.000          | 30       | 0.224     | 0.168 | 1.517   | 1.293    |
| oracle     | 1.000          | 30       | 0.120     | 0.229 | 2.789   | 2.669    |
| qwen3_5-9b | 0.000          | 25       | 2.062     | 0.138 | 0.648   | -1.414   |
| qwen3_5-9b | 1.000          | 16       | 1.756     | 0.273 | 0.523   | -1.233   |
| qwen3_5-9b | 3.000          | 17       | 1.008     | 0.234 | 0.459   | -0.550   |
| qwen3_5-9b | 10.000         | 44       | 0.957     | 0.188 | 0.715   | -0.242   |
| template   | 1.000          | 30       | 0.186     | 0.211 | 1.977   | 1.790    |
