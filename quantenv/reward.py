"""The reward rule: maps a submission's raw score to a reward.

Kept apart from scoring on purpose. The grader runs inside the task image
and returns only the raw outcome (`scoring.Score`, as a JSON-able dict); the
host applies this rule. Changing the rule therefore never needs the image
rebuilt. It imports nothing from verifiers, so scripts share it.
"""

from collections.abc import Mapping
from typing import Any

# Rewards outside the Sharpe scale: worse than any plausible honest alpha, so
# neither breaking the rules nor failing to submit pays.
INVALID_REWARD = -0.5
FLAGGED_REWARD = -1.0


def reward(score: Mapping[str, Any]) -> float:
    """The reward of one scored submission.

    Args:
        score: A `scoring.Score` as a dict, as the grader writes it
            (`dataclasses.asdict`).

    Returns:
        `INVALID_REWARD` if the submission was invalid, `FLAGGED_REWARD` if a
        lookahead check fired, else the robust annualised net Sharpe on the
        scored periods (see `metrics.robust_sharpe`).
    """
    if score["invalid"]:
        return INVALID_REWARD
    if score["used_leak"] or score["lookahead"]:
        return FLAGGED_REWARD
    # TODO: The reward is unbounded. Consider squashing it, e.g.
    # tanh(robust_sharpe / S0) with S0 = 3, once real runs show its range.
    return score["performance"]["robust_sharpe"]
