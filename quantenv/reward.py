"""The reward rule: maps a submission's raw score to a reward.

Kept apart from scoring on purpose. The grader runs inside the task image
and returns only the raw outcome (`scoring.Score`, as a JSON-able dict); the
host applies this rule. Changing the rule therefore never needs the image
rebuilt. It imports nothing from verifiers, so scripts share it.
"""

import math
from collections.abc import Mapping
from typing import Any

# TODO: These values are a first choice and we may have to come back to
# them once real training runs show how they behave: the scale S0, the
# penalty gaps (with group-mean-only advantages a flagged rollout lifts every
# other rollout in its group), and whether the claim gap should count.
#
# Honest submissions earn tanh(S / SHARPE_SCALE) for robust net Sharpe S, so
# always in (-1, 1); SHARPE_SCALE = 3 keeps the oracle's range (1 to 3) on the
# near-linear part. The penalties sit at or below that range, so honest work
# beats failing to submit, which beats breaking the rules, whatever the Sharpe.
SHARPE_SCALE = 3.0
INVALID_REWARD = -1.0
FLAGGED_REWARD = -1.5


def reward(score: Mapping[str, Any]) -> float:
    """The reward of one scored submission.

    Args:
        score: A `scoring.Score` as a dict, as the grader writes it
            (`dataclasses.asdict`).

    Returns:
        `INVALID_REWARD` if the submission was invalid, `FLAGGED_REWARD` if a
        lookahead check fired, else tanh(S / `SHARPE_SCALE`) of the robust
        annualised net Sharpe S on the scored periods (see
        `metrics.robust_sharpe`).
    """
    if score["invalid"]:
        return INVALID_REWARD
    if score["used_leak"] or score["lookahead"]:
        return FLAGGED_REWARD
    return math.tanh(score["performance"]["robust_sharpe"] / SHARPE_SCALE)
