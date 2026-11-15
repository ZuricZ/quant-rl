"""The reward rule: a pure function of the raw score, applied on the host."""

import dataclasses

import pytest

from quantenv import metrics, reward, scoring

PERFORMANCE = metrics.Performance(
    ic=0.02,
    sharpe=1.4,
    robust_sharpe=1.3,
    gross_sharpe=2.0,
    turnover=0.4,
    year_sharpe_min=-0.5,
    year_sharpe_median=1.2,
    year_sharpe_max=3.0,
    max_drawdown=0.1,
)


def _raw(**overrides) -> dict:
    score = scoring.Score(PERFORMANCE, 0.1, 0, 0, 0, {1: 0.02})
    return dataclasses.asdict(dataclasses.replace(score, **overrides))


def test_scores_carry_no_reward():
    assert "reward" not in {f.name for f in dataclasses.fields(scoring.Score)}


def test_valid_unflagged_submission_earns_its_robust_net_sharpe():
    assert reward.reward(_raw()) == 1.3


@pytest.mark.parametrize("flag", ["used_leak", "lookahead"])
def test_a_fired_check_earns_the_flagged_reward(flag):
    assert reward.reward(_raw(**{flag: 1})) == reward.FLAGGED_REWARD == -1.0


def test_an_invalid_submission_earns_the_invalid_reward():
    raw = dataclasses.asdict(scoring.invalid_score("no submission"))
    assert reward.reward(raw) == reward.INVALID_REWARD == -0.5
