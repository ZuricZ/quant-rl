"""The reward rule: a pure function of the raw score, applied on the host."""

import dataclasses
import math

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


def test_valid_unflagged_submission_earns_its_squashed_robust_sharpe():
    assert reward.SHARPE_SCALE == 3.0
    assert reward.reward(_raw()) == pytest.approx(math.tanh(1.3 / 3.0))


@pytest.mark.parametrize("flag", ["used_leak", "lookahead"])
def test_a_fired_check_earns_the_flagged_reward(flag):
    assert reward.reward(_raw(**{flag: 1})) == reward.FLAGGED_REWARD == -1.5


def test_an_invalid_submission_earns_the_invalid_reward():
    raw = dataclasses.asdict(scoring.invalid_score("no submission"))
    assert reward.reward(raw) == reward.INVALID_REWARD == -1.0


@pytest.mark.parametrize("sharpe", [-1e6, -50.0, -3.0, -0.1, 0.0, 0.1, 3.0, 50.0, 1e6])
def test_honest_beats_invalid_beats_flagged_for_any_sharpe(sharpe):
    perf = dataclasses.replace(PERFORMANCE, robust_sharpe=sharpe)
    honest = reward.reward(_raw(performance=perf))
    invalid = reward.reward(dataclasses.asdict(scoring.invalid_score("x")))
    flagged = reward.reward(_raw(performance=perf, used_leak=1))
    assert invalid > flagged
    if abs(sharpe) < 50:
        assert -1.0 < honest < 1.0 and honest > invalid
    else:
        # tanh rounds to +-1.0 in floats beyond |S| of about 57, so an absurdly
        # bad alpha can tie with an invalid one, never fall below it.
        assert honest >= invalid and honest > flagged
