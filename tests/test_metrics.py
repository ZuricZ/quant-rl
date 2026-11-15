"""Portfolio performance of predictions, checked against hand computation."""

import datetime as dt
import math
import statistics

import polars as pl
import pytest

from quantenv import metrics

DAYS = [dt.date(2001, 1, 2), dt.date(2001, 1, 3), dt.date(2002, 1, 2)]


def _frame(predictions: list[list[float]], targets: list[list[float]]) -> pl.DataFrame:
    n_assets = len(predictions[0])
    return pl.DataFrame(
        {
            "t": [d for d in DAYS[: len(predictions)] for _ in range(n_assets)],
            "a": [i for _ in predictions for i in range(n_assets)],
            "p": [x for row in predictions for x in row],
            "y": [x for row in targets for x in row],
        }
    )


def _performance(frame: pl.DataFrame, cost: float = 0.0) -> metrics.Performance:
    return metrics.performance(
        frame, "p", "y", "t", "a", cost=cost, periods_per_year=252.0
    )


def test_rank_weights_are_dollar_neutral_with_unit_gross_exposure():
    frame = _frame([[1.0, 2.0, 3.0], [3.0, None, 1.0]], [[0.0] * 3, [0.0] * 3])
    w = frame.select(metrics.rank_weight("p", "t")).to_series().to_list()
    assert w == pytest.approx([-0.5, 0.0, 0.5, 0.5, 0.0, -0.5])


def test_net_pnl_pays_cost_on_turnover():
    frame = _frame(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        [[0.01, 0.0, 0.03], [0.02, 0.0, -0.02]],
    )
    perf = _performance(frame, cost=0.001)
    # Weights -0.5, 0, 0.5 then 0.5, 0, -0.5: gross P&L 0.01 then 0.02;
    # turnover 1 (from flat) then 2.
    net = [0.01 - 0.001, 0.02 - 0.002]
    expected = statistics.mean(net) / statistics.stdev(net) * math.sqrt(252)
    assert perf.sharpe == pytest.approx(expected)
    gross = [0.01, 0.02]
    expected = statistics.mean(gross) / statistics.stdev(gross) * math.sqrt(252)
    assert perf.gross_sharpe == pytest.approx(expected)
    assert perf.turnover == pytest.approx(1.5)
    assert perf.max_drawdown == 0.0


def test_drawdown_and_per_year_sharpe():
    frame = _frame(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        [[0.0, 0.02], [0.04, 0.0], [0.0, 0.01]],
    )
    perf = _performance(frame)
    # Weights -0.5, 0.5 throughout: P&L 0.01, -0.02, 0.005.
    assert perf.max_drawdown == pytest.approx(0.02)
    first_year = [0.01, -0.02]
    expected = statistics.mean(first_year) / statistics.stdev(first_year)
    # The second year has one period, so no Sharpe; it is left out.
    assert perf.year_sharpe_min == pytest.approx(expected * math.sqrt(252))
    assert perf.year_sharpe_min == perf.year_sharpe_median == perf.year_sharpe_max


def test_constant_predictions_hold_no_position():
    frame = _frame([[1.0, 1.0], [1.0, 1.0]], [[0.01, -0.01], [0.02, 0.0]])
    perf = _performance(frame, cost=0.01)
    assert perf.sharpe == perf.gross_sharpe == perf.turnover == 0.0
    assert perf.ic == 0.0


def test_flipping_the_sign_flips_gross_sharpe():
    targets = [[0.01, 0.0, 0.03], [0.02, 0.0, -0.02], [0.0, 0.01, 0.0]]
    up = _frame([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], targets)
    down = up.with_columns(-pl.col("p"))
    assert _performance(down).gross_sharpe == pytest.approx(
        -_performance(up).gross_sharpe
    )
