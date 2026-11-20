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


def _performance(
    frame: pl.DataFrame, cost: float = 0.0, periods_per_year: float = 252.0
) -> metrics.Performance:
    return metrics.performance(
        frame, "p", "y", "t", "a", cost=cost, periods_per_year=periods_per_year
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
    # With 2 periods a year, 3 periods make one block of all 3: no stub.
    perf = _performance(frame, periods_per_year=2.0)
    # Weights -0.5, 0.5 throughout: P&L 0.01, -0.02, 0.005.
    assert perf.max_drawdown == pytest.approx(0.02)
    pnl = [0.01, -0.02, 0.005]
    expected = statistics.mean(pnl) / statistics.stdev(pnl) * math.sqrt(2)
    assert perf.years == 1
    assert perf.year_sharpe_min == pytest.approx(expected)
    assert perf.year_sharpe_min == perf.year_sharpe_median == perf.year_sharpe_max


def test_per_year_sharpe_uses_consecutive_one_year_blocks():
    days = [dt.date(2001, 1, 2) + dt.timedelta(days=i) for i in range(5)]
    targets = [[0.0, 0.02], [0.04, 0.0], [0.0, 0.03], [0.01, 0.0], [0.0, 0.01]]
    frame = pl.DataFrame(
        {
            "t": [d for d in days for _ in range(2)],
            "a": [0, 1] * 5,
            "p": [1.0, 2.0] * 5,
            "y": [x for row in targets for x in row],
        }
    )
    # 5 periods at 2 a year: blocks of 3 then 2 (i * 2 // 5), whatever the
    # calendar dates.
    perf = _performance(frame, periods_per_year=2.0)
    pnl = [0.01, -0.02, 0.015, -0.005, 0.005]
    sharpes = [
        statistics.mean(b) / statistics.stdev(b) * math.sqrt(2)
        for b in (pnl[:3], pnl[3:])
    ]
    assert perf.years == 2
    assert perf.year_sharpe_min == pytest.approx(min(sharpes))
    assert perf.year_sharpe_max == pytest.approx(max(sharpes))


def test_less_than_a_year_gives_no_per_year_sharpe():
    frame = _frame(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        [[0.0, 0.02], [0.04, 0.0], [0.0, 0.01]],
    )
    perf = _performance(frame)  # 3 periods at 252 a year.
    assert perf.years == 0
    assert math.isnan(perf.year_sharpe_min) and math.isnan(perf.year_sharpe_max)


def test_standard_errors_of_ic_and_sharpe():
    frame = _frame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        [[0.01, 0.0, 0.03], [0.0, 0.03, 0.01], [0.02, 0.0, -0.02]],
    )
    perf = _performance(frame)
    assert perf.periods == 3
    ics = [0.5, 0.5, 1.0]
    assert perf.ic == pytest.approx(statistics.mean(ics))
    assert perf.ic_se == pytest.approx(statistics.stdev(ics) / math.sqrt(3))
    per_period = perf.sharpe / math.sqrt(252)
    expected = math.sqrt(252 * (1 + per_period**2 / 2) / 3)
    assert perf.sharpe_se == pytest.approx(expected)


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


def _pnl(values: list[float]) -> float:
    return pl.select(metrics.robust_sharpe(pl.lit(pl.Series(values)), 252.0)).item()


def test_robust_sharpe_matches_plain_sharpe_without_outliers():
    values = pl.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005, 0.012])
    plain = pl.select(metrics.sharpe(pl.lit(values), 252.0)).item()
    assert _pnl(values.to_list()) == pytest.approx(plain)


def test_robust_sharpe_bounds_one_extreme_period():
    calm = [0.001 * ((i * 7) % 11 - 5) for i in range(500)]

    def plain(values: list[float]) -> float:
        return pl.select(metrics.sharpe(pl.lit(pl.Series(values)), 252.0)).item()

    # One windfall among 500 typical periods lifts the plain Sharpe from
    # about 0 to 0.7; the robust Sharpe moves a third as much...
    assert plain([*calm, 5.0]) - plain(calm) > 0.6
    assert _pnl([*calm, 5.0]) - _pnl(calm) < 0.35 * (plain([*calm, 5.0]) - plain(calm))
    # ...and a ten times larger windfall moves it no further.
    assert _pnl([*calm, 50.0]) == pytest.approx(_pnl([*calm, 5.0]))


def test_robust_sharpe_handles_constant_and_sparse_pnl():
    assert _pnl([0.0] * 10) == 0.0
    # Mostly flat: the MAD is 0, so the scale falls back to the std.
    sparse = [0.0] * 8 + [0.01, 0.02]
    plain = pl.select(metrics.sharpe(pl.lit(pl.Series(sparse)), 252.0)).item()
    assert _pnl(sparse) == pytest.approx(plain)
