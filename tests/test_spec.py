"""The tests the build spec asks for."""

import pytest

from quantenv import baselines, checks, metrics, walkforward
from quantenv.signal import compile_signal
from quantenv.sources import synthetic

WF = walkforward.WalkForward()
TOL = 0.01  # About 4 standard errors of a 750-period mean IC on 200 assets.


@pytest.fixture(scope="module")
def data():
    return synthetic.generate(seed=1)


def _ic(code: str, data) -> float:
    panel = data.panel
    s = panel.schema
    predictions = walkforward.run(compile_signal(code), panel, WF)
    frame = panel.frame.tail(predictions.len()).with_columns(predictions)
    return metrics.rank_ic(frame, walkforward.PREDICTION, s.target, s.timestamp)


def test_noise_feature_scores_about_zero(data):
    assert abs(_ic(baselines.column_code(data.truth.noise[0]), data)) < TOL


def test_real_feature_scores_planted_ic(data):
    for name in data.truth.real:
        ic = _ic(baselines.column_code(name), data)
        assert ic == pytest.approx(data.truth.planted_ic[name], abs=TOL)


def test_leaky_feature_scores_high_and_is_caught(data):
    leaky = data.truth.leaky
    assert leaky is not None
    code = baselines.column_code(leaky)
    assert _ic(code, data) == pytest.approx(0.3, abs=0.02)
    assert checks.column_dependence(compile_signal(code), data.panel, [leaky], WF)


def test_signal_ignoring_leak_is_not_flagged(data):
    real = data.truth.real[0]
    code = (
        "import random\n"
        "import numpy as np\n"
        "class Jitter(Signal):\n"
        "    def features(self, df):\n"
        "        noise = np.random.normal(size=df.height) + random.random()\n"
        "        return df.with_columns(jitter=pl.Series(noise))\n"
        "    def fit(self, train):\n"
        "        self.scale = 0.1 + 0.01 * np.random.random()\n"
        "    def predict(self, df):\n"
        f"        return pl.col({real!r}) + self.scale * pl.col('jitter')\n"
    )
    cls = compile_signal(code)
    assert not checks.column_dependence(cls, data.panel, [data.truth.leaky], WF)
    assert not checks.truncation(cls, data.panel, WF)


def test_reading_future_rows_is_caught_and_past_rows_are_not(data):
    template = (
        "class Lagged(Signal):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(x=pl.col('ret').shift({k}).over('asset'))\n"
        "    def predict(self, df):\n"
        "        return pl.col('x')\n"
    )
    future = compile_signal(template.format(k=-1))
    past = compile_signal(template.format(k=1))
    assert checks.truncation(future, data.panel, WF)
    assert not checks.truncation(past, data.panel, WF)


def test_fit_never_sees_unrealised_targets(data):
    code = (
        "class Probe(Signal):\n"
        "    def fit(self, train):\n"
        "        self.last = train['timestamp'].max()\n"
        "        assert train['fwd_return'].null_count() == 0\n"
        "    def predict(self, df):\n"
        "        assert df['timestamp'].min() > self.last\n"
        "        return pl.col('ret')\n"
    )
    walkforward.run(compile_signal(code), data.panel, WF)


def test_fitted_linear_model_recovers_oracle_ic(data):
    honest = [*data.truth.real, *data.truth.noise]
    code = (
        "import numpy as np\n"
        "class Ols(Signal):\n"
        f"    cols = {honest!r}\n"
        "    def fit(self, train):\n"
        "        x = train.select(self.cols).to_numpy()\n"
        "        y = train['fwd_return'].to_numpy()\n"
        "        self.w = np.linalg.lstsq(x, y, rcond=None)[0].tolist()\n"
        "    def predict(self, df):\n"
        "        return pl.sum_horizontal(\n"
        "            w * pl.col(c) for c, w in zip(self.cols, self.w))\n"
    )
    oracle = _ic(baselines.weighted_code(data.truth.weights), data)
    assert _ic(code, data) == pytest.approx(oracle, abs=0.01)


def test_same_seed_same_dataset_and_scores():
    a = synthetic.generate(seed=7)
    b = synthetic.generate(seed=7)
    assert a.panel.frame.equals(b.panel.frame)
    assert a.truth == b.truth
    code = baselines.weighted_code(a.truth.weights)
    assert _ic(code, a) == _ic(code, b)
