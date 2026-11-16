"""The tests the build spec asks for."""

import pytest

from quantenv import baselines, checks, metrics, walkforward
from quantenv.alpha import compile_alpha
from quantenv.panel import between
from quantenv.sources import synthetic

WF = walkforward.WalkForward()
TOL = 0.01  # About 4 standard errors of a 750-period mean IC on 200 assets.


def _local(code: str) -> walkforward.Local:
    return walkforward.Local(compile_alpha(code))


@pytest.fixture(scope="module")
def data():
    # PLAIN turns off the market's realism (factors, persistence, regimes,
    # fat tails, interaction), so planted ICs show up exactly. The realistic
    # default market has its own tests in test_generator.py.
    return synthetic.generate(seed=1, config=synthetic.PLAIN)


def _ic(code: str, data) -> float:
    panel = data.panel
    s = panel.schema
    run = walkforward.run(_local(code), panel, WF)
    frame = between(panel.frame, s.timestamp, run.start).with_columns(
        run.predictions.get_column(walkforward.PREDICTION)
    )
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
    assert checks.column_dependence(_local(code), data.panel, [leaky], WF, seed=0)


def test_signal_ignoring_leak_is_not_flagged(data):
    real = data.truth.real[0]
    code = (
        "import random\n"
        "import numpy as np\n"
        "class Jitter(Alpha):\n"
        "    def features(self, df):\n"
        "        noise = np.random.normal(size=df.height) + random.random()\n"
        "        return df.with_columns(jitter=pl.Series(noise))\n"
        "    def fit(self, train):\n"
        "        self.scale = 0.1 + 0.01 * np.random.random()\n"
        "    def predict(self, df):\n"
        f"        return pl.col({real!r}) + self.scale * pl.col('jitter')\n"
    )
    runner = _local(code)
    assert not checks.column_dependence(
        runner, data.panel, [data.truth.leaky], WF, seed=0
    )
    assert not checks.truncation(runner, data.panel, WF, seed=0)


def test_reading_future_rows_is_caught_and_past_rows_are_not(data):
    template = (
        "class Lagged(Alpha):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(x=pl.col('ret').shift({k}).over('asset'))\n"
        "    def predict(self, df):\n"
        "        return pl.col('x')\n"
    )
    future = _local(template.format(k=-1))
    past = _local(template.format(k=1))
    assert checks.truncation(future, data.panel, WF, seed=0)
    assert not checks.truncation(past, data.panel, WF, seed=0)


def test_fit_never_sees_unrealised_targets(data):
    code = (
        "class Probe(Alpha):\n"
        "    def fit(self, train):\n"
        "        self.last = train['timestamp'].max()\n"
        "        assert train['fwd_return'].null_count() == 0\n"
        "    def predict(self, df):\n"
        "        assert df['timestamp'].min() > self.last\n"
        "        return pl.col('ret')\n"
    )
    walkforward.run(_local(code), data.panel, WF)


def test_fitted_linear_model_recovers_oracle_ic(data):
    honest = [*data.truth.real, *data.truth.noise]
    code = (
        "import numpy as np\n"
        "class Ols(Alpha):\n"
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


def _screening_code(threshold: float, features: list[str]) -> str:
    """Keeps features whose training IC is above `threshold` but below 0.1,
    dropping implausibly strong ones, and weights them by that IC."""
    return (
        "class Screen(Alpha):\n"
        f"    names = {features!r}\n"
        "    def fit(self, train):\n"
        "        ics = (train.group_by('timestamp')\n"
        "               .agg(pl.corr(c, 'fwd_return', method='spearman') for c in self.names)\n"
        "               .select(pl.col(self.names).fill_nan(0.0).mean()).row(0, named=True))\n"
        f"        self.w = {{c: v for c, v in ics.items() if {threshold!r} <= abs(v) < 0.1}}\n"
        "    def predict(self, df):\n"
        "        if not self.w:\n"
        "            return pl.lit(0.0)\n"
        "        return pl.sum_horizontal(w * pl.col(c) for c, w in self.w.items())\n"
    )


LEAKY_SEEDS = [s for s in range(8) if synthetic.ground_truth(s).leaky][:2]


@pytest.mark.parametrize("seed", LEAKY_SEEDS)
def test_screening_alpha_is_not_flagged_and_leaky_baseline_is(seed):
    # Screening is data-driven: the base run drops the leak as implausibly
    # strong, but once it is shuffled it scores like noise and may pass the
    # screen. That changes the predictions without using the leak.
    syn = synthetic.generate(seed)
    leaky = syn.truth.leaky
    features = list(syn.panel.schema.features)
    for threshold in (0.0, 0.002, 0.005, 0.01):
        runner = _local(_screening_code(threshold, features))
        assert not checks.column_dependence(runner, syn.panel, [leaky], WF, seed=1), (
            threshold
        )
    leak = _local(baselines.column_code(leaky))
    assert checks.column_dependence(leak, syn.panel, [leaky], WF, seed=1)


def test_alpha_gets_a_seeded_generator_and_is_not_flagged(data):
    # Alphas should draw randomness from self.rng, seeded per step by the
    # environment, rather than from global RNGs.
    real = data.truth.real[0]
    code = (
        "class Noisy(Alpha):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(jitter=pl.Series(self.rng.normal(size=df.height)))\n"
        "    def fit(self, train):\n"
        "        self.scale = 0.1 + 0.01 * self.rng.random()\n"
        "    def predict(self, df):\n"
        f"        noise = pl.Series(self.rng.normal(size=df.height))\n"
        f"        return pl.col({real!r}) + self.scale * (pl.col('jitter') + noise)\n"
    )
    runner = _local(code)
    first = walkforward.run(runner, data.panel, WF).predictions
    assert walkforward.run(runner, data.panel, WF).predictions.equals(first)
    assert not checks.column_dependence(
        runner, data.panel, [data.truth.leaky], WF, seed=0
    )
    assert not checks.truncation(runner, data.panel, WF, seed=0)
