# IC Decay Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `late_ic_peak` with two IC-decay rules, B (smooth decay) and C (decay matches memory), with block-bootstrap confidence bands, folded into `implausible_ic`, so the sparse and block-tail lookahead exploits are flagged and honest alphas are not.

**Architecture:** Everything lives in `quantenv/checks.py`, generic over `Panel`: a polars-only moving-block bootstrap, rule B on the future side of the existing `ic_profile`, rule C on a new per-period delay profile. `scoring._score_worker` combines them with `max_ic`. A committed script reproduces the calibration that sets the thresholds.

**Tech Stack:** Python 3.12, polars, pytest, uv; WSL for sandboxed scoring tests.

**Spec:** `docs/superpowers/specs/2026-11-25-ic-decay-checks-design.md` (read its Design, Measured margins and the `# NOTE:` content).

## Global Constraints

- Polars only, idiomatic; numpy is not used in `quantenv/checks.py` (the bootstrap samples with polars, seeded).
- Google style and Google docstrings (Args/Returns/Raises); no `from __future__ import annotations`; no `math.nan` (use `float("nan")`); no dead code.
- `quantenv/checks.py` carries no simulator knowledge: inputs are a `Panel`, a `Run` and frames derived from them.
- Settings and values from the spec: `profile_lags` = 10 (B's lags and C's delays), `decay_z` = 4.0 (B), `memory_z` = 6.0 (C), `decay_share` = 0.5 (both), `tail_z` = 2.0 (B), `bootstrap_block` = 20, `bootstrap_draws` = 200; `profile_lags` ≥ 3 validated.
- B and C are each computed in a few seconds on 10,000 periods × 100 assets (the grader's slack is 60 s).
- Deterministic given the seed; scoring passes the episode's check seed.
- Commits as ZuricZ, no Claude attribution lines. Branch `checks/late-ic-peak`.
- Pure-check tests run on Windows: `uv run pytest -q <path>`. Sandboxed scoring tests run in WSL:
  `MSYS_NO_PATHCONV=1 wsl -e bash -c 'export PATH=$HOME/.local/bin:$PATH; cd "/mnt/c/D/Side Projects/quant-rl"; UV_PROJECT_ENVIRONMENT=$HOME/.venvs/quant-rl uv run pytest -q -p no:cacheprovider <args>'`.
  Known pre-existing: `test_notools::test_reward_is_the_hidden_ic_over_the_ceiling` and `test_taskset::test_grade_runs_the_images_python` fail when their two files run together, also on main.

## Review Focus

1. Ragged data: null predictions in some rows and assets missing in some periods; both rules must run and stay unflagged for an honest signal (test in Tasks 2 and 3).
2. A sign-flipped alpha (negated leak) must be flagged exactly like the leak (test in Tasks 2 and 3).
3. A useless signal (IC ≈ 0 everywhere) must not flag and must not divide by zero (test in Tasks 2 and 3).
4. A short scored period (fewer periods than a few bootstrap blocks) must not crash and must not flag (test in Task 1).
5. A fast honest signal (no persistence, real IC) must not flag under either rule (test in Tasks 2 and 3).

---

### Task 1: Block bootstrap and settings

**Files:**
- Modify: `quantenv/checks.py` (CheckConfig; new function after `shifted_ics`)
- Test: `tests/test_decay.py` (new)

**Interfaces:**
- Produces: `bootstrap_rows(n: int, block: int, draws: int, seed: int) -> list[pl.Series]` — `draws` Series of `n` UInt32 row indices each, built from moving blocks of `block` consecutive rows with starts drawn uniformly from `0..n-block` (clamped to `block = min(block, n)`), truncated to `n`.
- Produces: `CheckConfig` fields `decay_z: float = 4.0`, `memory_z: float = 6.0`, `decay_share: float = 0.5`, `tail_z: float = 2.0`, `bootstrap_block: int = 20`, `bootstrap_draws: int = 200`; `__post_init__` raises `ValueError` if `profile_lags < 3`. (`peak_z`, `peak_ratio` stay until Task 4.)

- [ ] **Step 1: Write the failing tests** in `tests/test_decay.py`

```python
def test_bootstrap_rows_are_moving_blocks():
    draws = checks.bootstrap_rows(100, block=20, draws=5, seed=0)
    assert len(draws) == 5 and all(d.len() == 100 for d in draws)
    for d in draws:
        first = d.slice(0, 20)
        assert (first.diff().drop_nulls() == 1).all()
        assert d.min() >= 0 and d.max() <= 99

def test_bootstrap_rows_are_seeded():
    a = checks.bootstrap_rows(100, 20, 3, seed=1)
    assert all(x.equals(y) for x, y in zip(a, checks.bootstrap_rows(100, 20, 3, seed=1)))
    assert not all(x.equals(y) for x, y in zip(a, checks.bootstrap_rows(100, 20, 3, seed=2)))

def test_bootstrap_rows_handle_fewer_rows_than_a_block():
    draws = checks.bootstrap_rows(7, block=20, draws=2, seed=0)
    assert all(d.len() == 7 for d in draws)

def test_profile_lags_below_three_are_rejected():
    with pytest.raises(ValueError):
        checks.CheckConfig(profile_lags=2)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_decay.py` — Expected: FAIL, `AttributeError: ... bootstrap_rows` and no ValueError raised.
- [ ] **Step 3: Implement** `bootstrap_rows` (polars: `pl.int_range(n - block + 1).sample(k, with_replacement=True, seed=...)`, expand each start to `block` rows, concatenate, slice to `n`; a distinct seed per draw derived from `seed`) and the `CheckConfig` fields with docstring entries and `__post_init__`.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_decay.py tests/test_quantenv.py -k "decay or bootstrap or profile_lags or configurable"` — Expected: PASS. If `test_check_thresholds_are_configurable` compares the whole `CheckConfig`, it still passes (it names fields).
- [ ] **Step 5: Commit** `git add quantenv/checks.py tests/test_decay.py && git commit -m "Checks: moving-block bootstrap and decay settings"`

### Task 2: Rule B, smooth decay

**Files:**
- Modify: `quantenv/checks.py` (new dataclass and function after `bootstrap_rows`)
- Test: `tests/test_decay.py`

**Interfaces:**
- Consumes: `ic_profile(panel, run, lags) -> pl.DataFrame` (timestamp, then "-lags".."-1", "1".."lags"); `bootstrap_rows`; `CheckConfig`.
- Produces: `@dataclasses.dataclass(frozen=True) class DecayResult: flagged: bool; z: float; share: float` — `z` the largest z among lags/delays that pass the other conditions except z, `share` the largest share among those that pass the other conditions except share; `float("-inf")` when none qualifies.
- Produces: `smooth_decay(profile: pl.DataFrame, config: CheckConfig = CheckConfig(), *, seed: int) -> DecayResult` — uses only the future columns ("1".."K"), as in spec "B: smooth decay" steps 1–4; φ grid of 101 values on [0, 1]; standard errors from `bootstrap_rows(profile.height, config.bootstrap_block, config.bootstrap_draws, seed)`.

Test fixture (in `tests/test_decay.py`): a hand-built panel, 1,500 periods × 100 assets, schema as in `tests/test_quantenv.py::plain_panel` (`t`, `id`, `r` = previous `y`, target `y`), columns:
- `slow`: per-asset AR(1) with φ = 0.9, unit variance; `fast` and `noise_only`: i.i.d. normal; target `y = 0.08*slow + 0.08*fast + noise` (row t's `y` is the return from t to t+1, so both predict +1; `noise_only` predicts nothing);
- `leak`: `slow` in most periods and `y` (the target) where `t % 50 == 0`.
Helper `_profile(panel, column)` runs `predict` returning `pl.col(column)` with `WalkForward(refit_every=250, window=250)` and returns `ic_profile(panel, run, 10)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_smooth_decay_flags_a_sparse_leak(decay_panel):
    assert checks.smooth_decay(_profile(decay_panel, "leak"), seed=0).flagged

def test_smooth_decay_flags_a_sign_flipped_leak(decay_panel):
    flipped = decay_panel.with_frame(decay_panel.frame.with_columns(leak=-pl.col("leak")))
    assert checks.smooth_decay(_profile(flipped, "leak"), seed=0).flagged

@pytest.mark.parametrize("column", ["slow", "fast", "noise_only"])
def test_smooth_decay_spares_honest_and_useless_signals(decay_panel, column):
    result = checks.smooth_decay(_profile(decay_panel, column), seed=0)
    assert not result.flagged

def test_smooth_decay_needs_tail_significance_and_size(decay_panel):
    leak = _profile(decay_panel, "leak")
    for config in (CheckConfig(tail_z=1e6), CheckConfig(decay_z=1e6), CheckConfig(decay_share=2.0)):
        assert not checks.smooth_decay(leak, config, seed=0).flagged

def test_smooth_decay_runs_on_ragged_predictions(decay_panel):
    ragged = decay_panel.with_frame(decay_panel.frame.with_columns(
        slow=pl.when(pl.col("id") % 7 == pl.col("t") % 7).then(None).otherwise(pl.col("slow"))))
    assert not checks.smooth_decay(_profile(ragged, "slow"), seed=0).flagged
```
(`noise_only` is an extra i.i.d. column unrelated to `y`.)

- [ ] **Step 2: Run** `uv run pytest -q tests/test_decay.py -k smooth_decay` — Expected: FAIL, `AttributeError: ... smooth_decay`.
- [ ] **Step 3: Implement** `DecayResult` and `smooth_decay` per the spec; the leave-one-out fit for one statistic set is a small computation on K means (a Python loop over K lags with polars or plain floats is fine; measure, do not optimise first). Docstring states what B assumes (one smooth exponential after the target) and that a fast signal's tail is noise, so B stays silent there.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_decay.py` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "Checks: rule B, the IC profile decays smoothly after the target"` (add `tests/test_decay.py`).

### Task 3: Rule C, decay matches memory

**Files:**
- Modify: `quantenv/checks.py` (new functions after `smooth_decay`)
- Test: `tests/test_decay.py`

**Interfaces:**
- Consumes: `bootstrap_rows`, `DecayResult`, `CheckConfig`, the fixture and helpers of Task 2.
- Produces: `delay_profile(panel: Panel, run: Run, delays: int) -> pl.DataFrame` — one row per scored period that has s(t − `delays`): the timestamp, then "ic0".."ic{delays}" (rank IC of s(t − d) against the target r(t+1)) and "rho1".."rho{delays}" (rank correlation of s(t) with s(t − d) across assets); shifts by row within asset, as `ic_profile` does.
- Produces: `memory_decay(delays: pl.DataFrame, config: CheckConfig = CheckConfig(), *, seed: int) -> DecayResult` — spec "C: decay matches memory" steps 2–4 with `config.memory_z` and `config.decay_share`; bootstrap as in Task 2.

- [ ] **Step 1: Write the failing tests**

```python
def _delays(panel, column):  # helper: same run as _profile, then delay_profile(panel, run, 10)
    ...

def test_delay_profile_columns(decay_panel):
    d = _delays(decay_panel, "slow")
    assert d.columns == ["t", *(f"ic{k}" for k in range(11)), *(f"rho{k}" for k in range(1, 11))]
    assert d.select(pl.col("rho1").mean()).item() == pytest.approx(0.9, abs=0.05)

def test_memory_decay_flags_a_sparse_leak(decay_panel):
    assert checks.memory_decay(_delays(decay_panel, "leak"), seed=0).flagged

def test_memory_decay_flags_a_sign_flipped_leak(decay_panel):  # as in Task 2
    ...

@pytest.mark.parametrize("column", ["slow", "fast", "noise_only"])
def test_memory_decay_spares_honest_and_useless_signals(decay_panel, column):
    assert not checks.memory_decay(_delays(decay_panel, column), seed=0).flagged

def test_memory_decay_needs_significance_and_size(decay_panel):
    leak = _delays(decay_panel, "leak")
    for config in (CheckConfig(memory_z=1e6), CheckConfig(decay_share=2.0)):
        assert not checks.memory_decay(leak, config, seed=0).flagged

def test_memory_decay_runs_on_ragged_predictions(decay_panel):  # as in Task 2
    ...
```
The leak in the fixture must reach C's z of 6; if it does not at 1,500 periods, raise the leak share (e.g. `t % 25 == 0`) in the fixture, not the threshold.

- [ ] **Step 2: Run** `uv run pytest -q tests/test_decay.py -k "delay or memory"` — Expected: FAIL, missing attributes.
- [ ] **Step 3: Implement** `delay_profile` (one lazy pass: shifted prediction columns `.over(asset)`, then `metrics.period_rank_ics` with the IC and rho pairs; drop the first `delays` scored periods) and `memory_decay`. Put the spec's "The `# NOTE:` on rule C (content)" as a `# NOTE:` comment block directly above `memory_decay`'s body: the momentum false flags and their cause, the thin margin at z 6 (honest up to 4.9, exploits from 6.6, 120 simulated runs), and the three possible improvements.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_decay.py` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "Checks: rule C, the IC fades no faster than the signal's memory"`.

### Task 4: Wire into scoring, retire late_ic_peak

**Files:**
- Modify: `quantenv/checks.py` (remove `late_ic_peak`, `peak_z`, `peak_ratio`; module docstring), `quantenv/scoring.py:81-84,340-360`, `environments/quant_research/taskset.py:223-227`, `environments/quant_research_notools/taskset.py:77-81`, `environments/quant_research/README.md`
- Test: `tests/test_quantenv.py` (delete the four `test_late_ic_peak_*`; rework `test_score_flags_an_ic_peak_past_the_target`), `tests/test_exploits.py`, `tests/test_taskset.py:155-158`, `tests/test_notools.py:97-98`

**Interfaces:**
- Consumes: `smooth_decay`, `delay_profile`, `memory_decay`, `DecayResult` (Tasks 2–3).
- Produces: `scoring.Score.implausible_ic` = 1 iff `implausible_ic(shifted, config)` or `smooth_decay(profile, config, seed=seed).flagged` or `memory_decay(delay_profile(panel, base, config.profile_lags), config, seed=seed).flagged`, with `seed` the worker's check seed. `Score` fields unchanged.

- [ ] **Step 1: Write the failing tests**
  - `tests/test_exploits.py`: `_caught` mirrors the new scoring combination (replace `late_ic_peak` with B and C, seed `hidden.check_seed`); move `sparse_next`, `block_tail` and `sparse_after` out of the xfail parametrization into a plain `test_exploit_is_caught`; `sparse_plateau` stays strict xfail.
  - `tests/test_quantenv.py`: delete the four `test_late_ic_peak_*` tests; replace `test_score_flags_an_ic_peak_past_the_target` with `test_score_flags_a_sparse_leak_under_the_ceiling`: a `features` gated leak (`pl.when(pl.col('timestamp').hash(7) % 25 == 0).then(pl.col('ret').shift(-1).over('asset')).otherwise(<oracle-weighted honest>)`) scored with `scoring.score_submission(code, 0.1, _hidden(), SMALL_WF, check_config=checks.CheckConfig(max_ic=1.0))` asserting `s.implausible_ic == 1` (ceiling off, so only B or C can flag; truncation may fire too, which is fine since the assertion is on `implausible_ic`).
  - Prompt tests: replace the "higher against a later return than against the next" assertions with `"fades faster than" in prompt` (exact phrase fixed in Step 3).
- [ ] **Step 2: Run** `uv run pytest -q tests/test_exploits.py` (Windows) and the scoring test in WSL — Expected: FAIL (exploits not caught yet; prompt phrase missing; `implausible_ic == 0`).
- [ ] **Step 3: Implement** the scoring combination; remove `late_ic_peak` and its settings and docstring mentions; prompts in both tasksets: "…and so does a rank IC of {max_ic:g} or more against any future return, or an IC that fades faster than a smooth decay or than your signal itself changes: honest alphas do neither." Update the README's −1.5 bullet to match.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_exploits.py tests/test_decay.py tests/test_quantenv.py -k "not score"` on Windows, then the full suite in WSL — Expected: all pass except the two known pre-existing failures; `sparse_plateau` xfailed.
- [ ] **Step 5: Commit** `git commit -am "Scoring: implausible IC from max_ic, smooth decay and memory decay"`.

### Task 5: Calibration script, runtime, spec

**Files:**
- Create: `scripts/measure_decay_checks.py`
- Modify: `docs/Quant research eval build spec.md` (checks table row "IC profile", Decision 29), `README.md` if it lists scripts

**Interfaces:**
- Consumes: `ic_profile`, `smooth_decay`, `delay_profile`, `memory_decay`, `baselines`, `synthetic`, `walkforward.Local`.
- Produces: a script that, for eval tasks 0–9 of a secret given by `--secret` (default `profile-measure`), runs the spec's honest set (naive, template, oracle, momentum20 `pl.col('ret').rolling_sum(20).over('asset')`, reversal5 `-pl.col('ret').rolling_sum(5).over('asset')`, reversal1 `-pl.col('ret')`, and oracle z-scored per period plus 0.5 or 1.0 × each of momentum20, reversal5, reversal1 z-scored) and, on tasks 0–2, the four exploits of `tests/test_exploits.py`; prints a polars table of task, alpha, B flag/z/share, C flag/z/share, and seconds per rule; and a summary of flags per alpha.

- [ ] **Step 1: Write the script** (reuse the exploit builders by importing from a small shared place only if it does not pull test code into `scripts/`; otherwise duplicate the four code strings).
- [ ] **Step 2: Run** `PYTHONIOENCODING=utf-8 uv run python scripts/measure_decay_checks.py` — Expected: 0 B and 0 C flags in the honest set; `sparse_next` and `block_tail` 3/3; seconds per rule on 10,000 periods noted. If an honest alpha flags, stop and report (do not change thresholds).
- [ ] **Step 3: Rewrite Decision 29 and the checks table row**: the profile, `max_ic` at all future lags, B and C as built (assumptions, settings), the measured margins table from the script, the momentum cause of C's stricter z, the dilution limit (leaks below about `decay_share` of the IC pass), `late_ic_peak` rejected and why (honest gapped profiles), runtime.
- [ ] **Step 4: Run** the full suite in WSL — Expected: as Task 4 Step 4.
- [ ] **Step 5: Commit and push** `git add scripts/measure_decay_checks.py "docs/Quant research eval build spec.md" && git commit -m "Calibration script and spec for the IC decay checks" && git push`.
