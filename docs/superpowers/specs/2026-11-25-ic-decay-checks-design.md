# IC decay checks: design

Date: 25 November 2026. Branch `checks/late-ic-peak` (PR #13). Supersedes the
`late_ic_peak` rule on that branch.

## Intent

Catch lookahead that passes truncation and `max_ic`, using the shape of the IC
profile across a range of lags (the "lag plot"), and keep it sound on real data
later, not only on the simulator.

Why: truncation is exact but compares only the k periods before each of its 8
cuts (for a leak k periods ahead), so a leak used in few periods, or only where
no cut lands, passes. `tests/test_exploits.py` pins four such exploits. Measured
on 3 eval tasks, 20 cut seeds each:

| Exploit | Net Sharpe honest → exploit | IC at +1 | Passes truncation |
| --- | --- | --- | --- |
| `sparse_next`: target in ~2% of periods | 1.44 → 2.56, 3.18 → 3.53, 2.09 → 2.89 | 0.037–0.043 | 80–90% |
| `block_tail`: target in 5 rows of full blocks | 1.44 → 2.51, 3.18 → 3.45, 2.09 → 2.92 | 0.036–0.042 | 100% |
| `sparse_plateau`: mean of next 10 returns, ~2% | 1.44 → 2.01, … | 0.022–0.027 | 10–20% |
| `sparse_after`: return after the target, ~2% | lower than honest | 0.017–0.021 | 60–65% |

All stay under `max_ic` = 0.05. The two that pay most (`sparse_next`,
`block_tail`) show a spike at +1: 0.037 against 0.013 at +2 and 0.014 at −1.

The user's experience on real data: an honest alpha's IC is roughly a smooth
exponential decline from the target onward, IC(k) ≈ IC(+1)·φ^(k−1); blends of
signals look the same, a bit more jagged. The past side (before the target) is
not a yardstick: here it equals +1 for feature alphas and reaches −0.40 for
reversal alphas.

## Goals

- A hard flag, folded into `implausible_ic` (reward −1.5), from two rules:
  - **B, smooth decay:** the future side of the profile is one smooth
    exponential; a lag that stands far above it is flagged.
  - **C, decay matches memory:** the IC fades no faster than the signal itself
    changes; an IC that drops faster than the signal's own autocorrelation
    implies is flagged.
- Catch `sparse_next` and `block_tail` at their measured size.
- Spare every honest alpha: naive, template, oracle, 20-period momentum,
  5-period reversal on 10 eval tasks, and blends of them.
- A confidence band for both rules from a block bootstrap over periods, so the
  flags are statistical tests, not eyeballed thresholds.

## Non-goals

- Fixing truncation's sampling (re-predicting full-length blocks, more cuts).
  It would close `block_tail` and `sparse_next` exactly; a separate later step.
- Calibrating on real data: there is none in the repo. Settings are
  `CheckConfig` fields so a real-data source can set its own.
- Catching a leak on top of a useless signal. With no measurable tail and no
  memory, the shape says nothing; only `max_ic` applies.
- Catching `sparse_plateau` or `sparse_after` is welcome but not required:
  the first is mostly caught by truncation, the second does not pay.

## Anti-goals

- Flagging an honest alpha. A −1.5 on honest work teaches the wrong thing and
  is worse than missing a diluted leak. Every setting errs lenient.
- Claiming the checks catch diluted leaks in general. A leak smaller than
  about the honest part's share (see `decay_share`) passes by design; the
  write-up must say so.
- Simulator knowledge in `quantenv/checks.py` (project rule): the checks take
  the panel and the run only.

## Constraints

- Project rules (`CLAUDE.md`): polars only and idiomatic, no numpy except
  seeded RNG (the bootstrap draws with polars `sample(..., seed=)`), Google
  docstrings, no `from __future__ import annotations`, no `math.nan`.
- Scoring runs these inside the grader worker, within its 60 s slack on top of
  the pipeline runs. Budget: a few seconds on 10,000 periods × 100 assets.
- Deterministic: the bootstrap seed comes from the episode's check seed
  (`EvalData.check_seed`), like truncation's cuts.
- Generic panels: Date or Datetime timestamps, ragged universes (correlations
  use only assets present at both dates; nulls dropped per period).

## Approaches considered

- **A. Local backward extrapolation:** for each lag, extrapolate the decay of
  the next few lags back and test the excess. Rejected by the user as too
  complicated (window, z, share and tests at every lag).
- **B. One global exponential fit**, leave-one-out per lag. Picked.
- **C. Decay anchored on the signal's own autocorrelation.** Picked.
- B and C together, either one flags. Picked: they fail differently.
  B misses fast honest signals by design (no tail) and sees box edges;
  C handles fast signals (low memory, steep decay expected) but is blind to
  a moving-window leak whose memory and IC fade together (`sparse_plateau`).
  Both see the +1 spike.

## Design

Both rules read per-period rank ICs (Spearman across assets within a period)
of the fixed predictions; lags as in `ic_profile`: +1 is the target, +k the
return k − 1 periods after it.

### B: smooth decay (future side)

1. Per-period profile at lags +1…+K from `ic_profile` (future columns only),
   K = `profile_lags` = 10. Means m(k).
2. Model f(k) = a·φ^(k−1), φ ∈ [0, 1]. Fit by least squares on the means:
   a grid of φ (e.g. 101 values) with a in closed form for each φ,
   a = Σ m(k)·φ^(k−1) / Σ φ^(2(k−1)). a is signed, so a sign-flipped alpha
   fits the same way.
3. Leave one out: for each lag k, fit on the other lags, predict f₋ₖ(k), and
   take the residual oriented by the fit's sign:
   e(k) = sign(a₋ₖ)·(m(k) − f₋ₖ(k)).
4. Flag lag k if all three hold:
   - **tail exists:** |f₋ₖ(k)| > `tail_z` × its bootstrap standard error
     (a fast or useless signal's tail is noise, so B stays silent);
   - **significant:** e(k) > `decay_z` × its bootstrap standard error;
   - **material:** e(k) ≥ `decay_share` × |m(k)|: more than this share of the
     IC at k is unexplained by the smooth decay of the other lags.

### C: decay matches memory (delay view)

1. Per period t and delay d = 0…K:
   - IC_t(d) = rank IC of s(t − d) against the target r(t+1) (signal shifted
     within asset, return fixed). IC_t(0) is the +1 IC.
   - ρ_t(d) = rank correlation of s(t) with s(t − d) across assets.
   Periods where s(t − d) does not exist (before the first prediction) drop
   out for that d.
2. Memory ρ(d) = mean of ρ_t(d). Implied delayed IC: ρ(d)·D(0), where D(0) is
   the mean of IC_t(0).
3. Paired per-period gap on the same return cross-section, oriented by the
   sign of D(0): g_t(d) = sign(D(0))·(ρ(d)·IC_t(0) − IC_t(d)). Mean ḡ(d).
   Pairing on one return cancels most of the return noise.
4. Flag delay d ≥ 1 if:
   - **significant:** ḡ(d) > `memory_z` (6, stricter than B's 4; see
     Measured margins) × its bootstrap standard error;
   - **material:** ḡ(d) ≥ `decay_share` × |ρ(d)·D(0)|: the IC lost more than
     this share of what the signal's memory keeps.

### Confidence band: block bootstrap

- Moving-block bootstrap over periods: blocks of `bootstrap_block` = 20
  consecutive periods, `bootstrap_draws` = 200 resamples, seed from the check
  seed. Each resample recomputes the statistic from the resampled per-period
  rows: B refits (means, leave-one-out fits, residuals, tail predictions);
  C recomputes ρ(d), D(0) and ḡ(d). The standard error is the replicates'
  standard deviation.
- Blocks keep the serial correlation that persistent signals give the
  per-period ICs; the uncertainty of ρ(d) and of the fit is included.

### Combination and settings

- `implausible_ic` = `max_ic` at any future lag (as on the branch) **or** B
  **or** C.
- Remove `late_ic_peak`, `peak_z`, `peak_ratio`.
- New `CheckConfig` fields: `decay_z` = 4.0 (B), `memory_z` = 6.0 (C),
  `decay_share` = 0.5 (both), `tail_z` = 2.0 (B), `bootstrap_block` = 20,
  `bootstrap_draws` = 200. Keep
  `profile_lags` = 10 (B's lags and C's delays). Validate `profile_lags` ≥ 3
  (B needs at least two lags left to fit when one is left out).
- Keep `ic_profile` and the logged two-sided `shifted_ics`.

## Measured margins (throwaway prototype, 25 November 2026)

Honest set: naive, template, oracle, 20-period momentum, 5-period and
1-period reversal, and the oracle z-scored per period plus 0.5 or 1.0 times
each of momentum, 5- and 1-period reversal (z-scored), on eval tasks 0–9 of
the `profile-measure` secret, default task config, full hidden length:
120 runs. Exploits as in `tests/test_exploits.py`, on tasks 0–2, built on
the oracle-weighted honest signal.

| | B at z 4 | C at z 4 | C at z 6 |
| --- | --- | --- | --- |
| Honest flags | 0 / 120 | 3 / 120 (z 4.1–4.9) | 0 / 120 |
| `sparse_next` | 3/3 (z 4.6–7.3, share 0.53–0.60) | 3/3 (z 12.6–13.4) | 3/3 |
| `block_tail` | 2/3 (one at share 0.49) | 3/3 (z 6.6–7.4) | 3/3 |
| `sparse_after` | 3/3 | 0/3 | 0/3 |
| `sparse_plateau` | 0/3 | 0/3 | 0/3 |

C's three honest flags are all momentum: the 20-period momentum alpha on a
reversal task, and oracle + momentum on one plain task, at delays 3 and 7.
A slow signal's memory counts all of its content, its IC only the
predictive part, so the memory outlasts the edge and C reads the gap as a
leak. Hence C at z 6 (decided by the user). Margins are thin on both sides
(honest C up to 4.9, exploits from 6.6; B's exploit shares 0.51–0.60 against
0.5). Real data has more slow momentum-style signals, so C is the likelier
source of honest flags there. The rule C function carries a `# NOTE:` with
these limits and possible improvements (see below).

## Acceptance

- Honest set (as defined under Measured margins): 0 flags from B or C.
  The measurement is committed as `scripts/measure_decay_checks.py`, which
  prints per rule the largest z and share; its numbers go in Decision 29.
- `tests/test_exploits.py`: `sparse_next` and `block_tail` caught, their xfail
  becomes a plain test; `sparse_after` flips too if B catches it there, as
  measured. `sparse_plateau` and `sparse_after` stay strict xfail
  unless caught, in which case they flip too.
- Unit tests on a hand-built panel: a fast honest signal (no persistence, real
  IC) is flagged by neither rule; a slow honest signal plus a sparse leak is
  flagged by both; each rule's three or two conditions are each necessary
  (setting one threshold out of reach clears the flag).
- Runtime of both rules on a 10,000-period panel measured and stated.
- Spec Decision 29 rewritten (what the rules assume, the honest-blend margin,
  the dilution limit); both taskset prompts state the rule in a sentence.

## Decided by the user

- B and C, either one flags; C held to z 6 (option 1 after the measurement).
- A `# NOTE:` in the code on possible future improvements to C.

## Calls made without the user (check these)

- `Score` is unchanged: B and C fold into `implausible_ic`; the check
  functions return their margins (largest z and share) for the measurement
  script only.
- Materiality relative to the observed IC (B) and to the memory-implied IC
  (C), both at 50%; z = 4 as in the other checks.
- B's tail requirement at 2 standard errors.
- Block bootstrap with 20-period blocks and 200 draws; polars sampling.
- B's fit unweighted across lags, by grid search over φ.

## The `# NOTE:` on rule C (content)

Possible improvements, none built now:
- Estimate memory from the predictive part only: weight the autocorrelation
  by each asset-period's contribution to the IC, or use the autocorrelation of
  the signal's projection on the target, so slow non-predictive content does
  not lengthen the memory (the cause of the momentum false flags).
- Judge only short delays (d = 1, 2), where every measured exploit peaks and
  no honest flag occurred; the false flags were at d = 3 and 7.
- Calibrate `memory_z` and `decay_share` per data source once real data is
  available; the z 6 margin rests on 120 simulated runs.

## Left to the builder

- How to vectorise the fits and the bootstrap in polars, within the budget.
- Function names and module layout inside `quantenv/checks.py`.
- Exact prompt wording.
