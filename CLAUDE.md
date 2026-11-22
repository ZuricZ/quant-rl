# quant-rl

Quant research eval: an agent researches an alpha on simulated data and is scored on hidden data. The design is in `docs/Quant research eval build spec.md`; read it before changing the design. `quantenv/` is the core package and must stay generic so it works on real data later: simulator-specific knowledge lives only in `quantenv/sources/`, and checks take column names as arguments. The repository root is the one Hub environment, `quant-research`: its `pyproject.toml` ships `quantenv` and both tasksets, and nothing they need may live outside it (spec, Decision 28).

## Coding rules

- **Polars only**, used idiomatically. pandas is not a dependency.
  - Express work as polars expressions (`group_by().agg()`, `.over()`, `pl.corr(..., method="spearman")`) rather than Python loops.
  - Use lazy frames for chained steps.
  - Make no needless copies. Polars frames are immutable and share buffers. `clone()` is shallow and only guards agent code against `df[...] = ...`.
  - Don't convert between polars and numpy mid-pipeline. numpy is only for the seeded normal RNG in the generator and for seeding `np.random` for agent code.
  - Slice time with lazy `filter` / `is_between` on the timestamp column, by timestamp value. Never use `search_sorted`, row offsets or period numbers.
  - Mark sorted keys with `set_sorted`. Use compact dtypes (Enum, Float32, UInt32) where precision allows.
  - Write the obvious idiom first; optimise past it only with a measured reason.
- **Google Python style** and Google docstrings (Args/Returns/Raises).
- **Banned:** `from __future__ import annotations` and `math.nan` (use `float("nan")`).
- No placeholder or dead code: no stubs, no `if False:` helpers.
- The timestamp column is Date (daily bars) or Datetime (intraday). Never hardcode a date column or assume daily bars.

## Git

- Commit as the repo owner. Before the first commit in a new environment (e.g. a cloud session), run:
  `git config user.name "ZuricZ"` and `git config user.email "zan.zuric@gmail.com"`.
  Never author commits as Claude.
- Commit messages must not contain `Co-Authored-By` lines or any Claude/Anthropic attribution.
- Work on a branch, not `main`, unless told otherwise.

## Environment

- Python 3.12, managed with `uv`. Run `uv sync` and `uv run pytest -q`.
- verifiers 0.4.1 (`import verifiers.v1 as vf`). `vf.Toolset` and `vf.Runtime` need Linux, because they use `fcntl`. On Windows, use WSL or a cloud session.
- `uv sync` downloads about 465 MB. In cloud sessions, run it in the background or with progress output: a long silent command trips the idle watchdog.
