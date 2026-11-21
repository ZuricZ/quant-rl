# Hub Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `environments/quant_research/` a self-contained environment directory that plain `prime env push` publishes, with no staging script, no PyPI release and no git dependency.

**Architecture:** Move `pyproject.toml`, `quantenv/` and both taskset modules into `environments/quant_research/`. One distribution, `quant-research`, ships three top-level packages: `quantenv` (generic core, unchanged), `quant_research` and `quant_research_notools` (Decision 25). The repository root becomes a non-package uv project that installs that directory in editable mode, so `scripts/` and `tests/` keep working.

**Tech Stack:** uv, hatchling, Prime CLI 0.23, verifiers 0.4.1, Docker, GitHub Actions.

**Spec:** `docs/Quant research eval build spec.md` (Code layout, Decision 25). This plan replaces PR #7 (`hub-packaging`), which is closed unmerged. Its metadata, `.dockerignore` allowlist, Dockerfile README fix and Hub README section move into Task 1. `scripts/hub_push.py` is dropped.

## Global Constraints

- Python `>=3.12,<3.13`; dependencies `numpy>=2.0`, `polars>=2.1.0`, `verifiers>=0.4.1,<0.5`.
- `quantenv` stays its own top-level package, imports no verifiers code, and keeps simulator knowledge in `quantenv/sources/` (CLAUDE.md).
- Taskset ids `quant-research` / `quant-research-notools` resolve to modules `quant_research` / `quant_research_notools`. The grader runs as `python -m quant_research.grade` and `python -m quant_research.preflight` in the image; the image tag stays `quant-research:0.1.0`.
- Hub metadata (from PR #7): description `"Agentic quant research: find an alpha with limited backtests, scored on hidden data with lookahead checks."`, `tags = ["quant", "finance", "backtesting", "multi-turn", "tools", "reward-hacking"]`. Version `0.1.1` (the Hub has 0.1.0). Visibility PRIVATE; owner `zuricz`.
- Moves use `git mv`. Google style; no `from __future__ import annotations`, no `math.nan`. Commits by ZuricZ, no Co-Authored-By.

## Review Focus

1. **Stale install instructions.** `runs/prime-rl/notools-smoke.toml:8` says `uv pip install -e <quant-rl>`, but the root is no longer installable; it must say `<quant-rl>/environments/quant_research`. Pinned by `test_no_reference_to_the_old_layout`.
2. **Wrong wheel contents.** The wheel must hold all three packages and no tests, scripts or results. Pinned by `test_wheel_holds_the_three_packages_only`.
3. **verifiers leaking into the core.** Pinned by `test_quantenv_imports_no_verifiers`.
4. **Hub upload contents.** `prime env push` must upload only `pyproject.toml`, `README.md` and the three packages; the `.prime/` folder it writes must be gitignored. Checked in Task 2.
5. **Windows development.** `uv sync` and `uv run pytest -q` still work, and the verifiers tests skip on `fcntl`. Checked in Task 1, step 5.

---

### Task 1: Move the package into the environment directory

**Files:**
- Move: `quantenv/` → `environments/quant_research/quantenv/`; `environments/quant_research/{__init__,grade,preflight,taskset}.py` and `servers/` → `environments/quant_research/quant_research/`; `environments/quant_research_notools/{__init__,taskset}.py` → `environments/quant_research/quant_research_notools/`
- Create: `environments/quant_research/pyproject.toml`, `tests/test_packaging.py`
- Modify: root `pyproject.toml`, `uv.lock`, `environments/quant_research/README.md` (merge in the notools README, then delete it), `Dockerfile`, `.dockerignore`, `.gitignore`, `runs/prime-rl/notools-smoke.toml:1-18`, `README.md` (Quickstart, Layout, Hub section), `CLAUDE.md:3`, the spec (Code layout, Decision 25 first bullet, new Decision 28)

**Interfaces:**
- Produces: `ENV = ROOT / "environments/quant_research"` in `tests/test_packaging.py`, where `ROOT = pathlib.Path(__file__).parents[1]`.

- [ ] **Step 1: Write the failing tests** in `tests/test_packaging.py`

```python
def test_wheel_holds_the_three_packages_only(tmp_path):
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ENV)], check=True)
    (wheel,) = tmp_path.glob("*.whl")
    tops = {n.split("/")[0] for n in zipfile.ZipFile(wheel).namelist()
            if not n.split("/")[0].endswith(".dist-info")}
    assert tops == {"quantenv", "quant_research", "quant_research_notools"}


def test_quantenv_imports_no_verifiers():
    offenders = [p for p in (ENV / "quantenv").rglob("*.py") if "verifiers" in p.read_text()]
    assert offenders == []


def test_no_reference_to_the_old_layout():
    docs = ["Dockerfile", "README.md", "CLAUDE.md", "runs/prime-rl/notools-smoke.toml"]
    stale = ("uv pip install -e <quant-rl>`", "environments/quant_research_notools")
    assert [(d, s) for d in docs for s in stale if s in (ROOT / d).read_text()] == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: FAIL. `environments/quant_research` has no `pyproject.toml`, and the smoke config still names the old install.

- [ ] **Step 3: Move the files and write the environment's `pyproject.toml`**

Use `git mv` as listed under Files. `[project]`: `name = "quant-research"`, `version = "0.1.1"`, the description and tags from Global Constraints (tags with the comment "Not a standard field: `prime env push` reads it for the Hub listing."), `readme = "README.md"`, the Python range and dependencies above. Hatchling, with `[tool.hatch.build.targets.wheel] packages = ["quantenv", "quant_research", "quant_research_notools"]`. Root `pyproject.toml`: `[project] name = "quant-rl"`, `version = "0.0.0"`, `dependencies = ["quant-research"]`; `[tool.uv] package = false`; `[tool.uv.sources] quant-research = { path = "environments/quant_research", editable = true }`; keep the `dev` group and the pytest config; drop the root build-system and hatch sections.

- [ ] **Step 4: Update the image, ignore files and references**

- Dockerfile: `COPY environments/quant_research ./`, then `RUN pip install --no-cache-dir . && rm -rf /opt/quant-research`. Update the comment to say the image must be rebuilt after changing `environments/quant_research/`.
- `.dockerignore`: an allowlist of `environments/quant_research/`, excluding `**/__pycache__/` and `**/*.pyc`.
- `.gitignore`: add `.prime/`.
- prime-rl comment: `uv pip install -e <quant-rl>/environments/quant_research`.
- README: install paths; layout; a Hub section (`prime env install zuricz/quant-research` provides both taskset ids; build the image from the repository; publish with `cd environments/quant_research && prime env push`, bumping `version` first).
- CLAUDE.md line 3: `environments/quant_research/quantenv/` is the core package.
- Spec: update the Code layout paths. In Decision 25's first bullet, "the root distribution" becomes "the `quant-research` distribution". Add **Decision 28, "The environment directory is the package"**, dated 21 November 2026: Prime pushes one self-contained directory, so `quantenv` ships inside the environment's wheel as its own top-level package. It moves out to its own distribution only if a second environment or repository needs it.

- [ ] **Step 5: Lock and run the suite on Windows and in WSL**

Run: `uv lock && uv sync && uv run pytest -q`
Expected: all tests PASS on Windows, with the verifiers tests skipping. In WSL (`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/quant-rl`), `test_reward_is_the_hidden_ic_over_the_ceiling` is known to fail on `main` in a full-file run ("alpha process died"); that failure is not a regression.

- [ ] **Step 6: Commit, push and open the PR; CI checks the image**

```bash
git add -A
git commit -m "Environment directory is the package: quantenv and both tasksets"
git push -u origin hub-layout
gh pr create --base main --title "Self-contained environment directory for the Prime Hub"
```

Expected: CI passes "Build the task image", "Sandbox tests as root" and "Validate the tasksets in Docker". Close PR #7 with a link to the new PR.

### Task 2: Push to the Hub (the owner runs or approves)

- [ ] **Step 1: Push from the environment directory**

Run: `cd environments/quant_research && PYTHONIOENCODING=utf-8 prime env push --visibility PRIVATE`
Expected: `zuricz/quant-research` 0.1.1 pushed.

- [ ] **Step 2: Check what was uploaded and that a clean install works**

Run: `prime env inspect zuricz/quant-research@0.1.1`
Expected: only `pyproject.toml`, `README.md`, `quantenv/`, `quant_research/` and `quant_research_notools/`, and `git status` is clean (`.prime/` is ignored).

Then: `prime env install zuricz/quant-research` into a fresh venv, and `python -c "import quantenv, quant_research, quant_research_notools"` succeeds.
