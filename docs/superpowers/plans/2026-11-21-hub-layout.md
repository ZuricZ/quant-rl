# Hub Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repository into Prime's layout: a standalone `quantenv` package and a self-contained environment directory that plain `prime env push` publishes, so no staging script is needed.

**Architecture:** The root becomes a virtual uv workspace with two members. `packages/quantenv` is the generic core, published to PyPI. `environments/quant_research` is the Hub environment: one distribution, `quant-research`, holding both taskset modules (`quant_research`, `quant_research_notools`; Decision 25 keeps them in one distribution) and depending on `quantenv>=0.1,<0.2`. In development, CI and the task image, `quantenv` comes from the workspace; on the Hub it comes from PyPI.

**Tech Stack:** uv workspaces, hatchling, Prime CLI 0.23 (`prime env push`), verifiers 0.4.1, Docker, GitHub Actions.

**Spec:** `docs/Quant research eval build spec.md` (Code layout, Decision 25, Stack notes). This plan replaces PR #7 (`hub-packaging`), which is closed unmerged. PR #7's metadata, `.dockerignore` allowlist, README fix and Hub README section move into the tasks below. `scripts/hub_push.py` is not carried over.

## Global Constraints

- Python `>=3.12,<3.13`; `verifiers>=0.4.1,<0.5`; `numpy>=2.0`; `polars>=2.1.0`. Polars only, no pandas (CLAUDE.md).
- `quantenv` imports no verifiers code and stays generic: simulator knowledge only in `quantenv/sources/` (CLAUDE.md, spec Code layout).
- Taskset ids `quant-research` and `quant-research-notools` keep resolving to modules `quant_research` and `quant_research_notools` (Decision 25).
- The grader runs as `python -m quant_research.grade` and `python -m quant_research.preflight` in the task image (`environments/quant_research/taskset.py:420,440`). The image tag stays `quant-research:0.1.0`.
- Hub metadata, from PR #7: description `"Agentic quant research: find an alpha with limited backtests, scored on hidden data with lookahead checks."`, `tags = ["quant", "finance", "backtesting", "multi-turn", "tools", "reward-hacking"]` (a non-standard `[project]` key that `prime env push` reads; hatchling accepts it).
- Hub visibility PRIVATE until a hosted run works. Hub owner `zuricz`.
- Moves use `git mv`, so history follows the files. Banned: `from __future__ import annotations`, `math.nan`. Google docstrings.
- Commits authored as ZuricZ, no Co-Authored-By lines (CLAUDE.md).
- CI (Docker image, sandbox tests, `vf-validate`) breaks after Task 1 and is expected to pass again only after Task 3, so push the branch once Task 3 is done. Tasks 1 and 2 are gated by the local test suite.

## Review Focus

1. **Stale paths.** The prime-rl config comment (`runs/prime-rl/notools-smoke.toml:8`) says `uv pip install -e <quant-rl>`, but the root is no longer installable. It must name both members. Also check the Dockerfile comment, README layout, spec Code layout, CLAUDE.md line 3 and the environment READMEs. Pinned by the grep test in Task 3.
2. **quantenv drift.** A PR that changes `quantenv` but not its version: the Hub environment would install the old PyPI release. Pinned by the version-range test in Task 2.
3. **Wrong wheel contents.** The environment wheel bundles a copy of `quantenv` or the tests, or omits `quant_research_notools`. Pinned by the wheel tests in Task 2.
4. **Hub upload contents.** After the move, `prime env push` from the environment directory must upload only `pyproject.toml`, `README.md` and `src/`. Prime also writes `.prime/.env-metadata.json` there, which git must ignore. Checked in Task 4, step 5.
5. **Windows development.** `uv sync` and `uv run pytest -q` must still work on Windows, with the verifiers tests skipping on `fcntl` as now. Checked in Task 1, step 5.

---

## Target layout

```
pyproject.toml                         virtual workspace root: members, dev group, pytest config
packages/quantenv/
  pyproject.toml                       quantenv 0.1.0: numpy, polars
  README.md
  src/quantenv/...                     moved from quantenv/
environments/quant_research/
  pyproject.toml                       quant-research 0.1.1: quantenv>=0.1,<0.2, verifiers
  README.md                            the Hub page: both tasksets
  src/quant_research/...               moved from environments/quant_research/
  src/quant_research_notools/...       moved from environments/quant_research_notools/
tests/test_packaging.py                new
```

### Task 1: Extract `quantenv` and make the root a workspace

**Files:**
- Move: `quantenv/` → `packages/quantenv/src/quantenv/`
- Create: `packages/quantenv/pyproject.toml`, `packages/quantenv/README.md`, `tests/test_packaging.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: workspace member `quantenv` (distribution `quantenv`, version `0.1.0`); test helper `build_wheel(package: str, out: pathlib.Path) -> zipfile.ZipFile` in `tests/test_packaging.py`, which runs `uv build --package <package> --wheel --out-dir <out>` from the repository root and opens the single wheel.

- [ ] **Step 1: Write the failing tests** in `tests/test_packaging.py`

```python
def test_quantenv_wheel_is_the_core_alone(tmp_path):
    wheel = build_wheel("quantenv", tmp_path)
    names = wheel.namelist()
    assert "quantenv/sources/synthetic.py" in names
    assert not any(n.startswith(("quant_research", "tests/")) for n in names)
    requires = wheel_requires(wheel)  # Requires-Dist lines of METADATA
    assert not any(r.startswith("verifiers") for r in requires)


def test_quantenv_source_imports_no_verifiers():
    src = ROOT / "packages/quantenv/src/quantenv"
    offenders = [p for p in src.rglob("*.py") if "verifiers" in p.read_text()]
    assert offenders == []
```

`ROOT` is the repository root, from `pathlib.Path(__file__).parents[1]`. `wheel_requires(wheel: zipfile.ZipFile) -> list[str]` reads `*.dist-info/METADATA` and returns the values of its `Requires-Dist:` lines.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: FAIL. `uv build --package quantenv` errors, because there is no such workspace member.

- [ ] **Step 3: Move the package and write the member's `pyproject.toml`**

`git mv quantenv packages/quantenv/src/quantenv`. Member `[project]`: `name = "quantenv"`, `version = "0.1.0"`, a description of "Panel data, walk-forward, sandboxed scoring and lookahead checks for alpha research", `readme = "README.md"`, `requires-python = ">=3.12,<3.13"`, `dependencies = ["numpy>=2.0", "polars>=2.1.0"]`, hatchling build backend. The src layout needs no `packages` setting. The README is 10–20 lines: what the package is and its modules, taken from the spec's Code layout.

- [ ] **Step 4: Make the root a virtual workspace**

Root `pyproject.toml`: `[project] name = "quant-rl"`, `version = "0.0.0"`, same `requires-python`, `dependencies = ["quantenv", "quant-research"]`; `[tool.uv] package = false`; `[tool.uv.workspace] members = ["packages/*", "environments/*"]`; `[tool.uv.sources]` with both set to `{ workspace = true }`. Keep the `dev` group (add `packaging>=24` for Task 2's test) and `[tool.pytest.ini_options]`. Drop the root build-system and hatch sections. `quant-research` is not a member until Task 2, so in this task list only `quantenv` in `dependencies` and sources, and keep the environment modules importable by temporarily adding `environments` to `pythonpath` in the pytest config. Task 2 removes that line.

- [ ] **Step 5: Lock, then run the full suite on Windows and in WSL**

Run: `uv lock && uv sync && uv run pytest -q`
Expected: the packaging tests PASS. Everything that passed on `main` passes; the verifiers tests skip on Windows. Run the same in WSL with `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/quant-rl`. There, `test_reward_is_the_hidden_ic_over_the_ceiling` is known to fail on `main` in a full-file run ("alpha process died"); that failure is not a regression.

- [ ] **Step 6: Commit**

```bash
git add -A packages pyproject.toml uv.lock tests/test_packaging.py
git commit -m "Extract quantenv into a workspace package"
```

### Task 2: Make `environments/quant_research` a self-contained environment

**Files:**
- Move: `environments/quant_research/{__init__,grade,preflight,taskset}.py`, `servers/` → `environments/quant_research/src/quant_research/`; `environments/quant_research_notools/{__init__,taskset}.py` → `environments/quant_research/src/quant_research_notools/`
- Create: `environments/quant_research/pyproject.toml`
- Modify: `environments/quant_research/README.md` (merge in `environments/quant_research_notools/README.md`, then delete that file), root `pyproject.toml`, `uv.lock`, `tests/test_packaging.py`

**Interfaces:**
- Consumes: `build_wheel`, `wheel_requires`, `ROOT` (Task 1).
- Produces: distribution `quant-research` 0.1.1 with top-level modules `quant_research` and `quant_research_notools`; taskset ids unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_env_wheel_holds_both_tasksets_and_not_quantenv(tmp_path):
    names = build_wheel("quant-research", tmp_path).namelist()
    assert "quant_research/grade.py" in names
    assert "quant_research/servers/research.py" in names
    assert "quant_research_notools/taskset.py" in names
    assert not any(n.startswith(("quantenv/", "tests/")) for n in names)


def test_env_requires_a_quantenv_range_that_admits_the_workspace_version(tmp_path):
    requires = wheel_requires(build_wheel("quant-research", tmp_path))
    (quantenv_req,) = [Requirement(r) for r in requires if r.startswith("quantenv")]
    assert quantenv_req.url is None  # resolvable from PyPI, not a path or git URL
    workspace = tomllib.loads((ROOT / "packages/quantenv/pyproject.toml").read_text())
    assert quantenv_req.specifier.contains(workspace["project"]["version"])


def test_env_metadata_is_the_hub_listing():
    project = tomllib.loads(
        (ROOT / "environments/quant_research/pyproject.toml").read_text()
    )["project"]
    assert project["readme"] == "README.md"
    assert set(project["tags"]) >= {"quant", "backtesting", "reward-hacking"}
```

`Requirement` comes from `packaging.requirements`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: the three new tests FAIL, because there is no member `quant-research`.

- [ ] **Step 3: Move the modules and write the environment's `pyproject.toml`**

Use `git mv` as listed under Files. `[project]`: `name = "quant-research"`, `version = "0.1.1"` (the Hub has 0.1.0), the description and tags from Global Constraints, `readme = "README.md"`, `requires-python = ">=3.12,<3.13"`, `dependencies = ["quantenv>=0.1,<0.2", "verifiers>=0.4.1,<0.5"]`. `[tool.uv.sources] quantenv = { workspace = true }`. `[tool.hatch.build.targets.wheel] packages = ["src/quant_research", "src/quant_research_notools"]`. Root `pyproject.toml`: add `quant-research` to `dependencies` and sources, and remove the temporary `pythonpath` line from Task 1.

- [ ] **Step 4: Write the Hub README**

`environments/quant_research/README.md` is the Hub page. Keep its current content, add a "No-tools variant" section made from the notools README, and add an install section: `prime env install zuricz/quant-research`, which provides both taskset ids, plus how to build the task image from the repository. Delete `environments/quant_research_notools/` once it is empty.

- [ ] **Step 5: Lock and run everything**

Run: `uv lock && uv sync && uv run pytest -q`, on Windows and in WSL.
Expected: all packaging tests PASS, and no other regressions (same WSL caveat as Task 1).

- [ ] **Step 6: Commit**

```bash
git add -A environments pyproject.toml uv.lock tests/test_packaging.py
git commit -m "Self-contained quant-research environment with both tasksets"
```

### Task 3: Task image, CI and every path reference

**Files:**
- Modify: `Dockerfile`, `.dockerignore`, `.gitignore`, `.github/workflows/test.yml` (only if a path in it changes), `runs/prime-rl/notools-smoke.toml:1-18`, `README.md` (Layout and Quickstart), `CLAUDE.md:3`, `docs/Quant research eval build spec.md` (Code layout, Decision 25 first bullet, new Decision 28), `tests/test_packaging.py`

**Interfaces:**
- Consumes: the layout of Tasks 1 and 2.

- [ ] **Step 1: Write the failing stale-path test**

```python
STALE = ("uv pip install -e <quant-rl>`", "environments/quant_research_notools/")


def test_no_reference_to_the_old_layout():
    docs = ["Dockerfile", "README.md", "CLAUDE.md", "runs/prime-rl/notools-smoke.toml",
            "environments/quant_research/README.md"]
    hits = [(d, s) for d in docs for s in STALE if s in (ROOT / d).read_text()]
    assert hits == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_packaging.py::test_no_reference_to_the_old_layout -v`
Expected: FAIL, listing at least `runs/prime-rl/notools-smoke.toml`.

- [ ] **Step 3: Update the image and the ignore files**

Dockerfile: `COPY packages/quantenv ./packages/quantenv` and `COPY environments/quant_research ./environments/quant_research`, then `RUN pip install --no-cache-dir ./packages/quantenv ./environments/quant_research && rm -rf /opt/quant-research`. pip satisfies the `quantenv` requirement from the local path, so the image never downloads it. `.dockerignore`: an allowlist of `packages/quantenv/` and `environments/quant_research/`, keeping `**/__pycache__/` and `**/*.pyc` excluded. Each member's README must be included, since hatchling reads it. `.gitignore`: add `.prime/`.

- [ ] **Step 4: Update the references**

prime-rl comment: `uv pip install -e <quant-rl>/packages/quantenv -e <quant-rl>/environments/quant_research`. Make the same path changes in the README Layout and Quickstart, the spec's Code layout and CLAUDE.md line 3 (`packages/quantenv/` is the core package). In Decision 25's first bullet, "a second package in the root distribution" becomes "a second module in the `quant-research` distribution". Add **Decision 28, "The repository follows Prime's environment layout"**, dated 21 November 2026, in three bullets: the target layout above; `quantenv` comes from PyPI on the Hub and from the workspace everywhere else; any change to `quantenv` bumps its version and is published before the environment is pushed (`test_env_requires_a_quantenv_range_that_admits_the_workspace_version`).

- [ ] **Step 5: Run the tests, then let CI check the image**

Run: `uv run pytest -q`
Expected: PASS. Docker isn't available locally, so push the branch and open the PR. CI must pass the "Build the task image", "Sandbox tests as root" and "Validate the tasksets in Docker" steps; the last one runs both graders as `python -m quant_research.grade`.

- [ ] **Step 6: Commit and open the PR**

```bash
git add -A
git commit -m "Task image, CI and docs for the workspace layout; Decision 28"
git push -u origin hub-layout
gh pr create --base main --title "Prime environment layout: quantenv package and self-contained environment"
```

Close PR #7 with a comment pointing to the new PR.

### Task 4: Publish `quantenv` and push the environment (run by the repository owner)

These steps publish packages, so the owner runs them or approves each one. No code changes, apart from a README note if a check fails.

- [ ] **Step 1: Build and check the `quantenv` distribution**

Run: `uv build --package quantenv --out-dir dist/quantenv`
Expected: `quantenv-0.1.0.tar.gz` and `quantenv-0.1.0-py3-none-any.whl`.

- [ ] **Step 2: Publish to TestPyPI, then PyPI** (the owner's tokens)

Run: `uv publish --publish-url https://test.pypi.org/legacy/ dist/quantenv/*`, then `uv publish dist/quantenv/*`
Expected: `https://pypi.org/project/quantenv/0.1.0/` exists. The name was free on 21 November 2026.

- [ ] **Step 3: Push the environment from its own directory**

Run: `cd environments/quant_research && PYTHONIOENCODING=utf-8 prime env push --visibility PRIVATE`
Expected: `zuricz/quant-research` 0.1.1 pushed.

- [ ] **Step 4: Install it in a clean environment**

Run: `uv venv /tmp/hubcheck && VIRTUAL_ENV=/tmp/hubcheck prime env install zuricz/quant-research && /tmp/hubcheck/bin/python -c "import quant_research, quant_research_notools, quantenv"`
Expected: no error, and `quantenv` 0.1.0 was installed from PyPI.

- [ ] **Step 5: Check what was uploaded**

Run: `prime env inspect zuricz/quant-research@0.1.1`
Expected: only `pyproject.toml`, `README.md` and `src/`. `git status` shows nothing from `.prime/`.
