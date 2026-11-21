# Hub Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each folder under `environments/` an independent Hub environment that plain `prime env push` publishes, with the shared code in `quantenv`.

**Architecture:** The pieces both tasksets share (splits and secret, task data, prompt text, the image, grading in the runtime) move out of `quant_research` into two `quantenv` modules: `quantenv/taskset.py` and `quantenv/grading.py`. These are the only `quantenv` modules that import verifiers. The container entry points move to `quantenv/grade.py` and `quantenv/preflight.py`. Neither environment imports the other. Each environment folder gets a `pyproject.toml` whose wheel force-includes `../../quantenv`. The root `pyproject.toml` stays the development and task-image build.

**Tech Stack:** hatchling (`force-include`, `bypass-selection`), uv, Prime CLI 0.23, verifiers 0.4.1, Docker, GitHub Actions.

**Spec:** `docs/Quant research eval build spec.md` (Code layout, Decision 25). This plan replaces PR #7 (`hub-packaging`), which is closed unmerged. Its Hub metadata and `.dockerignore` allowlist move into this plan; `scripts/hub_push.py` is dropped.

## Naming

The names follow verifiers' own modules. `verifiers/v1/taskset.py` and `verifiers/v1/tasksets/<name>/taskset.py` hold taskset-level code; `verifiers/v1/grading.py` is "Task scoring in a fresh runtime", which is exactly what ours does.

- `quantenv/taskset.py`, not `task.py`: it holds taskset-level settings (`SplitConfig`) and shared prompt text, not one `vf.Task`.
- `quantenv/grading.py`: the host side (`grade`, `scoring_timeout`, `check_sandbox_user`, `SANDBOX_USER`).
- `quantenv/grade.py` and `quantenv/preflight.py`: the scripts run inside the container (`python -m quantenv.grade`, `python -m quantenv.preflight`). They stay top-level modules, not part of a `grading` subpackage, because importing a subpackage runs its `__init__`, which would import verifiers on every grading run in the container. "Preflight" is the usual name for a check run before the real work.

## Global Constraints

- Python `>=3.12,<3.13`; dependencies `numpy>=2.0`, `polars>=2.1.0`, `verifiers>=0.4.1,<0.5`.
- Only `quantenv/taskset.py` and `quantenv/grading.py` may import verifiers. Every other `quantenv` module, `grade.py` and `preflight.py` included, stays verifiers-free, so `scripts/run_baselines.py` and the core tests keep running on Windows.
- Taskset ids `quant-research` / `quant-research-notools` keep resolving to modules `quant_research` / `quant_research_notools`. The image tag stays `quant-research:0.1.0`.
- The main prompt is unchanged byte for byte (Decision 25).
- Hub listings: `quant-research` 0.1.1 and `quant-research-notools` 0.1.0. Both use description and tags from PR #7, adapted per environment: tags `["quant", "finance", "backtesting", "reward-hacking"]`, plus `"multi-turn", "tools"` for the tools environment and `"single-turn"` for notools. Visibility PRIVATE; owner `zuricz`.
- Moves use `git mv`. Google style; no `from __future__ import annotations`, no `math.nan`. Commits by ZuricZ, no Co-Authored-By.

## Review Focus

1. **Grader command strings.** `grade` and `check_sandbox_user` must invoke `python -m quantenv.grade` and `python -m quantenv.preflight`. A stale `quant_research.grade` only fails inside Docker, so CI's "Validate the tasksets in Docker" step catches it; no local test can.
2. **A hidden coupling left behind.** Notools still importing `quant_research`, or the reverse. Pinned by `test_environments_do_not_import_each_other`.
3. **Verifiers leaking into the core.** Pinned by `test_only_taskset_and_grading_import_verifiers`.
4. **Wrong wheel contents per environment.** Pinned by `test_environment_wheels`.
5. **Hub upload contents.** `prime env push` uploads only the environment folder; its `.prime/` folder must be gitignored. Checked in Task 3.

---

### Task 1: Move the shared code into `quantenv`

**Files:**
- Create: `quantenv/taskset.py`, `quantenv/grading.py`
- Move: `environments/quant_research/grade.py` → `quantenv/grade.py`; `environments/quant_research/preflight.py` → `quantenv/preflight.py`
- Modify: `environments/quant_research/taskset.py`, `environments/quant_research/servers/research.py`, `environments/quant_research_notools/taskset.py`, `tests/test_taskset.py`, `tests/test_notools.py`, `tests/test_sandbox.py` (if it names the grader module)
- Create: `tests/test_layout.py`

**Interfaces:**
- Produces, in `quantenv.taskset`: `EVAL_TASKS`, `IMAGE`, `SECRET_VAR`, `DATA_SECTION`, `ALPHA_SECTION`, `PORTFOLIO`, `EXAMPLE`, `prompt_choices`, `prompt_values`, `secret_from_env`, `SplitConfig`, `ResearchData`, `describe_panel`, with their private helpers. These names and signatures stay as they are now.
- Produces, in `quantenv.grading`: `SANDBOX_USER`, `grade`, `scoring_timeout`, `check_sandbox_user`, with unchanged signatures.
- Everything else stays in its environment: the toolset, `QuantResearchTask*`, prompt pieces only the tools prompt uses (`WORKFLOW`, `OPENINGS`, `PROMPT`, ...), and all of notools.

- [ ] **Step 1: Write the failing tests** in `tests/test_layout.py`

```python
ROOT = pathlib.Path(__file__).parents[1]
VERIFIERS = re.compile(r"^\s*(import|from)\s+verifiers\b", re.MULTILINE)


def _imports_verifiers(path):
    return bool(VERIFIERS.search(path.read_text()))


def test_only_taskset_and_grading_import_verifiers():
    users = {p.name for p in (ROOT / "quantenv").rglob("*.py") if _imports_verifiers(p)}
    assert users == {"taskset.py", "grading.py"}


def test_environments_do_not_import_each_other():
    pattern = {"quant_research": r"\bquant_research_notools\b",
               "quant_research_notools": r"\bquant_research\b(?!_)"}
    hits = [p for env, pat in pattern.items()
            for p in (ROOT / "environments" / env).rglob("*.py")
            if re.search(rf"^\s*(import|from)\s+{pat}", p.read_text(), re.MULTILINE)]
    assert hits == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_layout.py -v`
Expected: both FAIL. The first finds no `quantenv` module importing verifiers; the second finds `environments/quant_research_notools/taskset.py`.

- [ ] **Step 3: Move the code**

Create the two modules with the names listed under Interfaces, cut from `environments/quant_research/taskset.py` and `servers/research.py`. Use `git mv` for `grade.py` and `preflight.py`, and change the commands in `grade` and `check_sandbox_user` to `python -m quantenv.grade` / `python -m quantenv.preflight`. Point both environments' imports at `quantenv.taskset` / `quantenv.grading`, and update the test imports to match. Re-exports from the old locations are not kept.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q` on Windows, then the same in WSL with `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/quant-rl`.
Expected: `tests/test_layout.py` PASSES, and so does everything that passed on `main`. In WSL, `test_reward_is_the_hidden_ic_over_the_ceiling` is known to fail on `main` in a full-file run ("alpha process died"); that failure is not a regression.

- [ ] **Step 5: Commit**

```bash
git add -A quantenv environments tests
git commit -m "Shared taskset and grading code moves into quantenv; environments stop importing each other"
```

### Task 2: One `pyproject.toml` per environment

**Files:**
- Create: `environments/quant_research/pyproject.toml`, `environments/quant_research_notools/pyproject.toml`
- Modify: root `pyproject.toml` (`name = "quant-rl"`; everything else unchanged), `uv.lock`, `.dockerignore`, `.gitignore`, `README.md` (Hub section), `CLAUDE.md:3`, the spec (Code layout, Decision 25, new Decision 28), `runs/prime-rl/notools-smoke.toml` comments if they name the grader module, `tests/test_layout.py`

**Interfaces:**
- Consumes: the layout from Task 1.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("env, own", [
    ("quant_research", {"__init__.py", "taskset.py", "servers"}),
    ("quant_research_notools", {"__init__.py", "taskset.py"}),
])
def test_environment_wheels(tmp_path, env, own):
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path),
                    str(ROOT / "environments" / env)], check=True)
    (wheel,) = tmp_path.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    tops = {n.split("/")[0] for n in names if ".dist-info/" not in n}
    assert tops == {env, "quantenv"}
    assert {n.split("/")[1] for n in names if n.startswith(f"{env}/")} == own
    assert "quantenv/grade.py" in names
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_layout.py::test_environment_wheels -v`
Expected: FAIL. `uv build` finds no `pyproject.toml` in the environment folders.

- [ ] **Step 3: Write both `pyproject.toml` files**

`[project]`: name, version, description, tags (with the comment "Not a standard field: `prime env push` reads it for the Hub listing."), `readme = "README.md"`, the Python range and the three dependencies. Hatchling, with `[tool.hatch.build.targets.wheel] bypass-selection = true` and a `force-include` table. It maps each of the environment's own files or folders to `<module>/<name>`, and `"../../quantenv" = "quantenv"`. This was prototyped on 21 November 2026: `bypass-selection` plus `force-include` produced exactly the expected top levels. A `sources = {"" = ...}` prefix does not work here, because it also prefixes the force-included `quantenv`.

- [ ] **Step 4: Ignore files, docs and spec**

- `.dockerignore`: PR #7's allowlist (`pyproject.toml`, `quantenv/`, `environments/`, excluding `**/__pycache__/` and `**/*.pyc`).
- `.gitignore`: add `.prime/`.
- README Hub section: `prime env install zuricz/quant-research` / `zuricz/quant-research-notools`; build the image from the repository; publish with `cd environments/<env> && prime env push` after bumping `version`.
- CLAUDE.md line 3: `quantenv/` is the core package; only `quantenv/taskset.py` and `quantenv/grading.py` import verifiers.
- Spec Code layout: list the four new `quantenv` modules and the per-environment `pyproject.toml`.
- Decision 25: notools is its own Hub environment and shares code only through `quantenv`.
- Add **Decision 28, "Each environment folder is a Hub environment"**, dated 21 November 2026: why the shared code moved into `quantenv`, why verifiers is confined to two modules, and the force-include packaging.

- [ ] **Step 5: Lock and run everything**

Run: `uv lock && uv sync && uv run pytest -q`
Expected: all PASS, with the same WSL caveat as Task 1.

- [ ] **Step 6: Commit, push and open the PR**

```bash
git add -A
git commit -m "Per-environment pyproject for the Hub; Decision 28"
git push -u origin hub-layout
gh pr create --base main --title "Each environment folder is a Hub environment"
```

Expected: CI passes, including "Validate the tasksets in Docker", which runs `python -m quantenv.grade` in the image. Close PR #7 with a link to the new PR.

### Task 3: Push both environments (the owner runs or approves)

- [ ] **Step 1: Push**

Run: `cd environments/quant_research && PYTHONIOENCODING=utf-8 prime env push --visibility PRIVATE`, then the same from `environments/quant_research_notools`.
Expected: `zuricz/quant-research` 0.1.1 and `zuricz/quant-research-notools` 0.1.0.

- [ ] **Step 2: Check the uploads and a clean install**

Run: `prime env inspect zuricz/quant-research@0.1.1` and `prime env inspect zuricz/quant-research-notools@0.1.0`
Expected: only each environment's own files, and `git status` is clean. A fresh venv with `prime env pull` plus `uv pip install .` for each can import its module and `quantenv`. Private environments install through `pull`, per `prime env info`.
