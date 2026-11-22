"""Repository layout: what may import verifiers, and environment independence."""

import pathlib
import re
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).parents[1]
VERIFIERS = re.compile(r"^\s*(import|from)\s+verifiers\b", re.MULTILINE)


def _imports_verifiers(path: pathlib.Path) -> bool:
    return bool(VERIFIERS.search(path.read_text(encoding="utf-8")))


def test_only_taskset_and_grading_import_verifiers():
    users = {p.name for p in (ROOT / "quantenv").rglob("*.py") if _imports_verifiers(p)}
    assert users == {"taskset.py", "grading.py"}


def test_the_rest_of_quantenv_runs_without_verifiers():
    # Catches indirect imports too, e.g. `grade.py` importing `quantenv.taskset`,
    # which would load verifiers in the container on every grading run.
    modules = [
        ".".join(p.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        for p in (ROOT / "quantenv").rglob("*.py")
        if p.name not in ("taskset.py", "grading.py")
    ]
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        "print('verifiers' in sys.modules)"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True
    )
    assert run.stdout.strip() == "False"


def test_environments_do_not_import_each_other():
    pattern = {
        "quant_research": r"\bquant_research_notools\b",
        "quant_research_notools": r"\bquant_research\b(?!_)",
    }
    hits = [
        p
        for env, pat in pattern.items()
        for p in (ROOT / "environments" / env).rglob("*.py")
        if re.search(
            rf"^\s*(import|from)\s+{pat}", p.read_text(encoding="utf-8"), re.MULTILINE
        )
    ]
    assert hits == []


@pytest.mark.parametrize(
    "env, own",
    [
        ("quant_research", {"__init__.py", "taskset.py", "servers"}),
        ("quant_research_notools", {"__init__.py", "taskset.py"}),
    ],
)
def test_environment_wheels(tmp_path, env, own):
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT / "environments" / env)],
        check=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    tops = {n.split("/")[0] for n in names if ".dist-info/" not in n}
    assert tops == {env, "quantenv"}
    assert {n.split("/")[1] for n in names if n.startswith(f"{env}/")} == own
    assert "quantenv/grade.py" in names
