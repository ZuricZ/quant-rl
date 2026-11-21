"""Repository layout: what may import verifiers, and environment independence."""

import pathlib
import re

ROOT = pathlib.Path(__file__).parents[1]
VERIFIERS = re.compile(r"^\s*(import|from)\s+verifiers\b", re.MULTILINE)


def _imports_verifiers(path: pathlib.Path) -> bool:
    return bool(VERIFIERS.search(path.read_text(encoding="utf-8")))


def test_only_taskset_and_grading_import_verifiers():
    users = {p.name for p in (ROOT / "quantenv").rglob("*.py") if _imports_verifiers(p)}
    assert users == {"taskset.py", "grading.py"}


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
