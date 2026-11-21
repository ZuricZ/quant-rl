"""Pushes the quant-research package to the Prime Environments Hub.

`prime env push` uploads every file under the directory it pushes, so pushing
from the repository root would also upload `results/`, `docs/`, `tests/` and
the run configs. This script copies only what the package is built from
(`pyproject.toml`, `README.md` and the packages the wheel lists) into a
temporary directory and pushes from there. Arguments after `--` go to
`prime env push` unchanged.

The one package holds both taskset ids, `quant-research` (tools) and
`quant-research-notools` (one reply), and the shared `quantenv` core.

Usage:
    uv run python scripts/hub_push.py [--visibility PRIVATE] [--dry-run]
        [-- <prime env push args>]
"""

import argparse
import pathlib
import shutil
import subprocess
import tempfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Files the wheel's metadata reads, besides the packages.
TOP_LEVEL = ("pyproject.toml", "README.md")
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc")


def packages() -> list[str]:
    """The package directories the wheel is built from.

    Returns:
        Paths relative to the repository root, as `pyproject.toml` lists them.
    """
    with (ROOT / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)
    return config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]


def stage(dest: pathlib.Path) -> None:
    """Copies the package sources into `dest`, keeping their relative paths.

    Args:
        dest: An empty directory.
    """
    for name in TOP_LEVEL:
        shutil.copy2(ROOT / name, dest / name)
    for package in packages():
        shutil.copytree(ROOT / package, dest / package, ignore=SKIP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--visibility", choices=["PRIVATE", "PUBLIC"], default="PRIVATE"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage and build the wheel, list its files, push nothing.",
    )
    parser.add_argument("push_args", nargs="*", help="Passed to prime env push.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="quant-research-hub-") as tmp:
        dest = pathlib.Path(tmp)
        stage(dest)
        if args.dry_run:
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", "dist"], cwd=dest, check=True
            )
            staged = sorted(
                p.relative_to(dest).as_posix()
                for p in dest.rglob("*")
                if p.is_file() and "dist" not in p.parts
            )
            print("\n".join(["Staged files:", *staged]))
            return
        subprocess.run(
            [
                "prime", "env", "push", "--plain",
                "--path", str(dest),
                "--visibility", args.visibility,
                *args.push_args,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
