"""Runs models from `runs/models.toml` on the taskset with `vf-run`.

Each model gets one `vf-run` run, written to `<out>/<model>/` (`traces.jsonl`
and the resolved configs). Models run one after another; episodes within a
run go in parallel up to `--concurrency`. Summarise the runs with
`scripts/results.py`.

The secret picks the datasets: pass the same one to every model and to
`scripts/run_baselines.py` so they score the same tasks.

The task image must exist where the runtime looks for it: build it with
`docker build -t quant-research:0.1.0 .` for `--runtime docker`, or push it
and pass `--image` for `--runtime prime`.

Usage:
    uv run python scripts/run_models.py --secret S qwen3-8b [qwen3-14b ...]
        [--split eval] [--tasks 30] [--rollouts 3] [--concurrency 4]
        [--runtime docker] [--out outputs/models] [--dry-run]
"""

import argparse
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "runs" / "models.toml"
TASKSET = "quant-research"
# Sent as the key to local servers without auth.
LOCAL_KEY_VAR = "LOCAL_MODEL_API_KEY"


def load_models(path: Path = REGISTRY) -> dict[str, dict[str, Any]]:
    """The model entries of a registry file, by name."""
    with path.open("rb") as f:
        return tomllib.load(f)["models"]


def command(name: str, entry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """The `vf-run` command for one model.

    Args:
        name: The entry's name; also the run's name and directory.
        entry: The model's registry entry.
        args: Parsed command-line arguments.

    Returns:
        The argument list.
    """
    cmd = [
        "vf-run",
        TASKSET,
        "--model",
        entry["model"],
        "--client.base-url",
        entry["base_url"],
        "--client.api-key-var",
        entry.get("api_key_var", LOCAL_KEY_VAR),
        "--select.limit",
        str(args.tasks),
        "--num-rollouts",
        str(args.rollouts),
        "--max-concurrent",
        str(args.concurrency),
        "--output-dir",
        str(args.out),
        "--run.name",
        name,
        "--env.taskset.secret",
        args.secret,
        "--env.taskset.split",
        args.split,
        "--env.agent.runtime.type",
        args.runtime,
    ]
    for key, value in entry.get("sampling", {}).items():
        cmd += [f"--sampling.{key.replace('_', '-')}", str(value)]
    if args.image:
        cmd += ["--env.taskset.task.image", args.image]
    if args.dry_run:
        cmd += ["--dry-run", "True"]
    return cmd


def main() -> None:
    """Parses arguments and runs each requested model in turn."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="entry names in the registry")
    parser.add_argument("--secret", required=True)
    parser.add_argument("--split", choices=["eval", "train"], default="eval")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--runtime", choices=["docker", "prime"], default="docker")
    parser.add_argument("--image", help="task image reference; default the taskset's")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "models")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = load_models(args.registry)
    unknown = [m for m in args.models if m not in models]
    if unknown:
        parser.error(f"not in {args.registry}: {', '.join(unknown)}")
    env = os.environ | {LOCAL_KEY_VAR: os.environ.get(LOCAL_KEY_VAR, "EMPTY")}
    for name in args.models:
        entry = models[name]
        key_var = entry.get("api_key_var")
        if key_var and key_var not in os.environ and not args.dry_run:
            sys.exit(f"{name}: set {key_var} to the endpoint's API key")
        cmd = command(name, entry, args)
        shown = ["***" if c == args.secret else c for c in cmd]
        print("$", shlex.join(shown), flush=True)
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
