"""Runs models from `runs/models.toml` on the taskset with `vf-run`.

Each model's tasks run in chunks of `--chunk` tasks, one `vf-run` per chunk,
written to `<out>/<model>/tasks-<first>-<last>/` (`traces.jsonl` and the
resolved configs). Rerunning the same command resumes: chunks whose episodes
all finished are skipped, and any other chunk is run again from scratch. vf-run
cannot resume a run itself, so a crash loses at most one chunk. Models run one
after another; episodes within a chunk go in parallel up to `--concurrency`.
Summarise the runs with `scripts/results.py <out>/<model>`.

The secret picks the datasets: pass the same one to every model and to
`scripts/run_baselines.py` so they score the same tasks.

The task image must exist where the runtime looks for it: build it with
`docker build -t quant-research:0.1.0 .` for `--runtime docker`, or push it
and pass `--image` for `--runtime prime`.

Usage:
    uv run python scripts/run_models.py --secret S qwen3_5-9b [...]
        [--split eval] [--tasks 30] [--rollouts 3] [--concurrency 4]
        [--chunk 4] [--runtime docker] [--out outputs/models]
        [--vf-config overrides.toml] [--dry-run]

`--vf-config` passes a TOML file of vf-run settings, for those its flags
cannot set, e.g. the check settings:

    [env.taskset.task.check_config]
    max_ic = 0.1
"""

import argparse
import json
import os
import shlex
import shutil
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


def chunk_dir(first: int, n: int) -> str:
    """Directory name of the chunk of `n` tasks from task `first`."""
    return f"tasks-{first:02d}-{first + n - 1:02d}"


def chunk_done(path: Path, episodes: int) -> bool:
    """Whether a chunk's run finished all its episodes without infrastructure
    errors.

    Args:
        path: The chunk's run directory.
        episodes: Episodes the chunk should hold: tasks times rollouts.

    Returns:
        True if its `traces.jsonl` holds that many episodes, all `ok`.
    """
    traces = path / "traces.jsonl"
    if not traces.exists():
        return False
    with traces.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return len(records) == episodes and all(r["ok"] for r in records)


def command(
    name: str, entry: dict[str, Any], args: argparse.Namespace, first: int, n: int
) -> list[str]:
    """The `vf-run` command for one chunk of one model's tasks.

    Args:
        name: The entry's name; also the run's name and its directory.
        entry: The model's registry entry.
        args: Parsed command-line arguments.
        first: First task of the chunk.
        n: Tasks in the chunk.

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
        "--select.skip",
        str(first),
        "--select.limit",
        str(n),
        "--num-rollouts",
        str(args.rollouts),
        "--max-concurrent",
        str(args.concurrency),
        "--output-dir",
        str(args.out / name),
        "--run.name",
        name,
        "--run.dir",
        chunk_dir(first, n),
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
    if args.vf_config:
        cmd += ["@", str(args.vf_config)]
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
    parser.add_argument("--chunk", type=int, default=4, help="tasks per vf-run")
    parser.add_argument("--runtime", choices=["docker", "prime"], default="docker")
    parser.add_argument("--image", help="task image reference; default the taskset's")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "models")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--vf-config", type=Path, help="TOML of vf-run settings")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = load_models(args.registry)
    unknown = [m for m in args.models if m not in models]
    if unknown:
        parser.error(f"not in {args.registry}: {', '.join(unknown)}")
    env = os.environ | {LOCAL_KEY_VAR: os.environ.get(LOCAL_KEY_VAR, "EMPTY")}
    incomplete = []
    for name in args.models:
        entry = models[name]
        key_var = entry.get("api_key_var")
        if key_var and key_var not in os.environ and not args.dry_run:
            sys.exit(f"{name}: set {key_var} to the endpoint's API key")
        for first in range(0, args.tasks, args.chunk):
            n = min(args.chunk, args.tasks - first)
            path = args.out / name / chunk_dir(first, n)
            if not args.dry_run and chunk_done(path, n * args.rollouts):
                print(f"{name} {path.name}: done, skipped", flush=True)
                continue
            shutil.rmtree(path, ignore_errors=True)
            cmd = command(name, entry, args, first, n)
            shown = ["***" if c == args.secret else c for c in cmd]
            print("$", shlex.join(shown), flush=True)
            subprocess.run(cmd, env=env, check=False)
            if not args.dry_run and not chunk_done(path, n * args.rollouts):
                incomplete.append(f"{name} {path.name}")
    if incomplete:
        sys.exit("incomplete, rerun to retry: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
