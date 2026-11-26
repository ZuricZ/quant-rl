"""Runs models from `runs/models.toml` on a taskset with `vf-run`.

Each model's tasks run in chunks of `--chunk` tasks, one `vf-run` per chunk,
written to `<out>/<model>/tasks-<first>-<last>/` (`traces.jsonl` and the
resolved configs). Rerunning the same command resumes: chunks whose episodes
all finished are skipped, and any other chunk is run again from scratch. vf-run
cannot resume a run itself, so a crash loses at most one chunk. Models run one
after another; episodes within a chunk go in parallel up to `--concurrency`.
Summarise the runs with `scripts/results.py <out>/<model>`.

Each run directory holds a `manifest.json` of the settings that shape its
results: the registry entry, split, rollouts, chunk size, budget, image,
runtime, and hashes of the secret and of `--vf-config`. A resume under other
settings is refused rather than mixed into the finished chunks. `--tasks` may
grow between resumes. `--dry-run` validates the vf-run configs in a temporary
directory and leaves `--out` untouched.

The secret reaches vf-run in a private (0600) JSON file, deleted afterwards:
not in its arguments, where `ps` would show it, and not as TOML, which vf-run
copies into the run directory.

The secret picks the datasets: pass the same one to every model and to
`scripts/run_baselines.py` so they score the same tasks.

The task image must exist where the runtime looks for it: build it with
`docker build -t quant-research:0.1.1 .` for `--runtime docker`, or push it
and pass `--image` for `--runtime prime`.

Usage:
    uv run python scripts/run_models.py --secret S qwen3_5-9b [...]
        [--taskset quant-research] [--split eval] [--tasks 30] [--rollouts 3]
        [--concurrency 4]
        [--chunk 4] [--runtime docker] [--out outputs/models]
        [--budget N] [--vf-config overrides.toml] [--dry-run]

`--taskset quant-research-notools` runs the single-reply taskset instead,
the one prime-rl trains on, and names the run `<model>-notools`, so it never
shares a directory with the same model's main-taskset runs. Its episodes are
the same datasets as the main taskset's with the same secret and split.

`--budget N` sets the backtest budget (the taskset's default otherwise) and
names the run `<model>-budget<N>`, so runs of one model at several budgets
sit side by side; `results.py` pools them per model in its claim-gap plot.
The no-tools taskset has no backtests, so it takes no budget.

`--vf-config` passes a TOML file of vf-run settings, for those its flags
cannot set, e.g. the check settings:

    [env.taskset.task.check_config]
    max_ic = 0.1
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# vf-run from this interpreter's environment, so an unactivated venv works.
VF_RUN = str(Path(sys.executable).with_name("vf-run"))
REGISTRY = ROOT / "runs" / "models.toml"
# The main taskset first: the default.
TASKSETS = ("quant-research", "quant-research-notools")
# Sent as the key to local servers without auth.
LOCAL_KEY_VAR = "LOCAL_MODEL_API_KEY"


def load_models(path: Path = REGISTRY) -> dict[str, dict[str, Any]]:
    """The model entries of a registry file, by name."""
    with path.open("rb") as f:
        return tomllib.load(f)["models"]


MANIFEST = "manifest.json"


def run_name(name: str, budget: int | None, taskset: str = TASKSETS[0]) -> str:
    """The run's name.

    Args:
        name: The registry entry's name.
        budget: The backtest budget, or None for the taskset's default.
        taskset: The taskset id.

    Returns:
        `name`, with `-notools` appended for the no-tools taskset and
        `-budget<N>` when a budget is set.
    """
    if taskset != TASKSETS[0]:
        name = f"{name}-notools"
    return name if budget is None else f"{name}-budget{budget}"


def chunk_dir(first: int, n: int) -> str:
    """Directory name of the chunk of `n` tasks from task `first`."""
    return f"tasks-{first:02d}-{first + n - 1:02d}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """The settings that shape a run's results, for checking resumes.

    Args:
        entry: The model's registry entry.
        args: Parsed command-line arguments.

    Returns:
        A JSON-serialisable dict. The secret and the vf-config appear only
        as SHA-256 hashes.
    """
    return {
        "entry": entry,
        "split": args.split,
        "rollouts": args.rollouts,
        "chunk": args.chunk,
        "budget": args.budget,
        "image": args.image,
        "runtime": args.runtime,
        "secret_sha256": _sha256(args.secret.encode()),
        "vf_config_sha256": (
            _sha256(args.vf_config.read_bytes()) if args.vf_config else None
        ),
    }


def check_manifest(run_dir: Path, expected: dict[str, Any]) -> list[str]:
    """Settings that differ from those the run was started with.

    Args:
        run_dir: The run's directory.
        expected: This invocation's `manifest`.

    Returns:
        Names of the differing settings; empty if they all match or the run
        has no manifest yet.
    """
    path = run_dir / MANIFEST
    if not path.exists():
        return []
    recorded = json.loads(path.read_text())
    keys = recorded.keys() | expected.keys()
    return sorted(k for k in keys if recorded.get(k) != expected.get(k))


def secret_file(directory: Path, secret: str) -> Path:
    """Writes the secret as a vf-run config file readable only by its owner.

    JSON, not TOML: vf-run copies root `@` TOML files verbatim into the run's
    `configs/`, which would put the secret on disk next to the traces.

    Args:
        directory: A private directory to write into.
        secret: The taskset's secret.

    Returns:
        The file's path, for vf-run's `@` argument.
    """
    path = directory / "secret.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"env": {"taskset": {"secret": secret}}}, f)
    return path


def chunk_done(path: Path, episodes: int) -> bool:
    """Whether a chunk's run finished all its episodes without infrastructure
    errors.

    Args:
        path: The chunk's run directory.
        episodes: Episodes the chunk should hold: tasks times rollouts.

    Returns:
        True if its `traces.jsonl` holds that many episodes, all `ok`. False
        if a line does not parse, e.g. one cut short by a crash.
    """
    traces = path / "traces.jsonl"
    if not traces.exists():
        return False
    try:
        with traces.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError:
        return False
    return len(records) == episodes and all(r["ok"] for r in records)


def command(
    name: str,
    entry: dict[str, Any],
    args: argparse.Namespace,
    first: int,
    n: int,
    out: Path,
    secret: Path,
) -> list[str]:
    """The `vf-run` command for one chunk of one model's tasks.

    Args:
        name: The entry's name; with the taskset and budget, the run's name
            (`run_name`).
        entry: The model's registry entry.
        args: Parsed command-line arguments.
        first: First task of the chunk.
        n: Tasks in the chunk.
        out: Directory that holds the runs.
        secret: The file holding the secret (`secret_file`).

    Returns:
        The argument list.
    """
    cmd = [
        VF_RUN if Path(VF_RUN).exists() else "vf-run",
        args.taskset,
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
        str(out / run_name(name, args.budget, args.taskset)),
        "--run.name",
        run_name(name, args.budget, args.taskset),
        "--run.dir",
        chunk_dir(first, n),
        "--env.taskset.split",
        args.split,
        "--env.agent.runtime.type",
        args.runtime,
    ]
    for key, value in entry.get("sampling", {}).items():
        cmd += [f"--sampling.{key.replace('_', '-')}", str(value)]
    if args.image:
        cmd += ["--env.taskset.task.image", args.image]
    if args.budget is not None:
        cmd += ["--env.taskset.task.tools.budget", str(args.budget)]
    cmd += ["@", str(secret)]
    if args.vf_config:
        cmd += ["@", str(args.vf_config)]
    if args.dry_run:
        cmd += ["--dry-run", "True"]
    return cmd


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def main() -> None:
    """Parses arguments and runs each requested model in turn."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="entry names in the registry")
    parser.add_argument("--secret", required=True)
    parser.add_argument("--taskset", choices=TASKSETS, default=TASKSETS[0])
    parser.add_argument("--split", choices=["eval", "train"], default="eval")
    parser.add_argument("--tasks", type=_positive, default=30)
    parser.add_argument("--rollouts", type=_positive, default=3)
    parser.add_argument("--concurrency", type=_positive, default=4)
    parser.add_argument("--chunk", type=_positive, default=4, help="tasks per vf-run")
    parser.add_argument("--budget", type=int, help="backtests allowed per episode")
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
    if args.budget is not None and args.budget < 0:
        parser.error(f"--budget must be at least 0, got {args.budget}")
    if args.budget is not None and args.taskset != TASKSETS[0]:
        parser.error(f"--budget needs backtests, which {args.taskset} has none of")
    env = os.environ | {LOCAL_KEY_VAR: os.environ.get(LOCAL_KEY_VAR, "EMPTY")}
    incomplete = []
    with tempfile.TemporaryDirectory(prefix="run_models-") as private:
        private = Path(private)
        secret = secret_file(private, args.secret)
        out = private / "dry-run" if args.dry_run else args.out
        for name in args.models:
            entry = models[name]
            key_var = entry.get("api_key_var")
            if key_var and not os.environ.get(key_var) and not args.dry_run:
                sys.exit(f"{name}: set {key_var} to the endpoint's API key")
            run_dir = out / run_name(name, args.budget, args.taskset)
            expected = manifest(entry, args)
            differ = check_manifest(run_dir, expected)
            if differ:
                sys.exit(
                    f"{run_dir} was started with other settings ({', '.join(differ)});"
                    " use another --out or delete it"
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / MANIFEST).write_text(json.dumps(expected, indent=1))
            for first in range(0, args.tasks, args.chunk):
                n = min(args.chunk, args.tasks - first)
                path = run_dir / chunk_dir(first, n)
                if chunk_done(path, n * args.rollouts):
                    print(f"{name} {path.name}: done, skipped", flush=True)
                    continue
                shutil.rmtree(path, ignore_errors=True)
                cmd = command(name, entry, args, first, n, out, secret)
                print("$", shlex.join(cmd), flush=True)
                done = subprocess.run(cmd, env=env, check=False)
                if args.dry_run:
                    if done.returncode:
                        incomplete.append(f"{name} {path.name} (vf-run rejected it)")
                elif not chunk_done(path, n * args.rollouts):
                    incomplete.append(f"{name} {path.name}")
        if args.dry_run:
            print(
                f"dry run: configs checked in {out}, now deleted; {args.out} untouched"
            )
    if incomplete:
        sys.exit("incomplete, rerun to retry: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
