"""Grading a submission inside the task's runtime, from the host.

Runs `quantenv.grade` and `quantenv.preflight` with the task image's own
Python. One of the two `quantenv` modules that import verifiers (with
`quantenv.taskset`).
"""

import dataclasses
import json
import uuid
from typing import Any

import verifiers.v1 as vf

from quantenv import checks, scoring
from quantenv.sources import synthetic
from quantenv.taskset import run_rows

# User that runs agent code in the sandbox; see `quantenv.sandbox`.
SANDBOX_USER = "nobody"


def scoring_timeout(
    check_config: checks.CheckConfig,
    config: synthetic.Config = synthetic.DEFAULT_CONFIG,
) -> float:
    """Seconds verifiers allows the grader.

    Every pipeline run at the scored run's limit, plus start-up for the
    container and the scoring worker. Counts the column dependence run even
    when the data has no suspects, so it bounds both cases.

    Args:
        check_config: Settings of the checks; uses `n_cuts`.
        config: The task's generator parameters.

    Returns:
        The timeout in seconds.
    """
    runs = 2 + check_config.n_cuts
    seconds = scoring.TimeLimit().seconds(run_rows(hidden=True, config=config))
    return runs * seconds + 3 * scoring.STARTUP_ALLOWANCE


async def grade(
    user: str | None,
    runtime: vf.Runtime,
    seed: int,
    code: str,
    claimed_sharpe: float | None,
    check_config: checks.CheckConfig = checks.CheckConfig(),
    variation: int | None = None,
) -> dict[str, Any]:
    """Scores a submission on the hidden periods inside `runtime`.

    The grader runs with the task image's own `python`, which has this
    package installed (see the repository's Dockerfile).

    Args:
        user: The sandbox user that runs the alpha; see `ResearchConfig`.
        runtime: A runtime of the task image.
        seed: Dataset seed.
        code: Alpha code.
        claimed_sharpe: The claimed net Sharpe on unseen data.
        check_config: Settings of the lookahead checks.
        variation: The task's variation seed, or None for the default
            market (`synthetic.task_config`).

    Returns:
        The fields of `quantenv.scoring.Score`.

    Raises:
        RuntimeError: If the grader itself fails. Bad alpha code does not
            raise; it scores as invalid.
    """
    stem = f"/tmp/vf-grade-{uuid.uuid4().hex}"
    request = {
        "seed": seed,
        "code": code,
        "claimed_sharpe": claimed_sharpe,
        "checks": dataclasses.asdict(check_config),
        "user": user,
        "variation": variation,
    }
    await runtime.write(f"{stem}.json", json.dumps(request).encode())
    result = await runtime.run(
        ["python", "-m", "quantenv.grade", f"{stem}.json", f"{stem}.out.json"],
        {},
    )
    if result.exit_code != 0:
        raise RuntimeError(f"grader failed: {(result.stderr or result.stdout)[-2000:]}")
    return json.loads(await runtime.read(f"{stem}.out.json"))


async def check_sandbox_user(runtime: vf.Runtime, user: str | None) -> None:
    """Checks that the runtime can run code as the sandbox user.

    Args:
        runtime: A runtime of the task image.
        user: The sandbox user; None skips the check.

    Raises:
        RuntimeError: If the runtime cannot run code as `user`.
    """
    if user is None:
        return
    check = await runtime.run(["python", "-m", "quantenv.preflight", user], {})
    if check.exit_code != 0:
        raise RuntimeError(
            f"sandbox user {user!r} cannot be used: "
            f"{(check.stderr or check.stdout).strip()[-500:]}"
        )
