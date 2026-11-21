"""Scores one submission on the hidden periods.

The task's reward runs this inside the rollout's runtime, after the agent has
finished:

    python -m quantenv.grade REQUEST RESULT

REQUEST is a JSON file with `seed`, `code`, `claimed_sharpe`, `user` (the
sandbox user), `checks` (a `quantenv.checks.CheckConfig` as a dict) and
`variation` (the task's variation seed, or null for the default market). It
holds the dataset seed, so it is deleted as soon as it is read, before any
agent code runs. The raw score (`scoring.Score`, no reward) is written as
JSON to RESULT; the host applies the reward rule (`quantenv.reward`). The
hidden periods are built only inside the scoring worker that
`quantenv.scoring` starts, never in this process.
"""

import dataclasses
import functools
import json
import sys
from pathlib import Path

from quantenv import checks, scoring
from quantenv.sources import synthetic


def main(argv: list[str]) -> None:
    """Reads a request, scores it and writes the result.

    Args:
        argv: The request path and the result path.
    """
    request_path, result_path = argv
    request_file = Path(request_path)
    request = json.loads(request_file.read_text())
    request_file.unlink()
    score = scoring.score_submission(
        request["code"],
        request["claimed_sharpe"],
        functools.partial(
            synthetic.hidden_data,
            request["seed"],
            synthetic.task_config(request["variation"]),
        ),
        user=request["user"],
        check_config=checks.CheckConfig(**request["checks"]),
    )
    Path(result_path).write_text(json.dumps(dataclasses.asdict(score)))


if __name__ == "__main__":
    main(sys.argv[1:])
