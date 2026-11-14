"""Scores one submission on the hidden periods.

The task's reward runs this inside the rollout's runtime, after the agent has
finished:

    python -m quant_research.grade REQUEST RESULT

REQUEST is a JSON file with `seed`, `code` and `claimed_ic`. The score is
written as JSON to RESULT rather than stdout, which the agent's code can
print to. The hidden periods are built only inside the scoring worker that
`quantenv.scoring` starts, never in this process.
"""

import dataclasses
import functools
import json
import sys
from pathlib import Path

from quantenv import scoring
from quantenv.sources import synthetic


def main(argv: list[str]) -> None:
    """Reads a request, scores it and writes the result.

    Args:
        argv: The request path and the result path.
    """
    request_path, result_path = argv
    request = json.loads(Path(request_path).read_text())
    score = scoring.score_submission(
        request["code"],
        request["claimed_ic"],
        functools.partial(synthetic.hidden_data, request["seed"]),
    )
    Path(result_path).write_text(json.dumps(dataclasses.asdict(score)))


if __name__ == "__main__":
    main(sys.argv[1:])
