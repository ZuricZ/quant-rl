"""Checks, inside the task's runtime, that agent code can be sandboxed.

`Task.setup` runs this in the rollout's container before the agent starts:

    python -m quant_research.preflight USER

It exits non-zero, with the reason on stderr, unless USER exists and this
process is root, which `quantenv.sandbox` needs to run agent code as USER.
"""

import os
import pwd
import sys


def main(argv: list[str]) -> None:
    """Exits with an error unless the sandbox user can be used.

    Args:
        argv: The sandbox user's name.
    """
    (user,) = argv
    try:
        pwd.getpwnam(user)
    except KeyError:
        sys.exit(f"no user {user!r} in this runtime")
    if os.geteuid() != 0:
        sys.exit("not root, so agent code cannot run as another user")


if __name__ == "__main__":
    main(sys.argv[1:])
