"""Puts the shared core, `quantenv`, into this environment's wheel.

In the repository `quantenv` is two folders up; in this environment's source
distribution, which verifiers installs into a sandbox, it sits next to this
file (`[tool.hatch.build.targets.sdist.force-include]`). Static wheel
settings name one path, so this hook picks whichever exists (spec,
Decision 28).
"""

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class QuantenvHook(BuildHookInterface):
    """Adds `quantenv` to the wheel from the repository or the sdist."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Force-includes the first `quantenv` folder found.

        Args:
            version: The wheel version being built ("standard" or "editable").
            build_data: Hatchling's build data; gains a `force_include` entry.

        Raises:
            FileNotFoundError: Neither location holds `quantenv`.
        """
        root = Path(self.root)
        for source in (root / "quantenv", root.parents[1] / "quantenv"):
            if (source / "__init__.py").is_file():
                build_data["force_include"][str(source)] = "quantenv"
                return
        raise FileNotFoundError(f"no quantenv next to or two folders above {root}")
