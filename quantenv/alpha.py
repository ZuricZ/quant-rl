"""The Alpha base class and compilation of agent code into a subclass."""

from typing import Any

import numpy as np
import polars as pl


class AlphaError(Exception):
    """Alpha code failed to compile, raised, or returned a bad value."""


class AlphaTimeout(AlphaError):
    """Alpha code ran past its time limit."""


class Alpha:
    """Base class for a submission. Subclasses override what they need.

    The environment calls `features` once on the whole panel, then, for each
    walk-forward block, makes a fresh instance, calls `fit` on past rows and
    `predict` on the block.

    Attributes:
        rng: Random generator for the current step, set by the environment
            before each of `features`, `fit` and `predict` and seeded from the
            step, so every rerun draws the same numbers. Draw randomness from
            it rather than from global generators.
    """

    rng: np.random.Generator

    def features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Feature engineering.

        Args:
            df: Observable columns, all rows, sorted by timestamp.

        Returns:
            The same rows in the same order, with any added columns. A value
            at timestamp t may only use rows at or before t.
        """
        return df

    def fit(self, train: pl.DataFrame) -> None:
        """Modelling. Keep fitted state on `self`.

        Args:
            train: Engineered rows before the block, plus the target, which
                is realised for every one of them.
        """

    def predict(self, df: pl.DataFrame) -> pl.Series | pl.Expr:
        """Scores one block.

        Args:
            df: Engineered rows of the block.

        Returns:
            One score per row in row order, as a Series or as an expression
            evaluated on `df`.
        """
        raise NotImplementedError


def compile_alpha(code: str) -> type[Alpha]:
    """Executes alpha code and returns the Alpha subclass it defines.

    Args:
        code: Python source defining exactly one subclass of `Alpha`.
            `pl` and `Alpha` are predefined.

    Returns:
        The subclass.

    Raises:
        AlphaError: If the code fails to run or does not define exactly one
            subclass of `Alpha`.
    """
    namespace: dict[str, Any] = {"pl": pl, "Alpha": Alpha, "__name__": "alpha_code"}
    try:
        exec(compile(code, "<alpha>", "exec"), namespace)
    except Exception as e:
        raise AlphaError(f"code failed to run: {e!r}") from e
    found = [
        v
        for v in namespace.values()
        if isinstance(v, type) and issubclass(v, Alpha) and v is not Alpha
    ]
    if len(found) != 1:
        raise AlphaError(
            f"code must define exactly one subclass of Alpha, found {len(found)}"
        )
    return found[0]
