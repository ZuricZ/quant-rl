"""The Signal base class and compilation of agent code into a subclass."""

from typing import Any

import polars as pl


class SignalError(Exception):
    """Signal code failed to compile, raised, or returned a bad value."""


class SignalTimeout(SignalError):
    """Signal code ran past its time limit."""


class Signal:
    """Base class for a submission. Subclasses override what they need.

    The environment calls `features` once on the whole split, then, for each
    walk-forward block, makes a fresh instance, calls `fit` on past rows and
    `predict` on the block.
    """

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


def compile_signal(code: str) -> type[Signal]:
    """Executes signal code and returns the Signal subclass it defines.

    Args:
        code: Python source defining exactly one subclass of `Signal`.
            `pl` and `Signal` are predefined.

    Returns:
        The subclass.

    Raises:
        SignalError: If the code fails to run or does not define exactly one
            subclass of `Signal`.
    """
    namespace: dict[str, Any] = {"pl": pl, "Signal": Signal, "__name__": "signal_code"}
    try:
        exec(compile(code, "<signal>", "exec"), namespace)
    except Exception as e:
        raise SignalError(f"code failed to run: {e!r}") from e
    found = [
        v
        for v in namespace.values()
        if isinstance(v, type) and issubclass(v, Signal) and v is not Signal
    ]
    if len(found) != 1:
        raise SignalError(
            f"code must define exactly one subclass of Signal, found {len(found)}"
        )
    return found[0]
