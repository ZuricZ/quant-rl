"""Documentation lookups for agent tools: a library object's signature,
docstring and public members, for allowed packages only.

The lookup imports modules and reads attributes; it never evaluates the
name. Every part of the name must be public, and every object on the way
must belong to an allowed package, so a lookup cannot reach a package that
was not allowed through a re-export, such as a library's own `os` import.
"""

import importlib
import inspect
import types
from collections.abc import Mapping
from typing import Any


def _package(obj: Any) -> str | None:
    """The top-level package an object belongs to, or None if it does not
    say (plain values, such as a float constant)."""
    name = obj.__name__ if isinstance(obj, types.ModuleType) else None
    name = name or getattr(obj, "__module__", None)
    return name.split(".")[0] if isinstance(name, str) else None


def _signature(obj: Any) -> str:
    """The call signature, or "" for an object without one."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _members(obj: Any) -> list[str]:
    """Public members: a module's `__all__` if it has one, else the names
    `dir` lists without a leading underscore."""
    names = getattr(obj, "__all__", None) if isinstance(obj, types.ModuleType) else None
    return sorted(n for n in (names or dir(obj)) if not n.startswith("_"))


def resolve(name: str, packages: Mapping[str, str]) -> tuple[str, Any]:
    """Finds the object a dotted name refers to.

    Args:
        name: A dotted name, such as "polars.Expr.rolling_mean"; its first
            part may be an alias, such as "pl".
        packages: Allowed top-level import names by the names that may
            start a lookup, aliases included, e.g. {"pl": "polars",
            "polars": "polars"}.

    Returns:
        The full name, with any alias replaced, and the object.

    Raises:
        LookupError: If the name is not a public dotted name, does not start
            with an allowed package, leaves the allowed packages or does not
            exist.
    """
    parts = name.strip().split(".")
    if not all(p.isidentifier() and not p.startswith("_") for p in parts):
        raise LookupError(f"not a public dotted name: {name!r}")
    allowed = set(packages.values())
    if parts[0] not in packages:
        raise LookupError(f"only {', '.join(sorted(allowed))} can be looked up")
    path = [packages[parts[0]]]
    obj: Any = importlib.import_module(path[0])
    for part in parts[1:]:
        path.append(part)
        try:
            obj = getattr(obj, part)
        except AttributeError:
            try:
                obj = importlib.import_module(".".join(path))
            except ImportError:
                raise LookupError(f"no {'.'.join(path)}") from None
        if _package(obj) not in allowed:
            raise LookupError(f"{'.'.join(path)} is not part of {', '.join(sorted(allowed))}")
    return ".".join(path), obj


def lookup(name: str, packages: Mapping[str, str], limit: int) -> str:
    """Documents the object a dotted name refers to.

    A function or method gets its signature and docstring. A class or module
    gets its signature, if any, the first paragraph of its docstring and its
    public members, so the caller can look one of them up next.

    Args:
        name: A dotted name; see `resolve`.
        packages: Allowed packages by the names that may start a lookup; see
            `resolve`.
        limit: The most characters returned; a longer text is cut there and
            says so.

    Returns:
        The documentation.

    Raises:
        LookupError: If `resolve` refuses the name.
    """
    full, obj = resolve(name, packages)
    doc = inspect.getdoc(obj) or "(no docstring)"
    if inspect.isclass(obj) or isinstance(obj, types.ModuleType):
        summary = doc.split("\n\n", 1)[0]
        text = (
            f"{full}{_signature(obj)}\n\n{summary}\n\n"
            f"Public members: {', '.join(_members(obj))}"
        )
    else:
        text = f"{full}{_signature(obj)}\n\n{doc}"
    if len(text) <= limit:
        return text
    note = f"\n[cut at {limit} characters]"
    return text[: limit - len(note)] + note
