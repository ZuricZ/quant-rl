"""Documentation lookups for agent tools: a library object's signature,
docstring and public members, for allowed packages only.

The lookup imports modules and reads attributes; it never evaluates the
name. Every part of the name must be public and not a test module, and
every object on the way must belong to an allowed package, so a lookup
cannot reach a package that was not allowed through a re-export, such as a
library's own `os` import. A property counts as its getter's, and a lookup
continues through it to the class its getter returns, so namespaces such
as polars' `Expr.str` can be browsed.
"""

import importlib
import inspect
import types
import typing
from collections.abc import Mapping
from typing import Any

# Parts that would import a library's test suite, which may run code.
_TEST_PARTS = frozenset({"tests", "testing", "conftest"})


def _package(obj: Any) -> str | None:
    """The top-level package an object belongs to, or None if it does not
    say (plain values, such as a float constant). A property belongs to its
    getter's package."""
    if isinstance(obj, property):
        obj = obj.fget
    name = obj.__name__ if isinstance(obj, types.ModuleType) else None
    name = name or getattr(obj, "__module__", None)
    return name.split(".")[0] if isinstance(name, str) else None


def _returned_class(prop: property) -> type | None:
    """The class a property's getter is annotated to return, if any."""
    try:
        returned = typing.get_type_hints(prop.fget).get("return")
    # Annotations that do not resolve, for any reason: nothing to follow.
    except Exception:  # noqa: BLE001
        return None
    return returned if inspect.isclass(returned) else None


def _signature(obj: Any) -> str:
    """The call signature, or "" for an object without one. A callable
    instance, such as polars' `col`, gets its `__call__` signature."""
    targets = [obj]
    if callable(obj) and not inspect.isclass(obj) and not inspect.isroutine(obj):
        targets.append(type(obj).__call__)
    for target in targets:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        if target is not obj:  # Drop `self`.
            params = list(signature.parameters.values())[1:]
            signature = signature.replace(parameters=params)
        return str(signature)
    return ""


def _step(obj: Any, path: list[str]) -> Any:
    """The next object along a name: an attribute of `obj`, or a submodule.

    Raises:
        LookupError: If there is neither.
    """
    holder = _returned_class(obj) if isinstance(obj, property) else obj
    if holder is None:
        raise LookupError(f"no {'.'.join(path)}")
    try:
        if inspect.isclass(holder):
            # From the class dict first, so a property stays a property.
            for klass in inspect.getmro(holder):
                if path[-1] in vars(klass):
                    return vars(klass)[path[-1]]
        return getattr(holder, path[-1])
    except AttributeError:
        pass
    # A library getter or lazy loader that fails, for any reason: not found.
    except Exception:  # noqa: BLE001
        raise LookupError(f"no {'.'.join(path)}") from None
    try:
        return importlib.import_module(".".join(path))
    # Importing a library module can fail in any way: not found.
    except Exception:  # noqa: BLE001
        raise LookupError(f"no {'.'.join(path)}") from None


def _members(obj: Any, allowed: set[str]) -> list[str]:
    """Public members that a lookup would accept: a module's `__all__` if it
    has one, else what `dir` lists, without leading underscores."""
    if isinstance(obj, property):
        obj = _returned_class(obj)
        if obj is None:
            return []
    names = getattr(obj, "__all__", None) if isinstance(obj, types.ModuleType) else None
    members = []
    for name in sorted(names or dir(obj)):
        if name.startswith("_") or name in _TEST_PARTS:
            continue
        try:
            member = inspect.getattr_static(obj, name)
        except AttributeError:
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if _package(member) in allowed:
            members.append(name)
    return members


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
        LookupError: If the name is not a public dotted name, names a test
            module, does not start with an allowed package, leaves the
            allowed packages or does not exist.
    """
    parts = name.strip().split(".")
    if not all(p.isidentifier() and not p.startswith("_") for p in parts):
        raise LookupError(f"not a public dotted name: {name!r}")
    if _TEST_PARTS.intersection(parts):
        raise LookupError("test modules cannot be looked up")
    allowed = set(packages.values())
    if parts[0] not in packages:
        raise LookupError(f"only {', '.join(sorted(allowed))} can be looked up")
    path = [packages[parts[0]]]
    obj: Any = importlib.import_module(path[0])
    for part in parts[1:]:
        path.append(part)
        obj = _step(obj, path)
        if _package(obj) not in allowed:
            raise LookupError(
                f"{'.'.join(path)} is not part of {', '.join(sorted(allowed))}"
            )
    return ".".join(path), obj


def lookup(name: str, packages: Mapping[str, str], limit: int) -> str:
    """Documents the object a dotted name refers to.

    A function or method gets its signature and docstring. A class, module
    or property gets its signature, if any, its docstring and the public
    members a lookup would accept (for a property, those of the class it
    returns), so the caller can look one of them up next.

    Args:
        name: A dotted name; see `resolve`.
        packages: Allowed packages by the names that may start a lookup; see
            `resolve`.
        limit: The most characters of signature and docstring returned; a
            longer text is cut there and says so. Member lists, names only,
            are given in full.

    Returns:
        The documentation.

    Raises:
        LookupError: If `resolve` refuses the name.
    """
    full, obj = resolve(name, packages)
    target = obj.fget if isinstance(obj, property) else obj
    text = f"{full}{_signature(target)}\n\n{inspect.getdoc(target) or '(no docstring)'}"
    if len(text) > limit:
        note = f"\n[cut at {limit} characters]"
        text = text[: limit - len(note)] + note
    browsable = (
        inspect.isclass(obj)
        or isinstance(obj, types.ModuleType)
        or (isinstance(obj, property) and _returned_class(obj) is not None)
    )
    if browsable:
        text += f"\n\nPublic members: {', '.join(_members(obj, set(packages.values())))}"
    return text
