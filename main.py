#!/usr/bin/env python3
"""Reflects a Python module into a context-free grammar of typed calls."""

from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import itertools
import keyword
import re
import sys
import types
import typing
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PARAMETERIZED_NAME_HASH_CHARS = 5
START_TYPE_NAME = "START"
DEFAULT_NUMPY_GROUND_DTYPES = (
    "bool_",
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
_BUILTIN_TYPE_NAMES = frozenset(
    {
        "object",
        "bool",
        "int",
        "float",
        "complex",
        "str",
        "bytes",
        "list",
        "tuple",
        "dict",
        "set",
        "slice",
    }
)


def _token(value: str) -> str:
    """Render one terminal/nonterminal atom without embedded whitespace."""

    return re.sub(r"\s+", "_", value)


def _sanitized_part(value: str) -> str:
    rendered = "".join(character if character.isalnum() else "_" for character in value).strip("_")
    return rendered or "X"


def _is_valid_module_alias(value: str) -> bool:
    return value.isidentifier() and not keyword.iskeyword(value)


@dataclass(frozen=True)
class TypeExpr:
    """A rendered type constructor or a type variable."""

    name: str
    arguments: tuple[TypeExpr, ...] = ()
    is_variable: bool = False

    @staticmethod
    def applied(name: str, *arguments: TypeExpr) -> TypeExpr:
        return TypeExpr(name=_token(name), arguments=tuple(arguments))

    @staticmethod
    def variable(name: str) -> TypeExpr:
        return TypeExpr(name=_token(name), is_variable=True)

    def render(self) -> str:
        rendered_name = _token(self.name)
        if not self.is_variable and self.name in _BUILTIN_TYPE_NAMES:
            # Keep the line format unambiguous when a builtin is both a Python
            # call token (``int(...)``) and a grammar nonterminal.
            rendered_name = f"builtins.{rendered_name}"
        if self.is_variable or not self.arguments:
            return rendered_name
        return f"{rendered_name}<{','.join(argument.render() for argument in self.arguments)}>"

    def variables(self) -> frozenset[str]:
        if self.is_variable:
            return frozenset((self.name,))
        return frozenset().union(*(argument.variables() for argument in self.arguments)) if self.arguments else frozenset()

    def depth(self) -> int:
        if self.is_variable or not self.arguments:
            return 0
        return 1 + max(argument.depth() for argument in self.arguments)

    def substitute(self, substitutions: Mapping[str, TypeExpr]) -> TypeExpr | None:
        if self.is_variable:
            return substitutions.get(self.name)
        substituted = tuple(argument.substitute(substitutions) for argument in self.arguments)
        if any(argument is None for argument in substituted):
            return None
        return TypeExpr.applied(self.name, *typing.cast(tuple[TypeExpr, ...], substituted))

    def rename_variables(self, replacements: Mapping[str, TypeExpr]) -> TypeExpr:
        if self.is_variable:
            return replacements.get(self.name, self)
        return TypeExpr.applied(self.name, *(argument.rename_variables(replacements) for argument in self.arguments))

    def walk(self) -> Iterator[TypeExpr]:
        yield self
        for argument in self.arguments:
            yield from argument.walk()


START_TYPE = TypeExpr.applied(START_TYPE_NAME)
STATEMENT_TYPE = TypeExpr.applied("__STATEMENT")
OBJECT_TYPE = TypeExpr.applied("object")
NONE_TYPE = TypeExpr.applied("NoneType")


@dataclass(frozen=True)
class TypeSymbol:
    type: TypeExpr

    def render(self) -> str:
        return self.type.render()


@dataclass(frozen=True)
class Token:
    value: str

    def render(self) -> str:
        return _token(self.value)


Symbol = TypeSymbol | Token


@dataclass(frozen=True)
class Production:
    lhs: TypeExpr
    rhs: tuple[Symbol, ...]

    def __init__(self, lhs: TypeExpr, rhs: Iterable[Symbol]):
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", tuple(rhs))

    def render(self) -> str:
        return f"{self.lhs.render()} -> {' '.join(symbol.render() for symbol in self.rhs)}"

    def types(self) -> frozenset[TypeExpr]:
        result = {self.lhs}
        result.update(symbol.type for symbol in self.rhs if isinstance(symbol, TypeSymbol))
        return frozenset(result)


@dataclass(frozen=True)
class GeneratedGrammar:
    text: str
    production_count: int
    nonterminal_count: int
    terminal_count: int

    @staticmethod
    def from_productions(productions: Iterable[Production]) -> GeneratedGrammar:
        unique = set(productions)
        ordered = sorted(
            unique,
            key=lambda production: (
                production.lhs.render(),
                " ".join(symbol.render() for symbol in production.rhs),
            ),
        )
        nonterminals: set[TypeExpr] = set()
        terminals: set[str] = set()
        for production in ordered:
            nonterminals.add(production.lhs)
            for symbol in production.rhs:
                if isinstance(symbol, TypeSymbol):
                    nonterminals.add(symbol.type)
                else:
                    terminals.add(symbol.value)
        return GeneratedGrammar(
            text="\n".join(production.render() for production in ordered),
            production_count=len(ordered),
            nonterminal_count=len(nonterminals),
            terminal_count=len(terminals),
        )


@dataclass(frozen=True)
class CommandLineArguments:
    module_name: str
    normalize_chomsky_normal_form: bool = False
    emit_parameterized_signatures: bool = False
    module_alias: str | None = None

    @staticmethod
    def parse(args: Sequence[str]) -> CommandLineArguments:
        positional: list[str] = []
        cnf = False
        parameterized = False
        module_alias: str | None = None
        for argument in args:
            if argument == "--cnf":
                cnf = True
            elif argument == "--parameterized":
                parameterized = True
            elif argument.startswith("--alias="):
                module_alias = argument.partition("=")[2]
                if not _is_valid_module_alias(module_alias):
                    raise ValueError("--alias must be a non-keyword Python identifier")
            elif argument.startswith("-"):
                raise ValueError(f"Unknown flag: {argument}")
            else:
                positional.append(argument)
        if cnf and parameterized:
            raise ValueError("--cnf and --parameterized are mutually exclusive")
        if len(positional) != 1:
            raise ValueError("Expected exactly one positional argument: <package>")
        return CommandLineArguments(positional[0], cnf, parameterized, module_alias)


def default_output_file(
    module_name: str,
    normalize_chomsky_normal_form: bool = False,
    emit_parameterized_signatures: bool = False,
) -> Path:
    if normalize_chomsky_normal_form:
        suffix = "cnf"
    elif emit_parameterized_signatures:
        suffix = "parameterized.cfg"
    else:
        suffix = "cfg"
    return Path("gen", f"{module_name.replace('.', '_')}.{suffix}")


@dataclass(frozen=True)
class LibraryProfile:
    """Declarative naming and optional reflection extension for one library.

    The reflection core only needs to know how a module is imported and how
    its qualified names should be rendered.  Library-specific typing oracles
    are selected by ``extension`` and can be added without changing the core
    scanner, signature renderer, or grammar machinery.
    """

    module_name: str
    namespace_module: str
    alias: str | None = None
    extension: str | None = None
    array_type_name: str | None = None
    ground_type_names: tuple[str, ...] = ()

    @staticmethod
    def for_module(
        module_name: str,
        *,
        alias: str | None = None,
        ground_type_names: tuple[str, ...] | None = None,
    ) -> LibraryProfile:
        if alias is not None and not _is_valid_module_alias(alias):
            raise ValueError("module_alias must be a non-keyword Python identifier")
        root_name = module_name.split(".", 1)[0]
        if root_name == "numpy":
            return LibraryProfile(
                module_name=module_name,
                namespace_module=root_name,
                alias="np" if alias is None else alias,
                extension="numpy",
                array_type_name="numpy.ndarray",
                ground_type_names=(
                    DEFAULT_NUMPY_GROUND_DTYPES
                    if ground_type_names is None
                    else ground_type_names
                ),
            )
        return LibraryProfile(
            module_name=module_name,
            namespace_module=module_name if alias is not None else root_name,
            alias=alias,
            ground_type_names=ground_type_names or (),
        )

    def render_qualified_name(self, qualified_name: str) -> str:
        if self.alias is None:
            return qualified_name
        if qualified_name == self.namespace_module:
            return self.alias
        prefix = f"{self.namespace_module}."
        if qualified_name.startswith(prefix):
            return f"{self.alias}.{qualified_name[len(prefix):]}"
        return qualified_name

    def import_qualified_name(self, rendered_name: str) -> str:
        if self.alias is None:
            return rendered_name
        if rendered_name == self.alias:
            return self.namespace_module
        prefix = f"{self.alias}."
        if rendered_name.startswith(prefix):
            return f"{self.namespace_module}.{rendered_name[len(prefix):]}"
        return rendered_name

    def module_access_parts(self, module_name: str, member_name: str) -> tuple[str, ...]:
        if self.alias is None:
            return (member_name,)
        namespace_parts = self.namespace_module.split(".")
        module_parts = module_name.split(".")
        suffix = (
            module_parts[len(namespace_parts) :]
            if module_parts[: len(namespace_parts)] == namespace_parts
            else []
        )
        return (self.alias, *suffix, member_name)

    def root_access_parts(self, member_name: str) -> tuple[str, ...]:
        root = self.alias or self.namespace_module
        return (root, member_name)


@dataclass(frozen=True)
class GeneratorOptions:
    module_name: str
    monomorphization_depth: int = 2
    max_type_arguments_per_variable: int = 26
    max_type_variables_per_callable: int = 2
    max_vararg_arity: int = 3
    max_array_literal_values: int = 3
    normalize_chomsky_normal_form: bool = False
    emit_parameterized_signatures: bool = False
    module_alias: str | None = None
    ground_type_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExportedMember:
    name: str
    value: object


class PythonModuleScanner:
    """Safely enumerate the requested module's directly exported namespace."""

    @staticmethod
    def exported_members(module: types.ModuleType) -> list[ExportedMember]:
        explicit = {
            name
            for name in getattr(module, "__all__", ())
            if isinstance(name, str)
        }
        names = explicit | {name for name in dir(module) if not name.startswith("_")}
        members: list[ExportedMember] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in sorted(names):
                try:
                    value = getattr(module, name)
                except BaseException:
                    # Lazy/deprecated module attributes are allowed to fail
                    # independently; one bad export must not hide its siblings.
                    continue
                members.append(ExportedMember(name, value))
        return members

    @classmethod
    def public_callables(cls, module: types.ModuleType) -> list[ExportedMember]:
        return [member for member in cls.exported_members(module) if callable(member.value)]


def module_public_callables(module: types.ModuleType) -> list[ExportedMember]:
    return PythonModuleScanner.public_callables(module)


@dataclass(frozen=True)
class ReflectedParameter:
    name: str
    type: TypeExpr
    kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD


@dataclass(frozen=True)
class SignatureSpec:
    parameters: tuple[ReflectedParameter, ...]
    return_type: TypeExpr
    domains: Mapping[str, tuple[TypeExpr, ...]] = field(default_factory=dict, compare=False, hash=False)
    bounds: Mapping[str, TypeExpr] = field(default_factory=dict, compare=False, hash=False)
    variable_labels: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)


_BINARY_OPERATOR_TOKENS = {
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__matmul__": "@",
    "__div__": "/",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
}
_REFLECTED_OPERATOR_TOKENS = {f"__r{name[2:]}": token for name, token in _BINARY_OPERATOR_TOKENS.items() if name not in {"__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__"}}
_REFLECTED_OPERATOR_TOKENS.update({"__rdivmod__": "divmod"})
_INPLACE_OPERATOR_TOKENS = {
    f"__i{name[2:]}": f"{token}="
    for name, token in _BINARY_OPERATOR_TOKENS.items()
    if name not in {"__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__"}
}
_UNARY_OPERATOR_TOKENS = {
    "__neg__": "-",
    "__pos__": "+",
    "__invert__": "~",
    "__abs__": "abs",
}
_UFUNC_OPERATOR_TOKENS = {
    "absolute": "abs",
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "matmul": "@",
    "true_divide": "/",
    "floor_divide": "//",
    "remainder": "%",
    "power": "**",
    "left_shift": "<<",
    "right_shift": ">>",
    "bitwise_and": "&",
    "bitwise_xor": "^",
    "bitwise_or": "|",
    "invert": "~",
    "negative": "-",
    "positive": "+",
    "less": "<",
    "less_equal": "<=",
    "equal": "==",
    "not_equal": "!=",
    "greater": ">",
    "greater_equal": ">=",
    "divmod": "divmod",
}
_UNARY_UFUNC_OPERATORS = {"absolute", "invert", "negative", "positive"}
_UFUNC_INPLACE_TOKENS = {
    name: f"{token}="
    for name, token in _UFUNC_OPERATOR_TOKENS.items()
    if token in {"+", "-", "*", "@", "/", "//", "%", "**", "<<", ">>", "&", "^", "|"}
}
_NUMPY_UFUNC_BACKED_DUNDERS = frozenset(
    {
        *_BINARY_OPERATOR_TOKENS,
        *_REFLECTED_OPERATOR_TOKENS,
        *_INPLACE_OPERATOR_TOKENS,
        *_UNARY_OPERATOR_TOKENS,
        "__divmod__",
    }
)

_NUMPY_LITERAL_TERMINALS: Mapping[str, tuple[str, ...]] = {
    "BOOL": ("False", "True"),
    "SIGNED": ("-1", "0", "1"),
    "UNSIGNED": ("0", "1", "2"),
    "FLOAT": ("-1.0", "0.0", "1.0"),
    "COMPLEX": ("-1j", "0j", "1j"),
    "DATETIME": ('"2000-01-01"', '"2000-01-02"', '"2000-01-03"'),
    "TIMEDELTA": ("0", "1", "2"),
    "STRING": ('""', '"a"', '"b"'),
    "BYTES": ('b""', 'b"a"', 'b"b"'),
    "OBJECT": ("0", "1", "2"),
}

_NUMPY_KIND_LITERAL_CATEGORY: Mapping[str, str] = {
    "b": "BOOL",
    "i": "SIGNED",
    "u": "UNSIGNED",
    "f": "FLOAT",
    "c": "COMPLEX",
    "M": "DATETIME",
    "m": "TIMEDELTA",
    "U": "STRING",
    "S": "BYTES",
    "O": "OBJECT",
    "V": "BYTES",
}
_PROTOCOL_DUNDERS = {
    "__array__",
    "__array_finalize__",
    "__array_function__",
    "__array_prepare__",
    "__array_ufunc__",
    "__array_wrap__",
    "__bytes__",
    "__bool__",
    "__call__",
    "__ceil__",
    "__complex__",
    "__contains__",
    "__copy__",
    "__deepcopy__",
    "__delitem__",
    "__dlpack__",
    "__dlpack_device__",
    "__divmod__",
    "__enter__",
    "__exit__",
    "__float__",
    "__floor__",
    "__format__",
    "__getitem__",
    "__hash__",
    "__index__",
    "__int__",
    "__iter__",
    "__len__",
    "__next__",
    "__repr__",
    "__reversed__",
    "__round__",
    "__setitem__",
    "__str__",
    "__trunc__",
}

# Public so completeness tests and downstream consumers can use exactly the
# same definition of "operator" as the generator.
OPERATOR_SPECS: Mapping[str, str] = {
    **_BINARY_OPERATOR_TOKENS,
    **_REFLECTED_OPERATOR_TOKENS,
    **_INPLACE_OPERATOR_TOKENS,
    **_UNARY_OPERATOR_TOKENS,
    **{name: name for name in sorted(_PROTOCOL_DUNDERS)},
}


def _split_top_level(value: str, separator: str = ",") -> list[str]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    closing = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in closing:
            stack.append(closing[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character == separator and not stack:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return [part for part in result if part]


def _balanced_doc_signature(documentation: str | None) -> str | None:
    if not documentation:
        return None
    lines = documentation.splitlines()
    candidate: list[str] = []
    depth = 0
    started = False
    for raw_line in lines[:12]:
        line = raw_line.strip()
        if not line and not started:
            continue
        if not started:
            if "(" not in line:
                return None
            started = True
        candidate.append(line)
        depth += line.count("(") - line.count(")")
        if started and depth <= 0:
            break
    rendered = " ".join(candidate)
    return rendered if started and "(" in rendered and ")" in rendered else None


def _doc_fields(documentation: str | None, section: str) -> list[tuple[str, str]]:
    if not documentation:
        return []
    lines = documentation.splitlines()
    start = None
    for index in range(len(lines) - 1):
        if lines[index].strip() == section and set(lines[index + 1].strip()) == {"-"}:
            start = index + 2
            break
    if start is None:
        return []
    result: list[tuple[str, str]] = []
    for index in range(start, len(lines)):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            continue
        if index + 1 < len(lines) and set(lines[index + 1].strip()) == {"-"} and raw == raw.lstrip():
            break
        if raw != raw.lstrip():
            continue
        if ":" in stripped:
            name, documented_type = stripped.split(":", 1)
            result.append((name.strip(), documented_type.strip()))
        elif section in {"Returns", "Yields"}:
            result.append(("", stripped))
    return result


class TypeRenderer:
    def __init__(
        self,
        module: types.ModuleType,
        members: Sequence[ExportedMember],
        profile: LibraryProfile | None = None,
    ):
        self.module = module
        self.root_name = module.__name__.split(".", 1)[0]
        self.profile = profile or LibraryProfile.for_module(module.__name__)
        aliases: dict[type, list[str]] = {}
        for member in members:
            if inspect.isclass(member.value):
                aliases.setdefault(typing.cast(type, member.value), []).append(member.name)
        self.aliases: dict[type, str] = {}
        for type_class, names in aliases.items():
            class_name = getattr(type_class, "__name__", "")
            self.aliases[type_class] = next((name for name in sorted(names) if name == class_name), sorted(names)[0])
        self._classes_by_rendered_name: dict[str, type] = {
            builtin.__qualname__: builtin
            for builtin in (object, bool, int, float, complex, str, bytes, list, tuple, dict, set, slice)
        }
        for type_class in self.aliases:
            self._classes_by_rendered_name[self.render_class(type_class).name] = type_class

    def render_class(self, type_class: type) -> TypeExpr:
        if type_class is type(None):
            return NONE_TYPE
        if type_class.__module__ == "builtins":
            return TypeExpr.applied(type_class.__qualname__)
        alias = self.aliases.get(type_class)
        if alias is not None:
            # Exported classes collide with their constructor terminal.  As in
            # Main.kt, qualify the nonterminal to disambiguate the plain text.
            return TypeExpr.applied(
                self.profile.render_qualified_name(f"{self.module.__name__}.{alias}")
            )
        module_name = getattr(type_class, "__module__", "")
        qualified_name = getattr(type_class, "__qualname__", getattr(type_class, "__name__", "object"))
        return TypeExpr.applied(
            self.profile.render_qualified_name(f"{module_name}.{qualified_name}".strip("."))
        )

    def is_subtype(self, actual: TypeExpr, bound: TypeExpr) -> bool:
        if bound == OBJECT_TYPE or actual == bound:
            return True
        actual_class = self._class_for_name(actual.name)
        bound_class = self._class_for_name(bound.name)
        if actual_class is None or bound_class is None:
            return False
        try:
            return issubclass(actual_class, bound_class)
        except TypeError:
            return False

    def _class_for_name(self, name: str) -> type | None:
        cached = self._classes_by_rendered_name.get(name)
        if cached is not None:
            return cached
        import_name = self.profile.import_qualified_name(name)
        parts = import_name.split(".")
        for split_at in range(len(parts) - 1, 0, -1):
            try:
                value: object = importlib.import_module(".".join(parts[:split_at]))
                for attribute in parts[split_at:]:
                    value = getattr(value, attribute)
            except (AttributeError, ImportError):
                continue
            if inspect.isclass(value):
                type_class = typing.cast(type, value)
                self._classes_by_rendered_name[name] = type_class
                return type_class
        return None

    def annotation(self, annotation: object) -> TypeExpr:
        if annotation is inspect.Signature.empty or annotation is Any or annotation is typing.Any:
            return OBJECT_TYPE
        if annotation is None or annotation is type(None):
            return NONE_TYPE
        if annotation is Ellipsis:
            return TypeExpr.applied("...")
        if isinstance(annotation, str):
            return self.documented_type(annotation)
        if isinstance(annotation, typing.ForwardRef):
            return self.documented_type(annotation.__forward_arg__)
        if type(annotation).__name__ == "TypeVar":
            return TypeExpr.variable(getattr(annotation, "__name__", "T"))
        if inspect.isclass(annotation):
            return self.render_class(typing.cast(type, annotation))

        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        if origin is typing.Annotated and arguments:
            return self.annotation(arguments[0])
        if origin is typing.Literal:
            literal_types = tuple(dict.fromkeys(self.render_class(type(value)) for value in arguments))
            return literal_types[0] if len(literal_types) == 1 else TypeExpr.applied("Union", *literal_types)
        if origin in {typing.Union, types.UnionType}:
            alternatives = tuple(dict.fromkeys(self.annotation(argument) for argument in arguments))
            return alternatives[0] if len(alternatives) == 1 else TypeExpr.applied("Union", *alternatives)
        if origin is not None:
            origin_type = self.render_class(origin) if inspect.isclass(origin) else TypeExpr.applied(str(origin).replace("typing.", ""))
            return TypeExpr.applied(origin_type.name, *(self.annotation(argument) for argument in arguments))
        return self.documented_type(str(annotation))

    def documented_type(self, documented: str) -> TypeExpr:
        text = documented.replace("`", "").replace("~", "").strip()
        text = re.sub(r",?\s*optional\b", "", text, flags=re.IGNORECASE)
        lowered = text.lower()
        if not text or lowered in {"any", "object", "array_like or scalar", "scalar or array_like"}:
            return OBJECT_TYPE
        if lowered in {"none", "nonetype"}:
            return NONE_TYPE
        if "ndarray" in lowered or "array_like" in lowered or "array-like" in lowered:
            if self.profile.array_type_name is None:
                return TypeExpr.applied("list") if "array" in lowered and "like" in lowered else OBJECT_TYPE
            array_type = self._named_export_type("ndarray") or TypeExpr.applied(
                self.profile.render_qualified_name(self.profile.array_type_name)
            )
            if "bool" in lowered:
                bool_type = self._named_export_type("bool_") or TypeExpr.applied("bool")
                return TypeExpr.applied(array_type.name, bool_type)
            return array_type
        if re.search(r"\b(dtype|data-type)\b", lowered):
            return self._named_export_type("dtype") or OBJECT_TYPE
        if re.search(r"\bscalar\b", lowered):
            return self._named_export_type("generic") or OBJECT_TYPE
        simple_types: tuple[tuple[str, type], ...] = (
            (r"\bboolean\b|\bbool\b", bool),
            (r"\binteger\b|\bint\b", int),
            (r"\bfloating\b|\bfloat\b", float),
            (r"\bcomplex\b", complex),
            (r"\bstring\b|\bstr\b", str),
            (r"\bbytes\b", bytes),
            (r"\bslice\b", slice),
        )
        if "tuple" in lowered:
            return TypeExpr.applied("tuple")
        if "list" in lowered or "sequence" in lowered or "iterable" in lowered:
            return TypeExpr.applied("list")
        if "dict" in lowered or "mapping" in lowered:
            return TypeExpr.applied("dict")
        if "callable" in lowered or "function" in lowered:
            return TypeExpr.applied("Callable")
        for pattern, type_class in simple_types:
            if re.search(pattern, lowered):
                return self.render_class(type_class)

        bare = re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", text)
        if bare:
            short_name = text.rsplit(".", 1)[-1]
            exported = self._named_export_type(short_name)
            if exported is not None:
                return exported
            if short_name[:1].isupper():
                return TypeExpr.applied(text)
        return OBJECT_TYPE

    def _named_export_type(self, name: str) -> TypeExpr | None:
        try:
            value = getattr(self.module, name)
        except BaseException:
            try:
                root = importlib.import_module(self.root_name)
                value = getattr(root, name)
            except BaseException:
                return None
        return self.render_class(value) if inspect.isclass(value) else None


def _documented_parameter_types(documentation: str | None, renderer: TypeRenderer) -> dict[str, TypeExpr]:
    result: dict[str, TypeExpr] = {}
    for names, documented_type in _doc_fields(documentation, "Parameters"):
        for name in names.split(","):
            cleaned = name.strip().lstrip("*")
            if cleaned:
                result[cleaned] = renderer.documented_type(documented_type)
    return result


def _documented_return_type(documentation: str | None, renderer: TypeRenderer) -> TypeExpr:
    fields = _doc_fields(documentation, "Returns") or _doc_fields(documentation, "Yields")
    if not fields:
        return OBJECT_TYPE
    return_types = tuple(renderer.documented_type(documented_type) for _, documented_type in fields)
    if len(return_types) == 1:
        return return_types[0]
    return TypeExpr.applied("tuple", *return_types)


def _parameters_from_doc_signature(documentation: str | None, renderer: TypeRenderer) -> list[inspect.Parameter] | None:
    signature_line = _balanced_doc_signature(documentation)
    if signature_line is None:
        return None
    start = signature_line.find("(")
    end = signature_line.rfind(")")
    if start < 0 or end <= start:
        return None
    content = signature_line[start + 1 : end]

    # NumPy documents optional groups both as defaults (``axis=None``) and as
    # brackets. Remove bracket groups completely because this generator emits
    # the minimal call. Importantly, required text outside a group survives:
    # ``arange([start,] stop[, step,])`` becomes ``arange(stop)``.
    required_content: list[str] = []
    bracket_depth = 0
    for character in content:
        if character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        elif bracket_depth == 0:
            required_content.append(character)
    parts = _split_top_level("".join(required_content))
    parameters: list[inspect.Parameter] = []
    keyword_only = False
    for index, raw_part in enumerate(parts):
        part = raw_part.strip()
        if part in {"/", ""}:
            continue
        if part == "*":
            keyword_only = True
            continue
        prefix = ""
        if part.startswith("**"):
            prefix, part = "**", part[2:]
        elif part.startswith("*"):
            prefix, part = "*", part[1:]
        has_default = "=" in part
        part = part.split("=", 1)[0].strip()
        identifiers = re.findall(r"[A-Za-z_]\w*", part)
        if not identifiers:
            continue
        name = identifiers[-1]
        if name in {"self", "cls"} or name.startswith("..."):
            continue
        kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        if prefix == "**":
            kind = inspect.Parameter.VAR_KEYWORD
        elif prefix == "*":
            kind = inspect.Parameter.VAR_POSITIONAL
            keyword_only = True
        elif keyword_only:
            kind = inspect.Parameter.KEYWORD_ONLY
        default = None if has_default and kind not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        } else inspect.Parameter.empty
        try:
            parameters.append(inspect.Parameter(name or f"arg{index}", kind, default=default))
        except ValueError:
            parameters.append(inspect.Parameter(f"arg{index}", kind, default=default))
    return parameters


def _resolved_type_hints(callable_object: object, renderer: TypeRenderer) -> Mapping[str, object]:
    sources: list[object]
    if inspect.isclass(callable_object):
        # ``inspect.signature(C)`` exposes C.__new__/C.__init__ parameters, not
        # class-variable annotations. Select the source whose unbound signature
        # is exactly the class signature so conflicting __new__/__init__ hints
        # receive the same precedence as inspect.signature itself.
        constructor_sources = [
            getattr(callable_object, "__new__", None),
            getattr(callable_object, "__init__", None),
        ]
        sources = []
        try:
            class_parameters = list(inspect.signature(callable_object).parameters.values())
        except (TypeError, ValueError):
            class_parameters = []
        for source in constructor_sources:
            if source is None:
                continue
            try:
                source_parameters = list(inspect.signature(source).parameters.values())[1:]
            except (TypeError, ValueError):
                continue
            if source_parameters == class_parameters:
                sources = [source]
                break
        if not sources:
            # Later updates take precedence; custom __new__ is what Python's
            # class-signature algorithm prefers when both hooks are usable.
            sources = [source for source in reversed(constructor_sources) if source is not None]
    else:
        sources = [callable_object]

    resolved: dict[str, object] = {}
    for source in sources:
        if source is None:
            continue
        global_namespace: dict[str, object] = {}
        callable_globals = getattr(source, "__globals__", None)
        if isinstance(callable_globals, dict):
            global_namespace.update(callable_globals)
        # This also handles synthetic/reflected modules whose function objects
        # were created elsewhere and then installed into the requested module.
        global_namespace.update(vars(renderer.module))
        try:
            resolved.update(
                typing.get_type_hints(
                    source,
                    globalns=global_namespace,
                    localns=global_namespace,
                    include_extras=True,
                )
            )
            continue
        except Exception:
            pass
        try:
            resolved.update(
                inspect.get_annotations(
                    typing.cast(typing.Callable[..., Any], source),
                    globals=global_namespace,
                    locals=global_namespace,
                    eval_str=True,
                )
            )
        except Exception:
            continue
    return resolved


def _annotation_type_variables(annotation: object) -> dict[str, object]:
    if type(annotation).__name__ == "TypeVar":
        return {getattr(annotation, "__name__", "T"): annotation}
    result: dict[str, object] = {}
    for argument in typing.get_args(annotation):
        result.update(_annotation_type_variables(argument))
    return result


def _signature_parameters(
    callable_object: object,
    renderer: TypeRenderer,
) -> tuple[list[inspect.Parameter], Mapping[str, TypeExpr], Mapping[str, object], bool]:
    documentation = inspect.getdoc(callable_object)
    documented = _documented_parameter_types(documentation, renderer)
    resolved_hints = _resolved_type_hints(callable_object, renderer)
    documented_parameters = _parameters_from_doc_signature(documentation, renderer)
    try:
        signature = inspect.signature(typing.cast(typing.Callable[..., Any], callable_object), follow_wrapped=True)
        parameters = list(signature.parameters.values())
        used_inspect_signature = True
        inspect_required = [
            parameter
            for parameter in parameters
            if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            and parameter.default is inspect.Parameter.empty
        ]
        documented_required = [
            parameter
            for parameter in documented_parameters or ()
            if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            and parameter.default is inspect.Parameter.empty
        ]
        if not inspect_required and documented_required:
            # Decorator-generated ``(*args, **kwargs)`` signatures can be less
            # informative than NumPy's first doc line (notably ``einsum``).
            parameters = typing.cast(list[inspect.Parameter], documented_parameters)
            used_inspect_signature = False
    except (TypeError, ValueError):
        parameters = documented_parameters or [
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ]
        used_inspect_signature = False
    return parameters, documented, resolved_hints, used_inspect_signature


def _reflected_signatures(
    callable_object: object,
    renderer: TypeRenderer,
    max_vararg_arity: int,
    *,
    min_vararg_arity: int = 0,
    strip_receiver: bool = False,
    forced_return_type: TypeExpr | None = None,
    fallback_return_type: TypeExpr | None = None,
) -> list[SignatureSpec]:
    parameters, documented, resolved_hints, used_inspect_signature = _signature_parameters(callable_object, renderer)
    # ``inspect.signature`` exposes ``self`` for Python functions and many
    # descriptors, whereas NumPy's C-level doc signatures are already written
    # as calls on the receiver (for example ``a.sum(axis=...)``).  A successful
    # signature on the unbound class member includes its receiver regardless of
    # the author's parameter spelling; a parsed doc signature does not.
    if strip_receiver and used_inspect_signature and parameters:
        parameters = parameters[1:]
    reflected: list[ReflectedParameter] = []
    vararg_index: int | None = None
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            # A call with no extra keyword is always in the bounded language.
            continue
        if parameter.default is not inspect.Parameter.empty:
            # Emit the minimal valid call. Optional arguments are deliberately
            # omitted instead of filling semantically unrelated value slots.
            continue
        parameter_annotation = resolved_hints.get(parameter.name, parameter.annotation)
        reflected_type = renderer.annotation(parameter_annotation)
        if parameter_annotation is inspect.Signature.empty:
            reflected_type = documented.get(parameter.name, OBJECT_TYPE)
        reflected_parameter = ReflectedParameter(parameter.name, reflected_type, parameter.kind)
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            vararg_index = len(reflected)
        reflected.append(reflected_parameter)

    documentation = inspect.getdoc(callable_object)
    if forced_return_type is not None:
        return_type = forced_return_type
    else:
        try:
            reflected_annotation = inspect.signature(
                typing.cast(typing.Callable[..., Any], callable_object),
                follow_wrapped=True,
            ).return_annotation
        except (TypeError, ValueError):
            reflected_annotation = inspect.Signature.empty
        return_annotation = resolved_hints.get("return", reflected_annotation)
        if return_annotation is not inspect.Signature.empty:
            return_type = renderer.annotation(return_annotation)
        elif _doc_fields(documentation, "Returns") or _doc_fields(documentation, "Yields"):
            return_type = _documented_return_type(documentation, renderer)
        else:
            return_type = fallback_return_type or OBJECT_TYPE

    reflected_type_variables: dict[str, object] = {}
    for annotation in resolved_hints.values():
        reflected_type_variables.update(_annotation_type_variables(annotation))
    domains: dict[str, tuple[TypeExpr, ...]] = {}
    bounds: dict[str, TypeExpr] = {}
    for variable, type_variable in reflected_type_variables.items():
        constraints = getattr(type_variable, "__constraints__", ())
        bound = getattr(type_variable, "__bound__", None)
        if constraints:
            domains[variable] = tuple(dict.fromkeys(renderer.annotation(constraint) for constraint in constraints))
        if bound is not None:
            bounds[variable] = renderer.annotation(bound)

    if vararg_index is None:
        return [SignatureSpec(tuple(reflected), return_type, domains, bounds)]
    vararg = reflected[vararg_index]
    fixed_before = reflected[:vararg_index]
    fixed_after = reflected[vararg_index + 1 :]
    result: list[SignatureSpec] = []
    for arity in range(min_vararg_arity, max(max_vararg_arity, min_vararg_arity) + 1):
        expanded = fixed_before + [
            ReflectedParameter(f"{vararg.name}{index + 1}", vararg.type, inspect.Parameter.POSITIONAL_ONLY)
            for index in range(arity)
        ] + fixed_after
        result.append(SignatureSpec(tuple(expanded), return_type, domains, bounds))
    return result


def _rhs_for_call(prefix: Iterable[Symbol], parameters: Sequence[ReflectedParameter]) -> tuple[Symbol, ...]:
    symbols = list(prefix)
    symbols.append(Token("("))
    for index, parameter in enumerate(parameters):
        if index:
            symbols.append(Token(","))
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            symbols.extend((Token(parameter.name), Token("=")))
        symbols.append(TypeSymbol(parameter.type))
    symbols.append(Token(")"))
    return tuple(symbols)


class CfgGenerator:
    def __init__(self, options: GeneratorOptions):
        if options.max_array_literal_values < 0:
            raise ValueError("max_array_literal_values must be non-negative")
        self.options = options
        self.profile = LibraryProfile.for_module(
            options.module_name,
            alias=options.module_alias,
            ground_type_names=options.ground_type_names,
        )
        if self.profile.extension == "numpy" and not self.profile.ground_type_names:
            raise ValueError("ground_type_names must contain at least one type")
        if len(set(self.profile.ground_type_names)) != len(self.profile.ground_type_names):
            raise ValueError("ground_type_names must not contain duplicate names")
        self.scanner = PythonModuleScanner()
        self.module: types.ModuleType | None = None
        self.renderer: TypeRenderer | None = None
        self.members: list[ExportedMember] = []
        self.parameterized_name_keys: dict[str, str] = {}
        self.type_argument_types: tuple[TypeExpr, ...] = ()
        self.ground_type_entries: tuple[tuple[str, Any, TypeExpr], ...] = ()

    def generate(self) -> GeneratedGrammar:
        module = importlib.import_module(self.options.module_name)
        return self.generate_module(module)

    def generate_module(self, module: types.ModuleType) -> GeneratedGrammar:
        self.module = module
        self.members = self.scanner.exported_members(module)
        self.renderer = TypeRenderer(module, self.members, self.profile)
        renderer = self.renderer
        self.ground_type_entries = self._build_ground_type_entries()
        self.type_argument_types = self._build_type_argument_types()
        productions: set[Production] = set(self._literal_productions())
        productions.update(self._library_literal_productions())

        callable_members = [member for member in self.members if callable(member.value)]
        class_members = [
            member
            for member in callable_members
            if inspect.isclass(member.value)
            and self._class_is_in_profile_scope(typing.cast(type, member.value))
        ]
        for member in class_members:
            productions.update(self._constructor_productions(member.name, typing.cast(type, member.value)))

        unique_classes: dict[type, str] = {}
        for member in class_members:
            type_class = typing.cast(type, member.value)
            canonical = renderer.aliases.get(type_class, member.name)
            unique_classes[type_class] = canonical
        for type_class, owner_token in sorted(unique_classes.items(), key=lambda item: renderer.render_class(item[0]).render()):
            productions.update(self._member_productions(type_class, owner_token))
            productions.update(self._subtype_productions(type_class))

        for member in callable_members:
            if inspect.isclass(member.value):
                continue
            specialized = self._library_callable_productions(member.name, member.value)
            if specialized is None:
                productions.update(self._top_level_callable_productions(member.name, member.value))
            else:
                productions.update(specialized)

        productions.update(self._supporting_type_productions(productions))
        if not self.options.emit_parameterized_signatures:
            productions = self._prune_undefined_nonterminals(productions)
            productions = self._prune_non_generating_productions(productions)
        productions.update(self._start_productions(productions))

        if self.options.normalize_chomsky_normal_form:
            productions = to_chomsky_normal_form(productions, START_TYPE)
        return GeneratedGrammar.from_productions(productions)

    @property
    def _type_renderer(self) -> TypeRenderer:
        assert self.renderer is not None
        return self.renderer

    @property
    def _uses_numpy_extension(self) -> bool:
        return self.profile.extension == "numpy"

    @property
    def _array_type_name(self) -> str | None:
        if self.profile.array_type_name is None:
            return None
        return self.profile.render_qualified_name(self.profile.array_type_name)

    @staticmethod
    def _dotted_prefix(parts: Sequence[str]) -> tuple[Symbol, ...]:
        result: list[Symbol] = []
        for index, part in enumerate(parts):
            if index:
                result.append(Token("."))
            result.append(Token(part))
        return tuple(result)

    def _module_access_prefix(self, name: str) -> tuple[Symbol, ...]:
        assert self.module is not None
        return self._dotted_prefix(self.profile.module_access_parts(self.module.__name__, name))

    def _library_root_access_prefix(self, name: str) -> tuple[Symbol, ...]:
        return self._dotted_prefix(self.profile.root_access_parts(name))

    def _build_ground_type_entries(self) -> tuple[tuple[str, Any, TypeExpr], ...]:
        if not self._uses_numpy_extension:
            return ()
        numpy = importlib.import_module("numpy")
        entries: list[tuple[str, Any, TypeExpr]] = []
        scalar_classes: set[type] = set()
        for name in self.profile.ground_type_names:
            try:
                scalar_class = getattr(numpy, name)
                dtype = numpy.dtype(scalar_class)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"Unknown NumPy ground dtype: {name}") from error
            if not inspect.isclass(scalar_class) or dtype.type is not scalar_class:
                raise ValueError(f"NumPy ground dtype must name a concrete scalar class: {name}")
            if scalar_class in scalar_classes:
                raise ValueError(f"NumPy ground dtype aliases an earlier entry: {name}")
            scalar_classes.add(scalar_class)
            entries.append((name, dtype, self._type_renderer.render_class(scalar_class)))
        return tuple(entries)

    @property
    def _ground_scalar_classes(self) -> frozenset[type]:
        return frozenset(entry[1].type for entry in self.ground_type_entries)

    def _class_is_in_profile_scope(self, type_class: type) -> bool:
        """Keep only configured concrete NumPy scalars and their supertypes."""

        if not self._uses_numpy_extension:
            return True
        numpy = importlib.import_module("numpy")
        try:
            if not issubclass(type_class, numpy.generic):
                return True
        except TypeError:
            return True
        if type_class in self._ground_scalar_classes:
            return True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                is_concrete = numpy.dtype(type_class).type is type_class
            except (TypeError, ValueError):
                is_concrete = False
        if is_concrete:
            return False
        return any(
            issubclass(scalar_class, type_class)
            for scalar_class in self._ground_scalar_classes
        )

    def _type_expr_is_in_scope(self, type_expr: TypeExpr) -> bool:
        if not self._uses_numpy_extension:
            return True
        for nested in type_expr.walk():
            if nested.is_variable:
                continue
            type_class = self._type_renderer._class_for_name(nested.name)
            if type_class is not None and not self._class_is_in_profile_scope(type_class):
                return False
        return True

    def _uses_strict_numpy_ufunc_operators(self, type_class: type) -> bool:
        if not self._uses_numpy_extension:
            return False
        numpy = importlib.import_module("numpy")
        try:
            return issubclass(type_class, (numpy.generic, numpy.ndarray))
        except TypeError:
            return False

    def _build_type_argument_types(self) -> tuple[TypeExpr, ...]:
        preferred: list[TypeExpr] = [
            TypeExpr.applied("int"),
            TypeExpr.applied("float"),
            TypeExpr.applied("bool"),
            TypeExpr.applied("str"),
            TypeExpr.applied("complex"),
            TypeExpr.applied("bytes"),
            TypeExpr.applied("list"),
            TypeExpr.applied("tuple"),
            OBJECT_TYPE,
        ]
        if self._uses_numpy_extension:
            numpy = importlib.import_module("numpy")
            for name in (*self.profile.ground_type_names, "generic", "ndarray", "dtype"):
                value = getattr(numpy, name, None)
                if inspect.isclass(value):
                    candidate = self._type_renderer.render_class(value)
                    if candidate not in preferred:
                        preferred.append(candidate)
        else:
            for member in self.members:
                if inspect.isclass(member.value):
                    candidate = self._type_renderer.render_class(typing.cast(type, member.value))
                    if candidate not in preferred:
                        preferred.append(candidate)
        return tuple(preferred[: self.options.max_type_arguments_per_variable])

    def _literal_productions(self) -> list[Production]:
        return [
            Production(NONE_TYPE, (Token("None"),)),
            Production(TypeExpr.applied("bool"), (Token("False"),)),
            Production(TypeExpr.applied("bool"), (Token("True"),)),
            Production(TypeExpr.applied("int"), (Token("-1"),)),
            Production(TypeExpr.applied("int"), (Token("0"),)),
            Production(TypeExpr.applied("int"), (Token("1"),)),
            Production(TypeExpr.applied("float"), (Token("-1.0"),)),
            Production(TypeExpr.applied("float"), (Token("0.0"),)),
            Production(TypeExpr.applied("float"), (Token("1.0"),)),
            Production(TypeExpr.applied("complex"), (Token("-1j"),)),
            Production(TypeExpr.applied("complex"), (Token("0j"),)),
            Production(TypeExpr.applied("complex"), (Token("1j"),)),
            Production(TypeExpr.applied("str"), (Token('""'),)),
            Production(TypeExpr.applied("str"), (Token('"a"'),)),
            Production(TypeExpr.applied("bytes"), (Token('b""'),)),
            Production(TypeExpr.applied("bytes"), (Token('b"a"'),)),
            Production(TypeExpr.applied("list"), (Token("["), Token("]"))),
            Production(TypeExpr.applied("tuple"), (Token("("), Token(")"))),
            Production(TypeExpr.applied("dict"), (Token("{"), Token("}"))),
            Production(TypeExpr.applied("set"), (Token("set"), Token("("), Token(")"))),
            Production(OBJECT_TYPE, (Token("None"),)),
            Production(
                TypeExpr.applied("Callable"),
                (Token("lambda"), Token("*"), Token("args"), Token(":"), Token("0")),
            ),
        ]

    def _library_literal_productions(self) -> list[Production]:
        """Return literal helpers supplied by the active library extension."""

        if not self._uses_numpy_extension:
            return []
        productions: list[Production] = []
        categories: set[str] = set()
        for _name, dtype, _type_expr in self.ground_type_entries:
            category = _NUMPY_KIND_LITERAL_CATEGORY.get(dtype.kind)
            if category is not None:
                categories.add(category)
        for category in sorted(categories):
            terminals = _NUMPY_LITERAL_TERMINALS[category]
            literal_type = TypeExpr.applied(f"__NP_LITERAL_{category}")
            list_type = TypeExpr.applied(f"__NP_LITERAL_LIST_{category}")
            productions.extend(Production(literal_type, (Token(value),)) for value in terminals)
            for length in range(self.options.max_array_literal_values + 1):
                symbols: list[Symbol] = [Token("[")]
                for index in range(length):
                    if index:
                        symbols.append(Token(","))
                    symbols.append(TypeSymbol(literal_type))
                symbols.append(Token("]"))
                productions.append(Production(list_type, symbols))
        for operand_count in range(1, max(self.options.max_vararg_arity, 1) + 1):
            subscripts = ",".join("i" for _ in range(operand_count))
            productions.append(
                Production(
                    TypeExpr.applied(f"__NP_EINSUM_SUBSCRIPT_{operand_count}"),
                    (Token(f'"{subscripts}"'),),
                )
            )
        return productions

    def _constructor_productions(self, token_name: str, type_class: type) -> list[Production]:
        result_type = self._type_renderer.render_class(type_class)
        if not self._class_is_in_profile_scope(type_class):
            return []
        if inspect.isabstract(type_class):
            return []
        if self._uses_numpy_extension and token_name in {"flatiter", "ufunc"}:
            # These public type objects explicitly reject direct construction.
            return []
        numpy_scalar = self._numpy_scalar_constructor_productions(token_name, type_class, result_type)
        if numpy_scalar is not None:
            return numpy_scalar
        signatures = _reflected_signatures(
            type_class,
            self._type_renderer,
            self.options.max_vararg_arity,
            forced_return_type=result_type,
        )
        return self._emit_signatures(
            signatures,
            scope=f"constructor|{token_name}|{result_type.render()}",
            rhs_builder=lambda parameters: _rhs_for_call(self._module_access_prefix(token_name), parameters),
        )

    def _numpy_scalar_constructor_productions(
        self,
        token_name: str,
        type_class: type,
        result_type: TypeExpr,
    ) -> list[Production] | None:
        """Return conservative constructors for concrete NumPy scalar classes.

        NumPy's scalar classes generally do not expose inspectable signatures;
        the generic fallback consequently used to give every scalar zero to
        three arbitrary arguments.  A concrete dtype plus a small literal
        domain is both more useful and faithful to the runtime API.
        """

        if not self._uses_numpy_extension:
            return None
        numpy = importlib.import_module("numpy")
        try:
            if not issubclass(type_class, numpy.generic):
                return None
        except TypeError:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dtype = numpy.dtype(type_class)
            except (TypeError, ValueError):
                return []
        # ``np.integer``, ``np.floating`` and friends coerce to a default
        # concrete dtype, but the classes themselves cannot be instantiated.
        if dtype.type is not type_class:
            return []
        configured_name = next(
            (
                name
                for name, configured_dtype, _type_expr in self.ground_type_entries
                if configured_dtype.type is type_class
            ),
            None,
        )
        if configured_name is None or token_name != configured_name:
            return []

        category = _NUMPY_KIND_LITERAL_CATEGORY.get(dtype.kind)
        if category is None:
            return []
        prefix = self._module_access_prefix(token_name)
        literal_type = TypeExpr.applied(f"__NP_LITERAL_{category}")
        if dtype.kind == "O":
            # np.object_(x) deliberately unwraps to the ordinary Python value.
            return [
                Production(
                    TypeExpr.applied("int"),
                    _rhs_for_call(prefix, (ReflectedParameter("value", literal_type),)),
                )
            ]
        parameters = [ReflectedParameter("value", literal_type)]
        if dtype.kind in {"M", "m"}:
            parameters.append(ReflectedParameter("unit", TypeExpr.applied("__NP_DAY_UNIT")))
        productions = self._emit_signatures(
            [SignatureSpec(tuple(parameters), result_type)],
            scope=f"numpy-scalar-constructor|{token_name}|{result_type.render()}",
            rhs_builder=lambda reflected: _rhs_for_call(prefix, reflected),
        )
        if dtype.kind in {"M", "m"}:
            productions.append(Production(TypeExpr.applied("__NP_DAY_UNIT"), (Token('"D"'),)))
        return productions

    def _top_level_callable_productions(self, name: str, value: object) -> list[Production]:
        if self._uses_numpy_extension:
            numpy_override = self._numpy_internal_function_productions(name)
            if numpy_override is not None:
                return numpy_override
        signature_value = value
        if self._uses_numpy_extension:
            signature_aliases = {
                "alltrue": "all",
                "cumproduct": "cumprod",
                "product": "prod",
                "sometrue": "any",
            }
            aliased_name = signature_aliases.get(name)
            if aliased_name is not None:
                signature_value = getattr(importlib.import_module("numpy"), aliased_name)
        minimum_varargs = 1 if self._uses_numpy_extension and name in {
            "einsum",
            "einsum_path",
            "result_type",
        } else 0
        signatures = _reflected_signatures(
            signature_value,
            self._type_renderer,
            self.options.max_vararg_arity,
            min_vararg_arity=minimum_varargs,
        )
        if name in {"einsum", "einsum_path"}:
            arity_specific_signatures: list[SignatureSpec] = []
            for signature in signatures:
                operand_count = sum(parameter.name != "subscripts" for parameter in signature.parameters)
                subscript_type = TypeExpr.applied(f"__NP_EINSUM_SUBSCRIPT_{operand_count}")
                arity_specific_signatures.append(
                    SignatureSpec(
                        tuple(
                            ReflectedParameter(parameter.name, subscript_type, parameter.kind)
                            if parameter.name == "subscripts"
                            else parameter
                            for parameter in signature.parameters
                        ),
                        signature.return_type,
                        signature.domains,
                        signature.bounds,
                        signature.variable_labels,
                    )
                )
            signatures = arity_specific_signatures
        return self._emit_signatures(
            signatures,
            scope=f"function|{self.options.module_name}|{name}",
            rhs_builder=lambda parameters: _rhs_for_call(self._module_access_prefix(name), parameters),
        )

    def _numpy_internal_function_productions(self, name: str) -> list[Production] | None:
        overrides: Mapping[str, tuple[TypeExpr, tuple[Symbol, ...]]] = {
            "_get_promotion_state": (
                TypeExpr.applied("str"),
                _rhs_for_call(self._module_access_prefix(name), ()),
            ),
            "_using_numpy2_behavior": (
                TypeExpr.applied("bool"),
                _rhs_for_call(self._module_access_prefix(name), ()),
            ),
            "_set_promotion_state": (
                NONE_TYPE,
                (
                    *self._module_access_prefix(name),
                    Token("("),
                    Token('"weak"'),
                    Token(")"),
                ),
            ),
        }
        override = overrides.get(name)
        if override is None:
            return None
        return [Production(*override)]

    @staticmethod
    def _has_usable_signature(value: object) -> bool:
        try:
            inspect.signature(typing.cast(typing.Callable[..., Any], value), follow_wrapped=True)
            return True
        except (TypeError, ValueError):
            return _balanced_doc_signature(inspect.getdoc(value)) is not None

    def _numpy_member_signature_override(
        self,
        name: str,
        reflected_member: object,
        receiver_type: TypeExpr,
        forced_return: TypeExpr | None,
        *,
        strip_receiver: bool,
    ) -> list[SignatureSpec] | None:
        if not self._uses_numpy_extension or self._has_usable_signature(reflected_member):
            return None
        numpy = importlib.import_module("numpy")
        ndarray_member = getattr(numpy.ndarray, name, None)
        if strip_receiver and callable(ndarray_member) and self._has_usable_signature(ndarray_member):
            return _reflected_signatures(
                ndarray_member,
                self._type_renderer,
                self.options.max_vararg_arity,
                strip_receiver=True,
                fallback_return_type=forced_return,
            )

        parameter_types: Mapping[str, tuple[TypeExpr, ...]] = {
            "__array_function__": (OBJECT_TYPE, TypeExpr.applied("tuple"), TypeExpr.applied("tuple"), TypeExpr.applied("dict")),
            "__array_ufunc__": (OBJECT_TYPE, TypeExpr.applied("str"), TypeExpr.applied("tuple"), TypeExpr.applied("dict")),
            "__complex__": (),
            "__enter__": (),
            "__exit__": (OBJECT_TYPE, OBJECT_TYPE, OBJECT_TYPE),
            "__format__": (TypeExpr.applied("str"),),
            "__getnewargs__": (),
            "__init_subclass__": (),
            "__reduce_ex__": (TypeExpr.applied("int"),),
            "__round__": (),
            "__sizeof__": (),
            "bit_count": (),
            "dot": (receiver_type,),
            "hex": (),
        }
        types_for_parameters = parameter_types.get(name)
        if types_for_parameters is None:
            return None
        known_returns: Mapping[str, TypeExpr] = {
            "__enter__": receiver_type,
            "__format__": TypeExpr.applied("str"),
            "__getnewargs__": TypeExpr.applied("tuple"),
            "__init_subclass__": NONE_TYPE,
            "__reduce_ex__": TypeExpr.applied("tuple"),
            "__sizeof__": TypeExpr.applied("int"),
            "bit_count": TypeExpr.applied("int"),
            "dot": receiver_type,
            "hex": TypeExpr.applied("str"),
        }
        if forced_return is not None:
            return_type = forced_return
        else:
            return_type = known_returns.get(name, OBJECT_TYPE)
        parameters = tuple(
            ReflectedParameter(f"arg{index + 1}", parameter_type, inspect.Parameter.POSITIONAL_ONLY)
            for index, parameter_type in enumerate(types_for_parameters)
        )
        return [SignatureSpec(parameters, return_type)]

    def _member_productions(self, type_class: type, owner_token: str) -> list[Production]:
        receiver_type = self._type_renderer.render_class(type_class)
        result: list[Production] = []
        try:
            names = sorted(set(dir(type_class)))
        except BaseException:
            return result
        for name in names:
            if name in {"__init__", "__new__", "__class__"}:
                continue
            if (
                name in _NUMPY_UFUNC_BACKED_DUNDERS
                and self._uses_strict_numpy_ufunc_operators(type_class)
            ):
                # Direct ufunc schemas emit both call and operator spellings.
                # Reflecting these dunders as ``object``-typed methods would
                # reintroduce mixed operands and NumPy promotion paths.
                continue
            is_dunder = name.startswith("__") and name.endswith("__")
            if name.startswith("_") and not is_dunder:
                continue
            try:
                raw_member = inspect.getattr_static(type_class, name)
                reflected_member = getattr(type_class, name)
            except BaseException:
                continue

            if callable(reflected_member):
                is_static = isinstance(raw_member, staticmethod)
                is_class = isinstance(raw_member, (classmethod, types.ClassMethodDescriptorType))
                strip_receiver = not (is_static or is_class)
                forced_return = self._special_method_return_type(name, receiver_type)
                signatures = self._numpy_member_signature_override(
                    name,
                    reflected_member,
                    receiver_type,
                    forced_return,
                    strip_receiver=strip_receiver,
                ) or _reflected_signatures(
                    reflected_member,
                    self._type_renderer,
                    self.options.max_vararg_arity,
                    strip_receiver=strip_receiver,
                    fallback_return_type=forced_return,
                )
                if is_static or is_class:
                    prefix = (*self._module_access_prefix(owner_token), Token("."), Token(name))
                else:
                    prefix = (TypeSymbol(receiver_type), Token("."), Token(name))
                def member_rhs(
                    parameters: Sequence[ReflectedParameter],
                    call_prefix: tuple[Symbol, ...] = prefix,
                ) -> tuple[Symbol, ...]:
                    return _rhs_for_call(call_prefix, parameters)

                result.extend(
                    self._emit_signatures(
                        signatures,
                        scope=f"member|{receiver_type.render()}|{name}",
                        rhs_builder=member_rhs,
                    )
                )
                if name in OPERATOR_SPECS and not (is_static or is_class):
                    result.extend(self._operator_syntax_productions(name, receiver_type, signatures))
                continue

            if name.startswith("_"):
                continue
            if not self._is_readable_descriptor(raw_member):
                continue
            if isinstance(raw_member, property) and raw_member.fget is not None:
                property_signatures = _reflected_signatures(
                    raw_member.fget,
                    self._type_renderer,
                    self.options.max_vararg_arity,
                    strip_receiver=True,
                )
            else:
                property_signatures = [
                    SignatureSpec((), self._property_type(raw_member, reflected_member))
                ]

            def property_rhs(
                _parameters: Sequence[ReflectedParameter],
                property_name: str = name,
                receiver: TypeExpr = receiver_type,
            ) -> tuple[Symbol, ...]:
                return (TypeSymbol(receiver), Token("."), Token(property_name))

            result.extend(
                self._emit_signatures(
                    property_signatures,
                    scope=f"property|{receiver_type.render()}|{name}",
                    rhs_builder=property_rhs,
                )
            )
        return result

    @staticmethod
    def _is_readable_descriptor(member: object) -> bool:
        return isinstance(member, (property, types.GetSetDescriptorType, types.MemberDescriptorType)) or hasattr(member, "__get__")

    def _property_type(self, raw_member: object, reflected_member: object) -> TypeExpr:
        if isinstance(raw_member, property) and raw_member.fget is not None:
            try:
                annotation = inspect.signature(raw_member.fget).return_annotation
            except (TypeError, ValueError):
                annotation = inspect.Signature.empty
            if annotation is not inspect.Signature.empty:
                return self._type_renderer.annotation(annotation)
        return _documented_return_type(inspect.getdoc(reflected_member), self._type_renderer)

    def _special_method_return_type(self, name: str, receiver_type: TypeExpr) -> TypeExpr | None:
        if name in {"__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__"}:
            if receiver_type.name.endswith(".ndarray"):
                bool_type = self._type_renderer._named_export_type("bool_") or TypeExpr.applied("bool")
                return TypeExpr.applied(receiver_type.name, bool_type)
            if self.profile.import_qualified_name(receiver_type.name).startswith("numpy."):
                return self._type_renderer._named_export_type("bool_") or TypeExpr.applied("bool")
            return TypeExpr.applied("bool")
        if name in {"__contains__", "__bool__"}:
            return TypeExpr.applied("bool")
        if name in {"__len__", "__index__", "__int__", "__floor__", "__ceil__", "__trunc__"}:
            return TypeExpr.applied("int")
        if name == "__hash__":
            return TypeExpr.applied("int")
        if name == "__float__":
            return TypeExpr.applied("float")
        if name == "__complex__":
            return TypeExpr.applied("complex")
        if name in {"__str__", "__repr__", "__format__"}:
            return TypeExpr.applied("str")
        if name == "__bytes__":
            return TypeExpr.applied("bytes")
        if name in {"__setitem__", "__delitem__"}:
            return NONE_TYPE
        if name == "at" and receiver_type.name.endswith(".ufunc"):
            return NONE_TYPE
        if name in _INPLACE_OPERATOR_TOKENS:
            return receiver_type
        return None

    def _operator_syntax_productions(
        self,
        name: str,
        receiver_type: TypeExpr,
        signatures: Sequence[SignatureSpec],
    ) -> list[Production]:
        if name in _INPLACE_OPERATOR_TOKENS or name in {"__setitem__", "__delitem__"}:
            # Assignment/delete forms are statements, not values. Keep them in
            # a root-only nonterminal and first bind the reflected receiver to
            # a legal target; this preserves every operator without allowing
            # ``np.array(...) += x`` to nest inside another expression.
            statement_signatures = [
                SignatureSpec(
                    signature.parameters,
                    STATEMENT_TYPE,
                    signature.domains,
                    signature.bounds,
                    signature.variable_labels,
                )
                for signature in signatures
            ]

            def statement_rhs(
                parameters: Sequence[ReflectedParameter],
                operator_name: str = name,
                receiver: TypeExpr = receiver_type,
            ) -> tuple[Symbol, ...]:
                prefix: list[Symbol] = [
                    Token("_value"),
                    Token("="),
                    TypeSymbol(receiver),
                    Token(";"),
                ]
                if operator_name in _INPLACE_OPERATOR_TOKENS and parameters:
                    return tuple(
                        [
                            *prefix,
                            Token("_value"),
                            Token(_INPLACE_OPERATOR_TOKENS[operator_name]),
                            TypeSymbol(parameters[0].type),
                        ]
                    )
                if operator_name == "__setitem__" and len(parameters) >= 2:
                    return tuple(
                        [
                            *prefix,
                            Token("_value"),
                            Token("["),
                            TypeSymbol(parameters[0].type),
                            Token("]"),
                            Token("="),
                            TypeSymbol(parameters[1].type),
                        ]
                    )
                if operator_name == "__delitem__" and parameters:
                    return tuple(
                        [
                            *prefix,
                            Token("del"),
                            Token("_value"),
                            Token("["),
                            TypeSymbol(parameters[0].type),
                            Token("]"),
                        ]
                    )
                raise AssertionError(f"Unsupported statement operator: {operator_name}")

            return self._emit_signatures(
                statement_signatures,
                scope=f"operator-statement|{receiver_type.render()}|{name}",
                rhs_builder=statement_rhs,
            )
        result: list[Production] = []
        for signature in signatures:
            preview = self._operator_rhs(name, receiver_type, signature.parameters)
            if preview is None:
                continue

            def operator_rhs(
                parameters: Sequence[ReflectedParameter],
                operator_name: str = name,
                receiver: TypeExpr = receiver_type,
            ) -> tuple[Symbol, ...]:
                rendered = self._operator_rhs(operator_name, receiver, parameters)
                assert rendered is not None
                return rendered

            result.extend(
                self._emit_signatures(
                    [signature],
                    scope=f"operator|{receiver_type.render()}|{name}|{' '.join(symbol.render() for symbol in preview)}",
                    rhs_builder=operator_rhs,
                )
            )
        return result

    def _operator_rhs(
        self,
        name: str,
        receiver_type: TypeExpr,
        parameters: Sequence[ReflectedParameter],
    ) -> tuple[Symbol, ...] | None:
        if name in _BINARY_OPERATOR_TOKENS and parameters:
            return (TypeSymbol(receiver_type), Token(_BINARY_OPERATOR_TOKENS[name]), TypeSymbol(parameters[0].type))
        if name in _REFLECTED_OPERATOR_TOKENS and parameters:
            token = _REFLECTED_OPERATOR_TOKENS[name]
            if token == "divmod":
                return _rhs_for_call((Token("divmod"),), (parameters[0], ReflectedParameter("right", receiver_type)))
            return (TypeSymbol(parameters[0].type), Token(token), TypeSymbol(receiver_type))
        if name in {"__neg__", "__pos__", "__invert__"}:
            return (Token(_UNARY_OPERATOR_TOKENS[name]), TypeSymbol(receiver_type))
        if name == "__abs__":
            return _rhs_for_call((Token("abs"),), (ReflectedParameter("value", receiver_type),))
        if name == "__divmod__" and parameters:
            return _rhs_for_call((Token("divmod"),), (ReflectedParameter("left", receiver_type), parameters[0]))
        if name == "__getitem__" and parameters:
            return (TypeSymbol(receiver_type), Token("["), TypeSymbol(parameters[0].type), Token("]"))
        if name == "__contains__" and parameters:
            return (TypeSymbol(parameters[0].type), Token("in"), TypeSymbol(receiver_type))
        if name == "__call__":
            return _rhs_for_call((TypeSymbol(receiver_type),), parameters)
        if name in {"__round__", "__format__"}:
            builtin = name[2:-2]
            return _rhs_for_call(
                (Token(builtin),),
                (ReflectedParameter("value", receiver_type), *parameters[:1]),
            )
        if name in {
            "__len__",
            "__bool__",
            "__int__",
            "__float__",
            "__complex__",
            "__iter__",
            "__next__",
            "__reversed__",
            "__hash__",
            "__str__",
            "__repr__",
            "__bytes__",
        }:
            builtin = name[2:-2]
            return _rhs_for_call((Token(builtin),), (ReflectedParameter("value", receiver_type),))
        if name in {"__floor__", "__ceil__", "__trunc__"}:
            if self._uses_numpy_extension:
                # NumPy already exposes these names as ufuncs with dtype-aware
                # return types.  Re-spelling a scalar protocol method as that
                # top-level ufunc creates a conflicting ``int`` production.
                return None
            builtin = name[2:-2]
            prefix = (Token("math"), Token("."), Token(builtin))
            return _rhs_for_call(
                prefix,
                (ReflectedParameter("value", receiver_type),),
            )
        return None

    def _subtype_productions(self, type_class: type) -> list[Production]:
        subtype = self._type_renderer.render_class(type_class)
        result: list[Production] = []
        try:
            bases = type_class.__bases__
        except BaseException:
            return result
        for base in bases:
            if inspect.isclass(base):
                supertype = self._type_renderer.render_class(base)
                if supertype != subtype:
                    result.append(Production(supertype, (TypeSymbol(subtype),)))
        return result

    def _is_numpy_ufunc(self, value: object) -> bool:
        if self.module is None or not self._uses_numpy_extension:
            return False
        try:
            numpy = importlib.import_module("numpy")
            return isinstance(value, numpy.ufunc)
        except (ImportError, AttributeError):
            return False

    def _library_callable_productions(self, name: str, value: object) -> list[Production] | None:
        """Dispatch a callable to an optional library-specific typing oracle."""

        if not self._is_numpy_ufunc(value):
            return None
        result = self._ufunc_productions(name, value)
        if result:
            result.extend(self._ufunc_member_productions(name, value))
        return result

    def _ufunc_productions(self, name: str, ufunc: object) -> list[Production]:
        numpy = importlib.import_module("numpy")
        include_scalar_signatures = getattr(ufunc, "signature", None) is None
        if callable(getattr(ufunc, "resolve_dtypes", None)):
            loops = self._resolved_ufunc_loops(numpy, ufunc)
        else:
            loops = self._raw_ufunc_loops(numpy, ufunc)
        specs = self._ufunc_schema_specs(
            loops,
            include_scalar=include_scalar_signatures,
        )
        if not specs:
            # A ufunc with no exact loop in the configured ground universe is
            # outside this grammar.  An ``object`` fallback would silently
            # reintroduce precisely the implicit conversions excluded here.
            return []

        result = self._emit_signatures(
            specs,
            scope=f"ufunc|{self.options.module_name}|{name}",
            rhs_builder=lambda parameters: _rhs_for_call(self._module_access_prefix(name), parameters),
        )
        operator_token = _UFUNC_OPERATOR_TOKENS.get(name)
        if operator_token is not None:
            def operator_rhs(parameters: Sequence[ReflectedParameter]) -> tuple[Symbol, ...]:
                if name in _UNARY_UFUNC_OPERATORS and parameters:
                    if operator_token == "abs":
                        return _rhs_for_call((Token("abs"),), (parameters[0],))
                    return (Token(operator_token), TypeSymbol(parameters[0].type))
                if operator_token == "divmod" and len(parameters) >= 2:
                    return _rhs_for_call((Token("divmod"),), parameters[:2])
                if len(parameters) >= 2:
                    return (
                        TypeSymbol(parameters[0].type),
                        Token(operator_token),
                        TypeSymbol(parameters[1].type),
                    )
                return _rhs_for_call(self._module_access_prefix(name), parameters)

            result.extend(
                self._emit_signatures(
                    specs,
                    scope=f"ufunc-operator|{self.options.module_name}|{name}",
                    rhs_builder=operator_rhs,
                )
            )
        inplace_token = _UFUNC_INPLACE_TOKENS.get(name)
        if inplace_token is not None:
            inplace_specs = [
                SignatureSpec(
                    signature.parameters,
                    STATEMENT_TYPE,
                    signature.domains,
                    signature.bounds,
                    signature.variable_labels,
                )
                for signature in specs
                if len(signature.parameters) == 2
                and signature.return_type == signature.parameters[0].type
            ]

            def inplace_rhs(parameters: Sequence[ReflectedParameter]) -> tuple[Symbol, ...]:
                return (
                    Token("_value"),
                    Token("="),
                    TypeSymbol(parameters[0].type),
                    Token(";"),
                    Token("_value"),
                    Token(inplace_token),
                    TypeSymbol(parameters[1].type),
                )

            result.extend(
                self._emit_signatures(
                    inplace_specs,
                    scope=f"ufunc-inplace|{self.options.module_name}|{name}",
                    rhs_builder=inplace_rhs,
                )
            )
        return result

    def _ufunc_member_productions(self, name: str, ufunc: object) -> list[Production]:
        """Emit only bound operations derivable from the strict dtype rows.

        ``at`` and the reduction methods have additional accumulator/casting
        rules, so reflecting their untyped compiled signatures would bypass
        the no-promotion policy.  Binary non-gufunc ``outer`` has the same
        elementwise dtype relation as its parent ufunc and is safe to retain.
        """

        if int(getattr(ufunc, "nin", 0)) != 2 or getattr(ufunc, "signature", None) is not None:
            return []
        numpy = importlib.import_module("numpy")
        if callable(getattr(ufunc, "resolve_dtypes", None)):
            loops = self._resolved_ufunc_loops(numpy, ufunc)
        else:
            loops = self._raw_ufunc_loops(numpy, ufunc)
        signatures = self._ufunc_schema_specs(loops, include_scalar=False)
        prefix = (*self._module_access_prefix(name), Token("."), Token("outer"))
        return self._emit_signatures(
            signatures,
            scope=f"ufunc-member|{self.options.module_name}|{name}|outer",
            rhs_builder=lambda parameters: _rhs_for_call(prefix, parameters),
        )

    def _resolved_ufunc_loops(
        self,
        numpy: types.ModuleType,
        ufunc: object,
    ) -> list[tuple[tuple[TypeExpr, ...], tuple[TypeExpr, ...]]]:
        resolve_dtypes = getattr(ufunc, "resolve_dtypes", None)
        input_count = int(getattr(ufunc, "nin", 0))
        output_count = int(getattr(ufunc, "nout", 0))
        if not callable(resolve_dtypes) or input_count <= 0:
            return []
        candidate_dtypes = tuple(entry[1] for entry in self.ground_type_entries)
        allowed_scalar_classes = self._ground_scalar_classes

        result: list[tuple[tuple[TypeExpr, ...], tuple[TypeExpr, ...]]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for requested_inputs in itertools.product(candidate_dtypes, repeat=input_count):
                requested = (*requested_inputs, *(None for _ in range(output_count)))
                try:
                    resolved = resolve_dtypes(requested, casting="no")
                except TypeError as error:
                    # Older NumPy releases lack the keyword.  Equality of the
                    # resolved input slots below still rejects every cast.
                    message = str(error).lower()
                    if "casting" not in message or "keyword" not in message:
                        continue
                    try:
                        resolved = resolve_dtypes(requested)
                    except (TypeError, ValueError):
                        continue
                except ValueError:
                    continue
                if len(resolved) != input_count + output_count:
                    continue
                resolved_inputs = tuple(resolved[:input_count])
                output_dtypes = tuple(resolved[input_count:])
                if any(actual != requested for actual, requested in zip(resolved_inputs, requested_inputs)):
                    continue
                if any(dtype.type not in allowed_scalar_classes for dtype in (*resolved_inputs, *output_dtypes)):
                    continue
                inputs = tuple(self._type_renderer.render_class(dtype.type) for dtype in resolved_inputs)
                outputs = tuple(self._type_renderer.render_class(dtype.type) for dtype in output_dtypes)
                result.append((inputs, outputs))
        return sorted(
            set(result),
            key=lambda loop: (
                tuple(type_expr.render() for type_expr in loop[0]),
                tuple(type_expr.render() for type_expr in loop[1]),
            ),
        )

    def _raw_ufunc_loops(
        self,
        numpy: types.ModuleType,
        ufunc: object,
    ) -> list[tuple[tuple[TypeExpr, ...], tuple[TypeExpr, ...]]]:
        """Compatibility fallback for ufuncs without ``resolve_dtypes``."""

        allowed_scalar_classes = self._ground_scalar_classes
        result: set[tuple[tuple[TypeExpr, ...], tuple[TypeExpr, ...]]] = set()
        for loop in getattr(ufunc, "types", ()):
            try:
                inputs_text, outputs_text = loop.split("->", 1)
                input_dtypes = tuple(numpy.dtype(code) for code in inputs_text)
                output_dtypes = tuple(numpy.dtype(code) for code in outputs_text)
            except (AttributeError, TypeError, ValueError, KeyError):
                continue
            if any(dtype.type not in allowed_scalar_classes for dtype in (*input_dtypes, *output_dtypes)):
                continue
            result.add(
                (
                    tuple(self._type_renderer.render_class(dtype.type) for dtype in input_dtypes),
                    tuple(self._type_renderer.render_class(dtype.type) for dtype in output_dtypes),
                )
            )
        return sorted(
            result,
            key=lambda row: (
                tuple(type_expr.render() for type_expr in row[0]),
                tuple(type_expr.render() for type_expr in row[1]),
            ),
        )

    def _ufunc_schema_specs(
        self,
        loops: Sequence[tuple[tuple[TypeExpr, ...], tuple[TypeExpr, ...]]],
        *,
        include_scalar: bool,
    ) -> list[SignatureSpec]:
        """Factor exact concrete ufunc rows into safe equality schemas.

        A variable is shared only when the corresponding dtype positions are
        equal in every represented row.  A multi-variable schema is emitted
        only if its observed assignments form a complete Cartesian product;
        otherwise it is split into exact singleton schemas.  Consequently,
        expanding the returned specs cannot invent a dtype combination.
        """

        if not loops:
            return []
        input_count = len(loops[0][0])
        output_count = len(loops[0][1])
        loops = tuple(
            row
            for row in loops
            if len(row[0]) == input_count and len(row[1]) == output_count
        )
        if not loops:
            return []

        def constant_at(values: Sequence[TypeExpr]) -> TypeExpr | None:
            first = values[0]
            return first if all(value == first for value in values[1:]) else None

        input_constants = tuple(
            constant_at(tuple(inputs[index] for inputs, _outputs in loops))
            for index in range(input_count)
        )
        output_constants = tuple(
            constant_at(tuple(outputs[index] for _inputs, outputs in loops))
            for index in range(output_count)
        )

        Marker = tuple[str, TypeExpr | int]
        grouped: dict[tuple[Marker, ...], set[tuple[TypeExpr, ...]]] = {}
        for inputs, outputs in loops:
            value_to_variable: dict[TypeExpr, int] = {}
            assignments: list[TypeExpr] = []
            markers: list[Marker] = []
            for index, value in enumerate(inputs):
                constant = input_constants[index]
                if constant is not None:
                    markers.append(("constant", constant))
                    continue
                variable_index = value_to_variable.get(value)
                if variable_index is None:
                    variable_index = len(assignments)
                    value_to_variable[value] = variable_index
                    assignments.append(value)
                markers.append(("variable", variable_index))
            for index, value in enumerate(outputs):
                constant = output_constants[index]
                if constant is not None:
                    markers.append(("constant", constant))
                    continue
                variable_index = value_to_variable.get(value)
                if variable_index is None:
                    markers.append(("constant", value))
                else:
                    markers.append(("variable", variable_index))
            grouped.setdefault(tuple(markers), set()).add(tuple(assignments))

        ground_order = {
            type_expr: index
            for index, (_name, _dtype, type_expr) in enumerate(self.ground_type_entries)
        }

        def ordered(values: Iterable[TypeExpr]) -> tuple[TypeExpr, ...]:
            return tuple(
                sorted(
                    set(values),
                    key=lambda value: (ground_order.get(value, len(ground_order)), value.render()),
                )
            )

        variable_names = ("T", "U", "V", "W", "X", "Y", "Z")
        result: list[SignatureSpec] = []
        for schema_markers, observed_assignments in grouped.items():
            variable_count = max(
                (
                    typing.cast(int, marker_value) + 1
                    for marker_kind, marker_value in schema_markers
                    if marker_kind == "variable"
                ),
                default=0,
            )
            readable_labels: list[str] = []
            for variable_index in range(variable_count):
                input_positions = [
                    index
                    for index, marker in enumerate(schema_markers[:input_count])
                    if marker == ("variable", variable_index)
                ]
                output_positions = [
                    index
                    for index, marker in enumerate(schema_markers[input_count:])
                    if marker == ("variable", variable_index)
                ]
                if len(input_positions) == input_count and output_positions:
                    label = "DType"
                elif len(input_positions) > 1 and not output_positions:
                    label = "OperandDType"
                elif input_positions and output_positions:
                    label = "ResultDType"
                elif input_positions:
                    label = f"Input{input_positions[0] + 1}DType"
                else:
                    label = f"Output{output_positions[0] + 1}DType"
                if label in readable_labels:
                    label = f"{label}{variable_index + 1}"
                readable_labels.append(label)
            domains = tuple(
                ordered(assignment[index] for assignment in observed_assignments)
                for index in range(variable_count)
            )
            rectangular = set(itertools.product(*domains)) == observed_assignments
            assignment_groups: tuple[set[tuple[TypeExpr, ...]], ...]
            if rectangular:
                assignment_groups = (observed_assignments,)
            else:
                assignment_groups = tuple({assignment} for assignment in sorted(
                    observed_assignments,
                    key=lambda assignment: tuple(value.render() for value in assignment),
                ))

            for assignment_group in assignment_groups:
                group_domains = tuple(
                    ordered(assignment[index] for assignment in assignment_group)
                    for index in range(variable_count)
                )
                variables = tuple(
                    TypeExpr.variable(
                        variable_names[index] if index < len(variable_names) else f"T{index + 1}"
                    )
                    for index in range(variable_count)
                )

                def marker_type(marker: Marker) -> TypeExpr:
                    marker_kind, marker_value = marker
                    if marker_kind == "constant":
                        return typing.cast(TypeExpr, marker_value)
                    return variables[typing.cast(int, marker_value)]

                input_types = tuple(marker_type(marker) for marker in schema_markers[:input_count])
                output_types = tuple(marker_type(marker) for marker in schema_markers[input_count:])
                schema_domains = {
                    variable.name: group_domains[index]
                    for index, variable in enumerate(variables)
                }
                variable_labels = {
                    variable.name: readable_labels[index]
                    for index, variable in enumerate(variables)
                }
                result.extend(
                    self._ufunc_signature_specs(
                        input_types,
                        output_types,
                        schema_domains,
                        variable_labels,
                        include_scalar=include_scalar,
                    )
                )
        return result

    def _ufunc_signature_specs(
        self,
        input_dtypes: tuple[TypeExpr, ...],
        output_dtypes: tuple[TypeExpr, ...],
        domains: Mapping[str, tuple[TypeExpr, ...]],
        variable_labels: Mapping[str, str],
        *,
        include_scalar: bool,
    ) -> list[SignatureSpec]:
        def returned(outputs: tuple[TypeExpr, ...]) -> TypeExpr:
            return outputs[0] if len(outputs) == 1 else TypeExpr.applied("tuple", *outputs)

        scalar_parameters = tuple(
            ReflectedParameter(f"x{index + 1}", input_type, inspect.Parameter.POSITIONAL_ONLY)
            for index, input_type in enumerate(input_dtypes)
        )
        assert self._array_type_name is not None
        array_inputs = tuple(TypeExpr.applied(self._array_type_name, dtype) for dtype in input_dtypes)
        array_outputs = tuple(TypeExpr.applied(self._array_type_name, dtype) for dtype in output_dtypes)
        array_parameters = tuple(
            ReflectedParameter(f"x{index + 1}", input_type, inspect.Parameter.POSITIONAL_ONLY)
            for index, input_type in enumerate(array_inputs)
        )
        result = [
            SignatureSpec(
                array_parameters,
                returned(array_outputs),
                domains,
                variable_labels=variable_labels,
            )
        ]
        if include_scalar:
            result.insert(
                0,
                SignatureSpec(
                    scalar_parameters,
                    returned(output_dtypes),
                    domains,
                    variable_labels=variable_labels,
                ),
            )
        return result

    def _emit_signatures(
        self,
        signatures: Iterable[SignatureSpec],
        *,
        scope: str,
        rhs_builder: typing.Callable[[Sequence[ReflectedParameter]], tuple[Symbol, ...]],
    ) -> list[Production]:
        result: list[Production] = []
        for signature_index, signature in enumerate(signatures):
            signature_scope = f"{scope}|overload={signature_index}|{self._signature_key(signature)}"
            signature_types = [signature.return_type, *(parameter.type for parameter in signature.parameters)]
            if any(not self._type_expr_is_in_scope(type_expr) for type_expr in signature_types):
                continue
            if any(not self._type_expr_is_in_scope(bound) for bound in signature.bounds.values()):
                continue
            variables = sorted(
                set(signature.return_type.variables()).union(
                    *(parameter.type.variables() for parameter in signature.parameters)
                )
            )
            if len(variables) > self.options.max_type_variables_per_callable:
                continue
            if not variables:
                if any(type_expr.depth() > self.options.monomorphization_depth for type_expr in signature_types):
                    continue
                result.append(Production(signature.return_type, rhs_builder(signature.parameters)))
                continue
            max_depths = self._max_candidate_depths(signature_types)
            domains = {
                variable: tuple(
                    dict.fromkeys(
                        candidate
                        for candidate in signature.domains.get(variable, self.type_argument_types)
                        if candidate.depth() <= max_depths.get(variable, self.options.monomorphization_depth)
                        and self._type_expr_is_in_scope(candidate)
                        and (
                            variable not in signature.bounds
                            or self._type_renderer.is_subtype(candidate, signature.bounds[variable])
                        )
                    )
                )[: self.options.max_type_arguments_per_variable]
                for variable in variables
            }
            if any(not domain for domain in domains.values()):
                continue
            if self.options.emit_parameterized_signatures:
                replacements: dict[str, TypeExpr] = {}
                domain_productions: list[Production] = []
                for variable in variables:
                    bound = signature.bounds.get(variable)
                    variable_depth = max_depths.get(variable, self.options.monomorphization_depth)
                    readable_variable = signature.variable_labels.get(variable, variable)
                    if bound is not None and bound != OBJECT_TYPE:
                        readable_variable += f"_bound_{_sanitized_part(bound.render())}"
                    variable_name = self._parameterized_symbol_name(
                        "__TP",
                        readable_variable,
                        f"type-parameter|{signature_scope}|{variable}",
                    )
                    domain_key = "|".join(candidate.render() for candidate in domains[variable])
                    domain_name = self._parameterized_symbol_name(
                        "__TP_DOMAIN",
                        self._domain_readable_name(domains[variable], variable_depth),
                        f"type-parameter-domain|depth={variable_depth}|{domain_key}",
                    )
                    parameterized_variable = TypeExpr.variable(variable_name)
                    domain_type = TypeExpr.applied(domain_name)
                    replacements[variable] = parameterized_variable
                    domain_productions.append(Production(parameterized_variable, (TypeSymbol(domain_type),)))
                    domain_productions.extend(
                        Production(domain_type, (TypeSymbol(candidate),)) for candidate in domains[variable]
                    )
                renamed_return = signature.return_type.rename_variables(replacements)
                renamed_parameters = tuple(
                    ReflectedParameter(parameter.name, parameter.type.rename_variables(replacements), parameter.kind)
                    for parameter in signature.parameters
                )
                result.append(Production(renamed_return, rhs_builder(renamed_parameters)))
                result.extend(domain_productions)
            else:
                for values in itertools.product(*(domains[variable] for variable in variables)):
                    substitution = dict(zip(variables, values))
                    return_type = signature.return_type.substitute(substitution)
                    if return_type is None or return_type.depth() > self.options.monomorphization_depth:
                        continue
                    parameters: list[ReflectedParameter] = []
                    for parameter in signature.parameters:
                        parameter_type = parameter.type.substitute(substitution)
                        if parameter_type is None or parameter_type.depth() > self.options.monomorphization_depth:
                            break
                        parameters.append(ReflectedParameter(parameter.name, parameter_type, parameter.kind))
                    else:
                        result.append(Production(return_type, rhs_builder(parameters)))
        return result

    @staticmethod
    def _signature_key(signature: SignatureSpec) -> str:
        parameters = ",".join(f"{parameter.kind.name}:{parameter.type.render()}" for parameter in signature.parameters)
        domains = ";".join(
            f"{variable}:{','.join(candidate.render() for candidate in domain)}"
            for variable, domain in sorted(signature.domains.items())
        )
        bounds = ";".join(
            f"{variable}:{bound.render()}"
            for variable, bound in sorted(signature.bounds.items())
        )
        return f"{signature.return_type.render()}|{parameters}|domains={domains}|bounds={bounds}"

    def _max_candidate_depths(self, types_to_visit: Sequence[TypeExpr]) -> dict[str, int]:
        limits: dict[str, int] = {}

        def visit(type_expr: TypeExpr, nesting_depth: int) -> None:
            if type_expr.is_variable:
                allowed = self.options.monomorphization_depth - nesting_depth
                limits[type_expr.name] = min(limits.get(type_expr.name, allowed), allowed)
                return
            for argument in type_expr.arguments:
                visit(argument, nesting_depth + 1)

        for type_expr in types_to_visit:
            visit(type_expr, 0)
        return limits

    @staticmethod
    def _domain_readable_name(domain: Sequence[TypeExpr], depth: int) -> str:
        def short_name(type_expr: TypeExpr) -> str:
            rendered = type_expr.render()
            if not type_expr.arguments:
                rendered = rendered.rsplit(".", 1)[-1]
            return _sanitized_part(rendered)

        choices = [short_name(candidate) for candidate in domain]
        if len(choices) <= 4:
            summary = "_".join(choices)
        else:
            summary = f"{len(choices)}_types_{choices[0]}_to_{choices[-1]}"
        return f"depth_{depth}_{summary}"

    def _parameterized_symbol_name(self, prefix: str, readable: str, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        encoded = base64.b32encode(digest).decode("ascii").rstrip("=")
        readable_part = _sanitized_part(readable)[:96]
        hash_length = PARAMETERIZED_NAME_HASH_CHARS
        while hash_length <= len(encoded):
            candidate = f"{prefix}_{readable_part}_{encoded[:hash_length]}"
            existing = self.parameterized_name_keys.setdefault(candidate, key)
            if existing == key:
                return candidate
            hash_length += 1
        raise RuntimeError(f"Unable to assign collision-free parameterized symbol for {key}")

    def _numpy_dtype_literal_details(self, dtype_type: TypeExpr) -> tuple[str, tuple[Symbol, ...]] | None:
        if not self._uses_numpy_extension or dtype_type.is_variable:
            return None
        type_class = self._type_renderer._class_for_name(dtype_type.name)
        if type_class is None:
            return None
        numpy = importlib.import_module("numpy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dtype = numpy.dtype(type_class)
            except (TypeError, ValueError):
                return None
        if dtype.type is not type_class or type_class not in self._ground_scalar_classes:
            return None
        category = _NUMPY_KIND_LITERAL_CATEGORY.get(dtype.kind)
        if category is None:
            return None
        dtype_name = next(
            name
            for name, configured_dtype, _type_expr in self.ground_type_entries
            if configured_dtype.type is type_class
        )
        dtype_symbols: tuple[Symbol, ...] = self._library_root_access_prefix(dtype_name)
        return category, dtype_symbols

    def _supporting_type_productions(self, productions: Iterable[Production]) -> set[Production]:
        result: set[Production] = set()
        all_types = {type_expr for production in productions for type_expr in production.types()}
        for type_expr in list(all_types):
            for nested in type_expr.walk():
                if nested.is_variable:
                    continue
                if nested.name == "Union" and nested.arguments:
                    result.update(Production(nested, (TypeSymbol(alternative),)) for alternative in nested.arguments)
                if nested.arguments:
                    result.add(Production(TypeExpr.applied(nested.name), (TypeSymbol(nested),)))
                if nested.name == "list" and nested.arguments:
                    result.add(Production(nested, (Token("["), Token("]"))))
                elif nested.name == "set" and nested.arguments:
                    result.add(Production(nested, (Token("set"), Token("("), Token(")"))))
                elif nested.name == "dict" and nested.arguments:
                    result.add(Production(nested, (Token("{"), Token("}"))))
                elif nested.name == "tuple" and nested.arguments:
                    if any(argument.name == "..." for argument in nested.arguments):
                        result.add(Production(nested, (Token("("), Token(")"))))
                    else:
                        tuple_symbols: list[Symbol] = [Token("(")]
                        for index, argument in enumerate(nested.arguments):
                            if index:
                                tuple_symbols.append(Token(","))
                            tuple_symbols.append(TypeSymbol(argument))
                        if len(nested.arguments) == 1:
                            tuple_symbols.append(Token(","))
                        tuple_symbols.append(Token(")"))
                        result.add(Production(nested, tuple_symbols))
                elif nested.name == self._array_type_name and not nested.arguments:
                    preferred_names = ("float64", "float32", "int64", "int32", "complex128", "complex64", "bool_")
                    seed_entry = next(
                        (
                            entry
                            for preferred_name in preferred_names
                            for entry in self.ground_type_entries
                            if entry[0] == preferred_name
                        ),
                        self.ground_type_entries[0],
                    )
                    seed_name, seed_dtype, _seed_type = seed_entry
                    seed_category = _NUMPY_KIND_LITERAL_CATEGORY.get(seed_dtype.kind)
                    if seed_category is None:
                        continue
                    result.add(
                        Production(
                            nested,
                            (
                                *self._library_root_access_prefix("array"),
                                Token("("),
                                TypeSymbol(TypeExpr.applied(f"__NP_LITERAL_LIST_{seed_category}")),
                                Token(","),
                                Token("dtype"),
                                Token("="),
                                *self._library_root_access_prefix(seed_name),
                                Token(")"),
                            ),
                        )
                    )
                elif nested.name == self._array_type_name and len(nested.arguments) == 1:
                    literal_details = self._numpy_dtype_literal_details(nested.arguments[0])
                    if literal_details is None:
                        continue
                    category, dtype_symbols = literal_details
                    list_type = TypeExpr.applied(f"__NP_LITERAL_LIST_{category}")
                    array_rhs: list[Symbol] = [
                        *self._library_root_access_prefix("array"),
                        Token("("),
                        TypeSymbol(list_type),
                        Token(","),
                        Token("dtype"),
                        Token("="),
                        *dtype_symbols,
                        Token(")"),
                    ]
                    result.add(
                        Production(
                            nested,
                            array_rhs,
                        )
                    )
        return result

    @staticmethod
    def _prune_undefined_nonterminals(productions: Iterable[Production]) -> set[Production]:
        remaining = set(productions)
        while True:
            defined = {production.lhs for production in remaining}
            pruned = {
                production
                for production in remaining
                if all(not isinstance(symbol, TypeSymbol) or symbol.type in defined for symbol in production.rhs)
            }
            if len(pruned) == len(remaining):
                return pruned
            remaining = pruned

    @staticmethod
    def _prune_non_generating_productions(productions: Iterable[Production]) -> set[Production]:
        remaining = set(productions)
        generating: set[TypeExpr] = set()
        changed = True
        while changed:
            changed = False
            for production in remaining:
                if production.lhs in generating:
                    continue
                if all(
                    isinstance(symbol, Token) or symbol.type in generating
                    for symbol in production.rhs
                ):
                    generating.add(production.lhs)
                    changed = True
        return {
            production
            for production in remaining
            if production.lhs in generating
            and all(isinstance(symbol, Token) or symbol.type in generating for symbol in production.rhs)
        }

    @staticmethod
    def _start_productions(productions: Iterable[Production]) -> set[Production]:
        return {
            Production(START_TYPE, (TypeSymbol(lhs),))
            for lhs in {production.lhs for production in productions}
            if lhs != START_TYPE
        }


class _ChomskyNormalFormConverter:
    def __init__(self, start: TypeExpr):
        self.start = start
        self.terminal_nonterminals: dict[str, TypeExpr] = {}
        self.suffix_nonterminals: dict[tuple[TypeSymbol, ...], TypeExpr] = {}

    def convert(self, productions: Iterable[Production]) -> set[Production]:
        normalized: set[Production] = set()
        ordered = sorted(set(productions), key=lambda production: production.render())
        for production in ordered:
            if production.rhs:
                normalized.update(self._normalize_shape(production))
        for terminal, nonterminal in self.terminal_nonterminals.items():
            normalized.add(Production(nonterminal, (Token(terminal),)))
        without_units = self._eliminate_units(normalized)
        return self._prune_useful(without_units)

    def _normalize_shape(self, production: Production) -> set[Production]:
        rhs: tuple[Symbol, ...]
        if len(production.rhs) == 1:
            rhs = production.rhs
        else:
            rhs = tuple(
                symbol if isinstance(symbol, TypeSymbol) else TypeSymbol(self._terminal_nonterminal(symbol.value))
                for symbol in production.rhs
            )
        if len(rhs) <= 2:
            return {Production(production.lhs, rhs)}
        typed = typing.cast(tuple[TypeSymbol, ...], rhs)
        result: set[Production] = set()
        current_lhs = production.lhs
        for index in range(len(typed) - 2):
            suffix = typed[index + 1 :]
            suffix_type = self._suffix_nonterminal(suffix)
            result.add(Production(current_lhs, (typed[index], TypeSymbol(suffix_type))))
            current_lhs = suffix_type
        result.add(Production(current_lhs, typed[-2:]))
        return result

    def _terminal_nonterminal(self, terminal: str) -> TypeExpr:
        if terminal not in self.terminal_nonterminals:
            name = _sanitized_part(terminal)[:32] or "TOKEN"
            self.terminal_nonterminals[terminal] = TypeExpr.applied(
                f"__CNF_T_{name}_{len(self.terminal_nonterminals) + 1:04d}"
            )
        return self.terminal_nonterminals[terminal]

    def _suffix_nonterminal(self, suffix: tuple[TypeSymbol, ...]) -> TypeExpr:
        if suffix not in self.suffix_nonterminals:
            self.suffix_nonterminals[suffix] = TypeExpr.applied(
                f"__CNF_N_{len(self.suffix_nonterminals) + 1:06d}"
            )
        return self.suffix_nonterminals[suffix]

    @staticmethod
    def _eliminate_units(productions: set[Production]) -> set[Production]:
        unit_targets: dict[TypeExpr, set[TypeExpr]] = {}
        non_units: dict[TypeExpr, set[Production]] = {}
        nonterminals: set[TypeExpr] = set()
        for production in productions:
            nonterminals.update(production.types())
            if len(production.rhs) == 1 and isinstance(production.rhs[0], TypeSymbol):
                unit_targets.setdefault(production.lhs, set()).add(production.rhs[0].type)
            else:
                non_units.setdefault(production.lhs, set()).add(production)
        result: set[Production] = set()
        for source in nonterminals:
            closure = {source}
            queue = deque((source,))
            while queue:
                current = queue.popleft()
                for target in unit_targets.get(current, ()):
                    if target not in closure:
                        closure.add(target)
                        queue.append(target)
            for target in closure:
                result.update(Production(source, production.rhs) for production in non_units.get(target, ()))
        return result

    def _prune_useful(self, productions: set[Production]) -> set[Production]:
        generating: set[TypeExpr] = set()
        changed = True
        while changed:
            changed = False
            for production in productions:
                if production.lhs not in generating and all(
                    not isinstance(symbol, TypeSymbol) or symbol.type in generating for symbol in production.rhs
                ):
                    generating.add(production.lhs)
                    changed = True
        generating_productions = {
            production
            for production in productions
            if production.lhs in generating
            and all(not isinstance(symbol, TypeSymbol) or symbol.type in generating for symbol in production.rhs)
        }
        by_lhs: dict[TypeExpr, set[Production]] = {}
        for production in generating_productions:
            by_lhs.setdefault(production.lhs, set()).add(production)
        reachable = {self.start}
        queue = deque((self.start,))
        while queue:
            current = queue.popleft()
            for production in by_lhs.get(current, ()):
                for symbol in production.rhs:
                    if isinstance(symbol, TypeSymbol) and symbol.type not in reachable:
                        reachable.add(symbol.type)
                        queue.append(symbol.type)
        return {
            production
            for production in generating_productions
            if production.lhs in reachable
            and all(not isinstance(symbol, TypeSymbol) or symbol.type in reachable for symbol in production.rhs)
        }


def to_chomsky_normal_form(productions: Iterable[Production], start: TypeExpr = START_TYPE) -> set[Production]:
    return _ChomskyNormalFormConverter(start).convert(productions)


def run(command_line: CommandLineArguments) -> tuple[GeneratedGrammar, Path]:
    options = GeneratorOptions(
        module_name=command_line.module_name,
        normalize_chomsky_normal_form=command_line.normalize_chomsky_normal_form,
        emit_parameterized_signatures=command_line.emit_parameterized_signatures,
        module_alias=command_line.module_alias,
    )
    output_file = default_output_file(
        command_line.module_name,
        command_line.normalize_chomsky_normal_form,
        command_line.emit_parameterized_signatures,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    grammar = CfgGenerator(options).generate()
    output_file.write_text(f"{grammar.text}\n", encoding="utf-8")
    print(
        f"Wrote |P|={grammar.production_count}, |V|={grammar.nonterminal_count}, "
        f"|Σ|={grammar.terminal_count} to {output_file}"
    )
    return grammar, output_file


def main(args: Sequence[str] | None = None) -> int:
    try:
        command_line = CommandLineArguments.parse(sys.argv[1:] if args is None else args)
        run(command_line)
    except (ImportError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
