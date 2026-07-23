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
    """Declarative import and qualified-name rendering for one library."""

    module_name: str
    namespace_module: str
    alias: str | None = None

    @staticmethod
    def for_module(
        module_name: str,
        *,
        alias: str | None = None,
    ) -> LibraryProfile:
        if alias is not None and not _is_valid_module_alias(alias):
            raise ValueError("module_alias must be a non-keyword Python identifier")
        root_name = module_name.split(".", 1)[0]
        return LibraryProfile(
            module_name=module_name,
            namespace_module=module_name if alias is not None else root_name,
            alias=alias,
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


@dataclass(frozen=True)
class GeneratorOptions:
    module_name: str
    monomorphization_depth: int = 2
    max_type_arguments_per_variable: int = 26
    max_type_variables_per_callable: int = 2
    max_vararg_arity: int = 3
    normalize_chomsky_normal_form: bool = False
    emit_parameterized_signatures: bool = False
    module_alias: str | None = None


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
_PROTOCOL_DUNDERS = {
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
        if not text or lowered in {"any", "object"}:
            return OBJECT_TYPE
        if lowered in {"none", "nonetype"}:
            return NONE_TYPE
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

    # Some APIs document optional groups both as defaults and as brackets.
    # Remove bracket groups completely because this generator emits
    # the minimal call. Importantly, required text outside a group survives:
    # ``call([start,] stop[, step,])`` becomes ``call(stop)``.
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
            # informative than a callable's documented signature.
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
    # descriptors, whereas C-level doc signatures may already be written as
    # calls on the receiver. A successful signature on the unbound class
    # member includes its receiver; a parsed doc signature does not.
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
        self.options = options
        self.profile = LibraryProfile.for_module(
            options.module_name,
            alias=options.module_alias,
        )
        self.scanner = PythonModuleScanner()
        self.module: types.ModuleType | None = None
        self.renderer: TypeRenderer | None = None
        self.members: list[ExportedMember] = []
        self.parameterized_name_keys: dict[str, str] = {}
        self.type_argument_types: tuple[TypeExpr, ...] = ()

    def generate(self) -> GeneratedGrammar:
        module = importlib.import_module(self.options.module_name)
        return self.generate_module(module)

    def generate_module(self, module: types.ModuleType) -> GeneratedGrammar:
        self.module = module
        self.members = self.scanner.exported_members(module)
        self.renderer = TypeRenderer(module, self.members, self.profile)
        renderer = self.renderer
        self.type_argument_types = self._build_type_argument_types()
        productions: set[Production] = set(self._literal_productions())

        callable_members = [member for member in self.members if callable(member.value)]
        class_members = [
            member
            for member in callable_members
            if inspect.isclass(member.value)
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
            productions.update(self._top_level_callable_productions(member.name, member.value))

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

    def _constructor_productions(self, token_name: str, type_class: type) -> list[Production]:
        result_type = self._type_renderer.render_class(type_class)
        if inspect.isabstract(type_class):
            return []
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

    def _top_level_callable_productions(self, name: str, value: object) -> list[Production]:
        signatures = _reflected_signatures(
            value,
            self._type_renderer,
            self.options.max_vararg_arity,
        )
        return self._emit_signatures(
            signatures,
            scope=f"function|{self.options.module_name}|{name}",
            rhs_builder=lambda parameters: _rhs_for_call(self._module_access_prefix(name), parameters),
        )

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
                signatures = _reflected_signatures(
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
            # constructor calls such as ``Vector(...) += x`` to nest inside
            # another expression.
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
