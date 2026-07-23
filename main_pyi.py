#!/usr/bin/env python3
"""Generate a CFG from statically parsed Python stub annotations.

Unlike ``main.py``, this module never imports the target package.  It discovers
installed files with ``importlib.metadata``, parses ``.pyi`` files with
``ast``, and uses inline ``.py`` files only to resolve type aliases or
``TYPE_CHECKING`` re-exports when an adjacent root stub does not exist.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import json
import keyword
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, cast

import main as cfg


_DISCOVERY_SCRIPT = r"""
import importlib.metadata as metadata
import json
import sys

requested = sys.argv[1]
root = requested.split(".", 1)[0]
distribution_names = list(metadata.packages_distributions().get(root, ()))
for candidate in (requested, root):
    try:
        distribution_names.append(metadata.distribution(candidate).metadata["Name"])
    except metadata.PackageNotFoundError:
        pass

records = []
seen_distributions = set()
for distribution_name in distribution_names:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        continue
    canonical_name = distribution.metadata.get("Name", distribution_name)
    canonical_key = canonical_name.lower().replace("_", "-")
    if canonical_key in seen_distributions:
        continue
    seen_distributions.add(canonical_key)
    files = []
    for entry in distribution.files or ():
        text = str(entry)
        if not (text.endswith(".pyi") or text.endswith(".py") or text.endswith("/py.typed")):
            continue
        located = distribution.locate_file(entry)
        files.append({"relative": text, "path": str(located)})
    records.append({"name": canonical_name, "files": files})

print(json.dumps({
    "python": sys.executable,
    "sys_path": sys.path,
    "distributions": records,
}, sort_keys=True))
"""


_BUILTIN_NAMES = {
    "bool",
    "bytes",
    "complex",
    "dict",
    "float",
    "int",
    "list",
    "object",
    "set",
    "slice",
    "str",
    "tuple",
}
_ANY_NAMES = {"Any", "typing.Any", "typing_extensions.Any"}
_NONE_NAMES = {"None", "NoneType", "types.NoneType"}
_ELLIPSIS_NAMES = {"EllipsisType", "types.EllipsisType"}
_NEVER_NAMES = {
    "Never",
    "NoReturn",
    "typing.Never",
    "typing.NoReturn",
    "typing_extensions.Never",
    "typing_extensions.NoReturn",
}
_SELF_NAMES = {"Self", "typing.Self", "typing_extensions.Self"}
_TYPE_GUARD_NAMES = {
    "TypeGuard",
    "TypeIs",
    "typing.TypeGuard",
    "typing_extensions.TypeGuard",
    "typing_extensions.TypeIs",
}
_UNWRAPPED_GENERIC_NAMES = {
    "Annotated",
    "ClassVar",
    "Final",
    "NotRequired",
    "ReadOnly",
    "Required",
    "Unpack",
    "typing.Annotated",
    "typing.ClassVar",
    "typing.Final",
    "typing.NotRequired",
    "typing.Required",
    "typing_extensions.Annotated",
    "typing_extensions.NotRequired",
    "typing_extensions.ReadOnly",
    "typing_extensions.Required",
    "typing_extensions.Unpack",
}
_ABSTRACT_CONTAINER_NAMES = {
    "Collection": "list",
    "Container": "list",
    "Iterable": "list",
    "Iterator": "list",
    "MutableSequence": "list",
    "Reversible": "list",
    "Sequence": "list",
    "AbstractSet": "set",
    "MutableSet": "set",
    "Mapping": "dict",
    "MutableMapping": "dict",
}
_NON_CONSTRUCTIBLE_BASE_NAMES = {
    "Enum",
    "IntEnum",
    "Protocol",
    "TypedDict",
    "enum.Enum",
    "enum.IntEnum",
    "typing.Protocol",
    "typing.TypedDict",
    "typing_extensions.Protocol",
    "typing_extensions.TypedDict",
}


class StubDiscoveryError(ValueError):
    """Raised when no usable static stub source can be found."""


@dataclass(frozen=True)
class CommandLineArguments:
    target: str
    python: Path = Path(sys.executable)
    output: Path | None = None
    alias: str | None = None
    api_module: str | None = None
    source_module: str | None = None
    stub: Path | None = None
    normalize_chomsky_normal_form: bool = False
    max_vararg_arity: int = 3

    @staticmethod
    def parse(args: Sequence[str]) -> CommandLineArguments:
        positional: list[str] = []
        python = Path(sys.executable)
        output: Path | None = None
        alias: str | None = None
        api_module: str | None = None
        source_module: str | None = None
        stub: Path | None = None
        cnf = False
        max_vararg_arity = 3

        index = 0
        while index < len(args):
            argument = args[index]
            if argument == "--cnf":
                cnf = True
            elif argument in {
                "--python",
                "--output",
                "--alias",
                "--api-module",
                "--source-module",
                "--stub",
                "--max-vararg-arity",
            }:
                index += 1
                if index >= len(args):
                    raise ValueError(f"{argument} requires a value")
                value = args[index]
                if argument == "--python":
                    python = Path(value)
                elif argument == "--output":
                    output = Path(value)
                elif argument == "--alias":
                    alias = value
                elif argument == "--api-module":
                    api_module = value
                elif argument == "--source-module":
                    source_module = value
                elif argument == "--stub":
                    stub = Path(value)
                else:
                    try:
                        max_vararg_arity = int(value)
                    except ValueError as error:
                        raise ValueError("--max-vararg-arity must be an integer") from error
            elif argument.startswith("--"):
                flag, separator, value = argument.partition("=")
                if not separator:
                    raise ValueError(f"Unknown flag: {argument}")
                if flag == "--python":
                    python = Path(value)
                elif flag == "--output":
                    output = Path(value)
                elif flag == "--alias":
                    alias = value
                elif flag == "--api-module":
                    api_module = value
                elif flag == "--source-module":
                    source_module = value
                elif flag == "--stub":
                    stub = Path(value)
                elif flag == "--max-vararg-arity":
                    try:
                        max_vararg_arity = int(value)
                    except ValueError as error:
                        raise ValueError("--max-vararg-arity must be an integer") from error
                else:
                    raise ValueError(f"Unknown flag: {flag}")
            elif argument.startswith("-"):
                raise ValueError(f"Unknown flag: {argument}")
            else:
                positional.append(argument)
            index += 1

        if len(positional) != 1:
            raise ValueError("Expected exactly one positional argument: <module-or-distribution>")
        if alias is not None and (not alias.isidentifier() or keyword.iskeyword(alias)):
            raise ValueError("--alias must be a non-keyword Python identifier")
        for flag, module_name in (
            ("--api-module", api_module),
            ("--source-module", source_module),
        ):
            if module_name is not None and not _is_module_name(module_name):
                raise ValueError(f"{flag} must be a dotted Python module name")
        if max_vararg_arity < 0:
            raise ValueError("--max-vararg-arity must be non-negative")
        return CommandLineArguments(
            target=positional[0],
            python=python,
            output=output,
            alias=alias,
            api_module=api_module,
            source_module=source_module,
            stub=stub,
            normalize_chomsky_normal_form=cnf,
            max_vararg_arity=max_vararg_arity,
        )


def _is_module_name(value: str) -> bool:
    return bool(value) and all(part.isidentifier() and not keyword.iskeyword(part) for part in value.split("."))


@dataclass(frozen=True)
class ModuleFile:
    module: str
    path: Path
    is_stub: bool
    distribution: str = ""


class EnvironmentIndex:
    """A static index of Python source files owned by the selected distribution."""

    def __init__(self, files: Iterable[ModuleFile]):
        choices: dict[str, list[ModuleFile]] = {}
        for module_file in files:
            choices.setdefault(module_file.module, []).append(module_file)
        self._choices = {
            module: tuple(
                sorted(
                    module_files,
                    key=lambda item: (
                        not item.is_stub,
                        _distribution_stub_priority(item.distribution),
                        str(item.path),
                    ),
                )
            )
            for module, module_files in choices.items()
        }

    @classmethod
    def discover(cls, python: Path, requested: str) -> EnvironmentIndex:
        if not python.exists():
            raise StubDiscoveryError(f"Python interpreter does not exist: {python}")
        try:
            completed = subprocess.run(
                [str(python), "-I", "-c", _DISCOVERY_SCRIPT, requested],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise StubDiscoveryError(
                f"Unable to inspect installed files with {python}: {detail.strip()}"
            ) from error
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise StubDiscoveryError(
                f"Interpreter {python} returned invalid discovery data"
            ) from error

        files: list[ModuleFile] = []
        for distribution in payload.get("distributions", ()):
            distribution_name = str(distribution.get("name", ""))
            for record in distribution.get("files", ()):
                relative = str(record.get("relative", ""))
                module = _module_name_from_relative_file(relative)
                if module is None:
                    continue
                path = Path(str(record.get("path", "")))
                if path.is_file():
                    files.append(
                        ModuleFile(
                            module=module,
                            path=path,
                            is_stub=path.suffix == ".pyi",
                            distribution=distribution_name,
                        )
                    )

        # Metadata can be absent for an editable/local package.  Looking only
        # for the exact requested module on the selected interpreter's sys.path
        # remains static and does not execute package initializers.
        requested_parts = requested.split(".")
        for raw_entry in payload.get("sys_path", ()):
            if not raw_entry:
                continue
            entry = Path(raw_entry)
            candidates = (
                entry.joinpath(*requested_parts).with_suffix(".pyi"),
                entry.joinpath(*requested_parts, "__init__.pyi"),
                entry.joinpath(*requested_parts).with_suffix(".py"),
                entry.joinpath(*requested_parts, "__init__.py"),
            )
            for candidate in candidates:
                if candidate.is_file():
                    files.append(
                        ModuleFile(
                            requested,
                            candidate,
                            candidate.suffix == ".pyi",
                        )
                    )
        return cls(files)

    @classmethod
    def for_explicit_stub(
        cls,
        stub: Path,
        source_module: str,
    ) -> EnvironmentIndex:
        resolved = stub.resolve()
        if not resolved.is_file():
            raise StubDiscoveryError(f"Stub file does not exist: {stub}")
        if resolved.suffix != ".pyi":
            raise StubDiscoveryError(f"Expected a .pyi file, got: {stub}")

        relative_parts = source_module.split(".")
        if resolved.name == "__init__.pyi":
            levels = len(relative_parts)
        else:
            levels = len(relative_parts) - 1
        root = resolved.parent
        for _ in range(levels):
            root = root.parent

        files: list[ModuleFile] = [ModuleFile(source_module, resolved, True)]
        top_level = relative_parts[0]
        package_root = root / top_level
        if package_root.is_dir():
            for path in sorted(
                itertools.chain(
                    package_root.rglob("*.pyi"),
                    package_root.rglob("*.py"),
                )
            ):
                module = _module_name_from_path(root, path)
                if module is not None:
                    files.append(ModuleFile(module, path, path.suffix == ".pyi"))
        return cls(files)

    def file_for(self, module: str, *, stub_only: bool = False) -> ModuleFile | None:
        for module_file in self._choices.get(module, ()):
            if not stub_only or module_file.is_stub:
                return module_file
        return None

    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(self._choices))


def _distribution_stub_priority(name: str) -> int:
    normalized = name.lower().replace("_", "-")
    return 0 if normalized.startswith("types-") or "stub" in normalized else 1


def _module_name_from_relative_file(relative: str) -> str | None:
    path = Path(relative)
    if path.suffix not in {".py", ".pyi"}:
        return None
    parts = list(path.parts)
    if not parts:
        return None
    if parts[0].endswith("-stubs"):
        # PEP 561 stub-only distributions use ``package-stubs`` on disk while
        # describing the import namespace ``package``.
        parts[0] = parts[0].removesuffix("-stubs")
    filename = parts.pop()
    stem = filename.removesuffix(".pyi").removesuffix(".py")
    if stem != "__init__":
        parts.append(stem)
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _module_name_from_path(root: Path, path: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    return _module_name_from_relative_file(str(relative))


@dataclass(frozen=True)
class TypeVariable:
    name: str
    constraints: tuple[ast.expr, ...] = ()
    bound: ast.expr | None = None


@dataclass
class StaticModule:
    name: str
    path: Path
    is_stub: bool
    is_package: bool
    tree: ast.Module
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    attributes: dict[str, ast.AnnAssign] = field(default_factory=dict)
    assignments: dict[str, ast.expr] = field(default_factory=dict)
    type_aliases: dict[str, ast.expr] = field(default_factory=dict)
    type_variables: dict[str, TypeVariable] = field(default_factory=dict)
    imported_symbols: dict[str, str] = field(default_factory=dict)
    imported_modules: dict[str, str] = field(default_factory=dict)
    star_imports: list[str] = field(default_factory=list)
    type_checking_star_imports: list[str] = field(default_factory=list)
    explicit_reexports: set[str] = field(default_factory=set)
    all_names: set[str] | None = None

    @classmethod
    def parse(cls, module_file: ModuleFile) -> StaticModule:
        try:
            source = module_file.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = module_file.path.read_text()
        tree = ast.parse(source, filename=str(module_file.path), type_comments=True)
        model = cls(
            name=module_file.module,
            path=module_file.path,
            is_stub=module_file.is_stub,
            is_package=module_file.path.name in {"__init__.py", "__init__.pyi"},
            tree=tree,
        )
        model._collect()
        return model

    def _collect(self) -> None:
        all_names: set[str] = set()
        saw_all_operation = False
        for statement, under_type_checking in _module_statements(self.tree.body):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(statement.name, []).append(statement)
                continue
            if isinstance(statement, ast.ClassDef):
                self.classes[statement.name] = statement
                continue
            if isinstance(statement, ast.Import):
                for imported in statement.names:
                    local_name = imported.asname or imported.name.split(".", 1)[0]
                    self.imported_modules[local_name] = imported.name
                    if imported.asname == imported.name:
                        self.explicit_reexports.add(local_name)
                continue
            if isinstance(statement, ast.ImportFrom):
                source_module = self._absolute_import(statement.module, statement.level)
                for imported in statement.names:
                    if imported.name == "*":
                        self.star_imports.append(source_module)
                        if under_type_checking:
                            self.type_checking_star_imports.append(source_module)
                        continue
                    local_name = imported.asname or imported.name
                    target = f"{source_module}.{imported.name}".strip(".")
                    self.imported_symbols[local_name] = target
                    if imported.asname == imported.name:
                        self.explicit_reexports.add(local_name)
                continue
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                name = statement.target.id
                self.attributes[name] = statement
                if statement.value is not None:
                    self.assignments[name] = statement.value
                if _annotation_name(statement.annotation).endswith("TypeAlias") and statement.value is not None:
                    self.type_aliases[name] = statement.value
                continue
            if isinstance(statement, ast.Assign):
                names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
                for name in names:
                    if name == "__all__":
                        values = _literal_string_collection(statement.value)
                        if values is not None:
                            all_names = set(values)
                            saw_all_operation = True
                        continue
                    self.assignments[name] = statement.value
                    type_variable = _type_variable_from_assignment(name, statement.value)
                    if type_variable is not None:
                        self.type_variables[name] = type_variable
                continue
            if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                if statement.target.id == "__all__" and isinstance(statement.op, ast.Add):
                    values = _literal_string_collection(statement.value)
                    if values is not None:
                        all_names.update(values)
                        saw_all_operation = True
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    if call.func.value.id == "__all__" and call.args:
                        if call.func.attr == "append":
                            value = _literal_string(call.args[0])
                            if value is not None:
                                all_names.add(value)
                                saw_all_operation = True
                        elif call.func.attr == "extend":
                            values = _literal_string_collection(call.args[0])
                            if values is not None:
                                all_names.update(values)
                                saw_all_operation = True
                continue

            # Python 3.12's PEP 695 ``type Alias = ...`` node.
            type_alias_class = getattr(ast, "TypeAlias", None)
            if type_alias_class is not None and isinstance(statement, type_alias_class):
                alias_name = getattr(getattr(statement, "name", None), "id", None)
                alias_value = getattr(statement, "value", None)
                if isinstance(alias_name, str) and isinstance(alias_value, ast.expr):
                    self.type_aliases[alias_name] = alias_value

        self.all_names = all_names if saw_all_operation else None

    def _absolute_import(self, imported_module: str | None, level: int) -> str:
        if level == 0:
            return imported_module or ""
        package_parts = self.name.split(".") if self.is_package else self.name.split(".")[:-1]
        keep = max(0, len(package_parts) - (level - 1))
        prefix = package_parts[:keep]
        if imported_module:
            prefix.extend(imported_module.split("."))
        return ".".join(prefix)

    def public_names(self) -> set[str]:
        if self.all_names is not None:
            return set(self.all_names)
        declared = set(self.functions) | set(self.classes) | set(self.attributes)
        return {
            name
            for name in declared | self.explicit_reexports
            if not name.startswith("_")
        }


def _module_statements(
    statements: Sequence[ast.stmt],
    under_type_checking: bool = False,
) -> Iterator[tuple[ast.stmt, bool]]:
    for statement in statements:
        yield statement, under_type_checking
        if isinstance(statement, ast.If):
            nested_type_checking = under_type_checking or _is_type_checking_test(statement.test)
            yield from _module_statements(statement.body, nested_type_checking)
            yield from _module_statements(statement.orelse, under_type_checking)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            yield from _module_statements(statement.body, under_type_checking)
            for handler in statement.handlers:
                yield from _module_statements(handler.body, under_type_checking)
            yield from _module_statements(statement.orelse, under_type_checking)
            yield from _module_statements(statement.finalbody, under_type_checking)


def _is_type_checking_test(node: ast.expr) -> bool:
    name = _annotation_name(node)
    return name == "TYPE_CHECKING" or name.endswith(".TYPE_CHECKING")


def _literal_string(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _literal_string_collection(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_collection(node.left)
        right = _literal_string_collection(node.right)
        if left is not None and right is not None:
            return (*left, *right)
        return None
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    values = tuple(_literal_string(element) for element in node.elts)
    return tuple(value for value in values if value is not None) if all(value is not None for value in values) else None


def _annotation_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _annotation_name(node.value)
        return f"{prefix}.{node.attr}".strip(".")
    return ""


def _type_variable_from_assignment(name: str, value: ast.expr) -> TypeVariable | None:
    if not isinstance(value, ast.Call):
        return None
    constructor = _annotation_name(value.func).rsplit(".", 1)[-1]
    if constructor not in {"TypeVar", "ParamSpec", "TypeVarTuple"}:
        return None
    constraints = tuple(value.args[1:]) if constructor == "TypeVar" else ()
    bound = next(
        (keyword.value for keyword in value.keywords if keyword.arg == "bound"),
        None,
    )
    return TypeVariable(name, constraints, bound)


def _looks_like_type_expression(node: ast.expr) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _looks_like_type_expression(node.left) and _looks_like_type_expression(node.right)
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, str)
    return False


@dataclass(frozen=True)
class ResolvedSymbol:
    model: StaticModule
    name: str
    kind: str
    value: object


class StubRepository:
    def __init__(self, index: EnvironmentIndex):
        self.index = index
        self._modules: dict[str, StaticModule] = {}
        self.diagnostics: list[str] = []

    def module(self, name: str, *, stub_only: bool = False) -> StaticModule | None:
        cached = self._modules.get(name)
        if cached is not None and (not stub_only or cached.is_stub):
            return cached
        module_file = self.index.file_for(name, stub_only=stub_only)
        if module_file is None:
            return None
        try:
            parsed = StaticModule.parse(module_file)
        except (OSError, SyntaxError) as error:
            self._diagnose(f"{module_file.path}: {error}")
            return None
        self._modules[name] = parsed
        return parsed

    def resolve_symbol(
        self,
        model: StaticModule,
        name: str,
        visited: frozenset[tuple[str, str]] = frozenset(),
    ) -> ResolvedSymbol | None:
        key = (model.name, name)
        if key in visited:
            return None
        visited = visited | {key}
        if name in model.functions:
            return ResolvedSymbol(model, name, "function", tuple(model.functions[name]))
        if name in model.classes:
            return ResolvedSymbol(model, name, "class", model.classes[name])
        if name in model.attributes:
            return ResolvedSymbol(model, name, "attribute", model.attributes[name])

        target = model.imported_symbols.get(name)
        if target is not None:
            target_model_name, target_name = self.split_reference(target)
            if target_model_name is not None and target_name is not None:
                target_model = self.module(target_model_name)
                if target_model is not None:
                    return self.resolve_symbol(target_model, target_name, visited)
        for star_module_name in model.star_imports:
            star_model = self.module(star_module_name)
            if star_model is None:
                continue
            if star_model.all_names is not None and name not in star_model.all_names:
                continue
            resolved = self.resolve_symbol(star_model, name, visited)
            if resolved is not None:
                return resolved
        return None

    def split_reference(self, reference: str) -> tuple[str | None, str | None]:
        parts = reference.split(".")
        for split_at in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:split_at])
            if self.index.file_for(module_name) is not None:
                return module_name, ".".join(parts[split_at:])
        return None, None

    def _diagnose(self, message: str) -> None:
        if message not in self.diagnostics and len(self.diagnostics) < 50:
            self.diagnostics.append(message)


@dataclass(frozen=True)
class Projection:
    source: StaticModule
    api_module: str
    public_names: frozenset[str]


@dataclass(frozen=True)
class GeneratorOptions:
    api_module: str
    alias: str | None = None
    max_vararg_arity: int = 3
    normalize_chomsky_normal_form: bool = False


class AnnotationRenderer:
    def __init__(
        self,
        repository: StubRepository,
        options: GeneratorOptions,
        public_type_names: Mapping[str, str],
    ):
        self.repository = repository
        self.options = options
        self.profile = cfg.LibraryProfile.for_module(options.api_module, alias=options.alias)
        self.public_type_names = dict(public_type_names)

    def render(
        self,
        node: ast.expr | None,
        model: StaticModule,
        *,
        owner: cfg.TypeExpr | None = None,
        local_type_variables: frozenset[str] = frozenset(),
        alias_stack: frozenset[tuple[str, str]] = frozenset(),
    ) -> cfg.TypeExpr:
        if node is None:
            return cfg.OBJECT_TYPE
        if isinstance(node, ast.Constant):
            if node.value is None:
                return cfg.NONE_TYPE
            if node.value is Ellipsis:
                return cfg.TypeExpr.applied("...")
            if isinstance(node.value, str):
                try:
                    parsed = ast.parse(node.value, mode="eval")
                except SyntaxError:
                    return self._qualified_or_unknown(node.value, model)
                return self.render(
                    parsed.body,
                    model,
                    owner=owner,
                    local_type_variables=local_type_variables,
                    alias_stack=alias_stack,
                )
            return cfg.TypeExpr.applied(type(node.value).__name__)
        if isinstance(node, ast.Name):
            return self._render_name(
                node.id,
                model,
                owner=owner,
                local_type_variables=local_type_variables,
                alias_stack=alias_stack,
            )
        if isinstance(node, ast.Attribute):
            reference = self._reference_name(node, model)
            return self._render_reference(
                reference,
                model,
                owner=owner,
                local_type_variables=local_type_variables,
                alias_stack=alias_stack,
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            alternatives = _unique_types(
                (
                    self.render(
                        node.left,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                        alias_stack=alias_stack,
                    ),
                    self.render(
                        node.right,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                        alias_stack=alias_stack,
                    ),
                )
            )
            return alternatives[0] if len(alternatives) == 1 else cfg.TypeExpr.applied("Union", *alternatives)
        if isinstance(node, ast.Subscript):
            base_reference = self._reference_name(node.value, model)
            arguments = _subscript_arguments(node.slice)
            short_base = base_reference.rsplit(".", 1)[-1]
            if base_reference in _UNWRAPPED_GENERIC_NAMES or short_base in _UNWRAPPED_GENERIC_NAMES:
                return self.render(
                    arguments[0] if arguments else None,
                    model,
                    owner=owner,
                    local_type_variables=local_type_variables,
                    alias_stack=alias_stack,
                )
            if short_base == "Optional":
                rendered = self.render(
                    arguments[0] if arguments else None,
                    model,
                    owner=owner,
                    local_type_variables=local_type_variables,
                    alias_stack=alias_stack,
                )
                return cfg.TypeExpr.applied("Union", rendered, cfg.NONE_TYPE)
            if short_base == "Union":
                alternatives = _unique_types(
                    self.render(
                        argument,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                        alias_stack=alias_stack,
                    )
                    for argument in arguments
                )
                return alternatives[0] if len(alternatives) == 1 else cfg.TypeExpr.applied("Union", *alternatives)
            if short_base == "Literal":
                literal_types = _unique_types(self._literal_type(argument) for argument in arguments)
                return literal_types[0] if len(literal_types) == 1 else cfg.TypeExpr.applied("Union", *literal_types)
            if base_reference in _TYPE_GUARD_NAMES or short_base in _TYPE_GUARD_NAMES:
                return cfg.TypeExpr.applied("bool")
            rendered_arguments: list[cfg.TypeExpr] = []
            for argument in arguments:
                if isinstance(argument, (ast.List, ast.Tuple)) and short_base == "Callable":
                    rendered_arguments.extend(
                        self.render(
                            element,
                            model,
                            owner=owner,
                            local_type_variables=local_type_variables,
                            alias_stack=alias_stack,
                        )
                        for element in argument.elts
                    )
                else:
                    rendered_arguments.append(
                        self.render(
                            argument,
                            model,
                            owner=owner,
                            local_type_variables=local_type_variables,
                            alias_stack=alias_stack,
                        )
                    )
            base = self._render_reference(
                base_reference,
                model,
                owner=owner,
                local_type_variables=local_type_variables,
                alias_stack=alias_stack,
            )
            return cfg.TypeExpr.applied(base.name, *rendered_arguments)
        if isinstance(node, ast.Starred):
            return self.render(
                node.value,
                model,
                owner=owner,
                local_type_variables=local_type_variables,
                alias_stack=alias_stack,
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return cfg.TypeExpr.applied(
                "tuple" if isinstance(node, ast.Tuple) else "list",
                *(
                    self.render(
                        element,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                        alias_stack=alias_stack,
                    )
                    for element in node.elts
                ),
            )
        self.repository._diagnose(
            f"{model.path}:{getattr(node, 'lineno', '?')}: unsupported annotation {type(node).__name__}"
        )
        return cfg.OBJECT_TYPE

    def _render_name(
        self,
        name: str,
        model: StaticModule,
        *,
        owner: cfg.TypeExpr | None,
        local_type_variables: frozenset[str],
        alias_stack: frozenset[tuple[str, str]],
    ) -> cfg.TypeExpr:
        if name in local_type_variables or name in model.type_variables:
            return cfg.TypeExpr.variable(name)
        if name in _BUILTIN_NAMES:
            return cfg.TypeExpr.applied(name)
        if name in _ANY_NAMES:
            return cfg.OBJECT_TYPE
        if name in _NONE_NAMES:
            return cfg.NONE_TYPE
        if name in _ELLIPSIS_NAMES:
            return cfg.TypeExpr.applied("...")
        if name in _NEVER_NAMES:
            return cfg.TypeExpr.applied("Never")
        if name in _SELF_NAMES:
            return owner or cfg.OBJECT_TYPE
        if name == "AnyStr":
            return cfg.TypeExpr.applied(
                "Union",
                cfg.TypeExpr.applied("str"),
                cfg.TypeExpr.applied("bytes"),
            )
        if name == "LiteralString":
            return cfg.TypeExpr.applied("str")
        if name in model.imported_symbols:
            return self._render_reference(
                model.imported_symbols[name],
                model,
                owner=owner,
                local_type_variables=local_type_variables,
                alias_stack=alias_stack,
            )
        alias = model.type_aliases.get(name)
        if alias is None:
            assignment = model.assignments.get(name)
            if assignment is not None and _looks_like_type_expression(assignment):
                alias = assignment
        alias_key = (model.name, name)
        if alias is not None and alias_key not in alias_stack:
            return self.render(
                alias,
                model,
                owner=owner,
                local_type_variables=local_type_variables,
                alias_stack=alias_stack | {alias_key},
            )
        if name in model.classes:
            return cfg.TypeExpr.applied(self._render_qualified(f"{model.name}.{name}"))
        return self._qualified_or_unknown(name, model)

    def _render_reference(
        self,
        reference: str,
        context: StaticModule,
        *,
        owner: cfg.TypeExpr | None,
        local_type_variables: frozenset[str],
        alias_stack: frozenset[tuple[str, str]],
    ) -> cfg.TypeExpr:
        if not reference:
            return cfg.OBJECT_TYPE
        short_name = reference.rsplit(".", 1)[-1]
        if reference in _ANY_NAMES or short_name == "Any":
            return cfg.OBJECT_TYPE
        if reference in _NONE_NAMES:
            return cfg.NONE_TYPE
        if reference in _ELLIPSIS_NAMES or short_name == "EllipsisType":
            return cfg.TypeExpr.applied("...")
        if reference in _NEVER_NAMES or short_name in {"Never", "NoReturn"}:
            return cfg.TypeExpr.applied("Never")
        if reference in _SELF_NAMES or short_name == "Self":
            return owner or cfg.OBJECT_TYPE
        variable_prefix = reference.split(".", 1)[0]
        if variable_prefix in local_type_variables or variable_prefix in context.type_variables:
            # ParamSpec ``P.args``/``P.kwargs`` and similar variadic type
            # projections do not denote independently constructible values.
            return cfg.OBJECT_TYPE
        if short_name in _BUILTIN_NAMES and reference.startswith("builtins."):
            return cfg.TypeExpr.applied(short_name)
        if short_name in _ABSTRACT_CONTAINER_NAMES:
            return cfg.TypeExpr.applied(_ABSTRACT_CONTAINER_NAMES[short_name])
        if short_name in {"Callable", "Pattern", "Match", "IO"} and reference.startswith(
            ("typing.", "collections.abc.", "re.")
        ):
            return cfg.TypeExpr.applied(short_name)
        if short_name == "LiteralString":
            return cfg.TypeExpr.applied("str")
        if short_name == "AnyStr":
            return cfg.TypeExpr.applied(
                "Union",
                cfg.TypeExpr.applied("str"),
                cfg.TypeExpr.applied("bytes"),
            )

        module_name, symbol_path = self.repository.split_reference(reference)
        if module_name is not None and symbol_path is not None and "." not in symbol_path:
            # A name exported from the requested API keeps its public spelling.
            # Following ``torch.Tensor`` into the implementation import
            # ``torch._tensor.Tensor`` would otherwise leak a private module
            # into every generated PyTorch signature.
            if (
                module_name == self.options.api_module
                or self.options.api_module.startswith(f"{module_name}.")
            ):
                return cfg.TypeExpr.applied(self._render_qualified(reference))
            target_model = self.repository.module(module_name)
            if target_model is not None:
                if symbol_path in target_model.type_variables:
                    return cfg.TypeExpr.variable(target_model.type_variables[symbol_path].name)
                target_alias = target_model.type_aliases.get(symbol_path)
                if target_alias is None:
                    assignment = target_model.assignments.get(symbol_path)
                    if assignment is not None and _looks_like_type_expression(assignment):
                        target_alias = assignment
                alias_key = (module_name, symbol_path)
                if target_alias is not None and alias_key not in alias_stack:
                    return self.render(
                        target_alias,
                        target_model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                        alias_stack=alias_stack | {alias_key},
                    )
                imported = target_model.imported_symbols.get(symbol_path)
                if imported is not None:
                    return self._render_reference(
                        imported,
                        target_model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                        alias_stack=alias_stack,
                    )
        return cfg.TypeExpr.applied(self._render_qualified(reference))

    def _reference_name(self, node: ast.expr, model: StaticModule) -> str:
        if isinstance(node, ast.Name):
            if node.id in model.imported_modules:
                return model.imported_modules[node.id]
            if node.id in model.imported_symbols:
                return model.imported_symbols[node.id]
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._reference_name(node.value, model)
            return f"{prefix}.{node.attr}".strip(".")
        return _annotation_name(node)

    def _qualified_or_unknown(self, name: str, model: StaticModule) -> cfg.TypeExpr:
        if "." in name:
            return cfg.TypeExpr.applied(self._render_qualified(name))
        # Names that are not definitions or imports are usually forward
        # references to a declaration in the same stub module.
        return cfg.TypeExpr.applied(self._render_qualified(f"{model.name}.{name}"))

    def _render_qualified(self, qualified_name: str) -> str:
        public = self.public_type_names.get(qualified_name, qualified_name)
        return self.profile.render_qualified_name(public)

    @staticmethod
    def _literal_type(node: ast.expr) -> cfg.TypeExpr:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return cfg.NONE_TYPE
            if node.value is Ellipsis:
                return cfg.TypeExpr.applied("...")
            return cfg.TypeExpr.applied(type(node.value).__name__)
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            return cfg.TypeExpr.applied(type(node.operand.value).__name__)
        return cfg.OBJECT_TYPE

    def type_variable_domain(
        self,
        variable: str,
        model: StaticModule,
        *,
        owner: cfg.TypeExpr | None = None,
    ) -> tuple[cfg.TypeExpr, ...]:
        definition = model.type_variables.get(variable)
        if definition is None and variable in model.imported_symbols:
            module_name, target_name = self.repository.split_reference(model.imported_symbols[variable])
            target_model = self.repository.module(module_name) if module_name is not None else None
            if target_model is not None and target_name is not None:
                definition = target_model.type_variables.get(target_name)
                model = target_model
        if definition is None:
            return (cfg.OBJECT_TYPE,)
        if definition.constraints:
            rendered = _unique_types(
                self.render(constraint, model, owner=owner)
                for constraint in definition.constraints
            )
            return rendered or (cfg.OBJECT_TYPE,)
        if definition.bound is not None:
            return (self.render(definition.bound, model, owner=owner),)
        return (cfg.OBJECT_TYPE,)


def _subscript_arguments(node: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.Tuple):
        return tuple(node.elts)
    return (node,)


def _unique_types(types: Iterable[cfg.TypeExpr]) -> tuple[cfg.TypeExpr, ...]:
    return tuple(dict.fromkeys(types))


@dataclass(frozen=True)
class ParsedParameter:
    name: str
    type: cfg.TypeExpr
    kind: inspect._ParameterKind
    has_default: bool = False
    is_vararg: bool = False


@dataclass(frozen=True)
class ParsedSignature:
    parameters: tuple[ParsedParameter, ...]
    return_type: cfg.TypeExpr


class PyiCfgGenerator:
    def __init__(
        self,
        repository: StubRepository,
        projections: Sequence[Projection],
        options: GeneratorOptions,
    ):
        self.repository = repository
        self.projections = tuple(projections)
        self.options = options
        self.public_type_names = self._public_type_names()
        self.renderer = AnnotationRenderer(repository, options, self.public_type_names)

    def generate(self) -> cfg.GeneratedGrammar:
        helper = cfg.CfgGenerator(cfg.GeneratorOptions(self.options.api_module))
        productions: set[cfg.Production] = set(helper._literal_productions())
        for projection in self.projections:
            for public_name in sorted(projection.public_names):
                symbol = self.repository.resolve_symbol(projection.source, public_name)
                if symbol is None:
                    continue
                if symbol.kind == "function":
                    function_nodes = cast(
                        tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
                        symbol.value,
                    )
                    productions.update(
                        self._function_productions(
                            projection,
                            public_name,
                            symbol.model,
                            function_nodes,
                        )
                    )
                elif symbol.kind == "class":
                    productions.update(
                        self._class_productions(
                            projection,
                            public_name,
                            symbol.model,
                            cast(ast.ClassDef, symbol.value),
                        )
                    )
                elif symbol.kind == "attribute":
                    production = self._attribute_production(
                        projection,
                        public_name,
                        symbol.model,
                        cast(ast.AnnAssign, symbol.value),
                    )
                    if production is not None:
                        productions.add(production)

        productions.update(helper._supporting_type_productions(productions))
        productions.update(self._extra_supporting_productions(productions))
        productions = helper._prune_undefined_nonterminals(productions)
        productions = helper._prune_non_generating_productions(productions)
        productions.update(helper._start_productions(productions))
        if self.options.normalize_chomsky_normal_form:
            productions = cfg.to_chomsky_normal_form(productions, cfg.START_TYPE)
        return cfg.GeneratedGrammar.from_productions(productions)

    def _public_type_names(self) -> dict[str, str]:
        candidates: dict[str, list[tuple[int, str]]] = {}
        for projection in self.projections:
            for public_name in sorted(projection.public_names):
                symbol = self.repository.resolve_symbol(projection.source, public_name)
                if symbol is None or symbol.kind != "class":
                    continue
                candidates.setdefault(
                    f"{symbol.model.name}.{symbol.name}",
                    [],
                ).append(
                    (
                        0 if public_name == symbol.name else 1,
                        f"{projection.api_module}.{public_name}",
                    )
                )
        return {
            canonical: min(names)[1]
            for canonical, names in candidates.items()
        }

    def _function_productions(
        self,
        projection: Projection,
        public_name: str,
        model: StaticModule,
        nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> set[cfg.Production]:
        productions: set[cfg.Production] = set()
        prefix = self._access_prefix((public_name,))
        for node in _signature_nodes(nodes):
            signature = self._parse_signature(node, model)
            for expanded in self._expand_signature(signature, model):
                productions.add(
                    cfg.Production(
                        expanded.return_type,
                        cfg._rhs_for_call(prefix, self._reflected_parameters(expanded.parameters)),
                    )
                )
        return productions

    def _class_productions(
        self,
        projection: Projection,
        public_name: str,
        model: StaticModule,
        node: ast.ClassDef,
    ) -> set[cfg.Production]:
        productions: set[cfg.Production] = set()
        owner = cfg.TypeExpr.applied(
            self.renderer._render_qualified(f"{model.name}.{node.name}")
        )
        base_names = {
            self.renderer._reference_name(base, model)
            for base in node.bases
        }
        non_constructible = any(
            base in _NON_CONSTRUCTIBLE_BASE_NAMES
            or base.rsplit(".", 1)[-1] in {"Enum", "IntEnum", "Protocol", "TypedDict"}
            for base in base_names
        )
        for base in node.bases:
            base_type = self._ground_type(
                self.renderer.render(base, model, owner=owner),
                model,
                owner=owner,
            )
            if base_type.name.rsplit(".", 1)[-1] not in {
                "Generic",
                "NamedTuple",
                "Protocol",
                "TypedDict",
            }:
                productions.add(cfg.Production(base_type, (cfg.TypeSymbol(owner),)))

        methods: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        attributes: list[ast.AnnAssign] = []
        enum_assignments: list[str] = []
        for statement, _ in _module_statements(node.body):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.setdefault(statement.name, []).append(statement)
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                attributes.append(statement)
            elif isinstance(statement, ast.Assign) and any(
                base.rsplit(".", 1)[-1] in {"Enum", "IntEnum"} for base in base_names
            ):
                enum_assignments.extend(
                    target.id for target in statement.targets if isinstance(target, ast.Name)
                )

        constructor_nodes = methods.get("__init__") or methods.get("__new__") or []
        if not non_constructible:
            if constructor_nodes:
                for constructor in _signature_nodes(constructor_nodes):
                    signature = self._parse_signature(
                        constructor,
                        model,
                        owner=owner,
                        strip_first=True,
                        forced_return=owner,
                    )
                    for expanded in self._expand_signature(signature, model, owner=owner):
                        productions.add(
                            cfg.Production(
                                owner,
                                cfg._rhs_for_call(
                                    self._access_prefix((public_name,)),
                                    self._reflected_parameters(expanded.parameters),
                                ),
                            )
                        )
            else:
                productions.add(
                    cfg.Production(
                        owner,
                        cfg._rhs_for_call(self._access_prefix((public_name,)), ()),
                    )
                )

        for method_name, method_nodes in sorted(methods.items()):
            if method_name in {"__init__", "__new__"}:
                continue
            if method_name.startswith("_") and method_name not in cfg.OPERATOR_SPECS:
                continue
            selected_nodes = _signature_nodes(method_nodes)
            for method in selected_nodes:
                decorators = {_annotation_name(decorator) for decorator in method.decorator_list}
                if any(name.endswith(".setter") or name.endswith(".deleter") for name in decorators):
                    continue
                is_property = any(name.rsplit(".", 1)[-1] == "property" for name in decorators)
                is_static = any(name.rsplit(".", 1)[-1] == "staticmethod" for name in decorators)
                is_class = any(name.rsplit(".", 1)[-1] == "classmethod" for name in decorators)
                signature = self._parse_signature(
                    method,
                    model,
                    owner=owner,
                    strip_first=not is_static,
                )
                for expanded in self._expand_signature(signature, model, owner=owner):
                    if is_property:
                        rhs = (
                            cfg.TypeSymbol(owner),
                            cfg.Token("."),
                            cfg.Token(method_name),
                        )
                    elif is_static or is_class:
                        rhs = cfg._rhs_for_call(
                            self._access_prefix((public_name, method_name)),
                            self._reflected_parameters(expanded.parameters),
                        )
                    else:
                        rhs = cfg._rhs_for_call(
                            (
                                cfg.TypeSymbol(owner),
                                cfg.Token("."),
                                cfg.Token(method_name),
                            ),
                            self._reflected_parameters(expanded.parameters),
                        )
                    productions.add(cfg.Production(expanded.return_type, rhs))

        for attribute in attributes:
            assert isinstance(attribute.target, ast.Name)
            name = attribute.target.id
            if name.startswith("_"):
                continue
            attribute_type = self._ground_type(
                self.renderer.render(attribute.annotation, model, owner=owner),
                model,
                owner=owner,
            )
            productions.add(
                cfg.Production(
                    attribute_type,
                    (cfg.TypeSymbol(owner), cfg.Token("."), cfg.Token(name)),
                )
            )
        for member_name in enum_assignments:
            if member_name.startswith("_"):
                continue
            productions.add(
                cfg.Production(
                    owner,
                    self._access_prefix((public_name, member_name)),
                )
            )
        return productions

    def _attribute_production(
        self,
        projection: Projection,
        public_name: str,
        model: StaticModule,
        node: ast.AnnAssign,
    ) -> cfg.Production | None:
        if _annotation_name(node.annotation).endswith("TypeAlias"):
            return None
        return cfg.Production(
            self._ground_type(self.renderer.render(node.annotation, model), model),
            self._access_prefix((public_name,)),
        )

    def _ground_type(
        self,
        type_expr: cfg.TypeExpr,
        model: StaticModule,
        *,
        owner: cfg.TypeExpr | None = None,
    ) -> cfg.TypeExpr:
        variables = sorted(type_expr.variables())
        if not variables:
            return type_expr
        substitutions = {
            variable: self.renderer.type_variable_domain(
                variable,
                model,
                owner=owner,
            )[0]
            for variable in variables
        }
        return type_expr.substitute(substitutions) or cfg.OBJECT_TYPE

    def _parse_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        model: StaticModule,
        *,
        owner: cfg.TypeExpr | None = None,
        strip_first: bool = False,
        forced_return: cfg.TypeExpr | None = None,
    ) -> ParsedSignature:
        local_type_variables = frozenset(
            getattr(parameter, "name", "")
            for parameter in getattr(node, "type_params", ())
            if getattr(parameter, "name", "")
        )
        positional = [*node.args.posonlyargs, *node.args.args]
        positional_kinds = [
            *([inspect.Parameter.POSITIONAL_ONLY] * len(node.args.posonlyargs)),
            *([inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(node.args.args)),
        ]
        default_start = len(positional) - len(node.args.defaults)
        parameters: list[ParsedParameter] = []
        for index, (argument, kind) in enumerate(zip(positional, positional_kinds)):
            parameters.append(
                ParsedParameter(
                    argument.arg,
                    self.renderer.render(
                        argument.annotation,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                    ),
                    kind,
                    has_default=index >= default_start,
                )
            )
        if node.args.vararg is not None:
            parameters.append(
                ParsedParameter(
                    node.args.vararg.arg,
                    self.renderer.render(
                        node.args.vararg.annotation,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                    ),
                    inspect.Parameter.VAR_POSITIONAL,
                    is_vararg=True,
                )
            )
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            parameters.append(
                ParsedParameter(
                    argument.arg,
                    self.renderer.render(
                        argument.annotation,
                        model,
                        owner=owner,
                        local_type_variables=local_type_variables,
                    ),
                    inspect.Parameter.KEYWORD_ONLY,
                    has_default=default is not None,
                )
            )
        # **kwargs always has a valid zero-argument realization and is omitted.
        if strip_first and parameters:
            parameters = parameters[1:]
        return_type = forced_return or self.renderer.render(
            node.returns,
            model,
            owner=owner,
            local_type_variables=local_type_variables,
        )
        return ParsedSignature(tuple(parameters), return_type)

    def _expand_signature(
        self,
        signature: ParsedSignature,
        model: StaticModule,
        *,
        owner: cfg.TypeExpr | None = None,
    ) -> tuple[ParsedSignature, ...]:
        required = tuple(parameter for parameter in signature.parameters if not parameter.has_default)
        vararg = next((parameter for parameter in required if parameter.is_vararg), None)
        arities = range(self.options.max_vararg_arity + 1) if vararg is not None else (0,)
        expanded_varargs: list[tuple[ParsedParameter, ...]] = []
        for arity in arities:
            replacements = tuple(
                ParsedParameter(
                    f"{vararg.name}{index + 1}",
                    vararg.type,
                    inspect.Parameter.POSITIONAL_ONLY,
                )
                for index in range(arity)
            ) if vararg is not None else ()
            expanded_parameters: list[ParsedParameter] = []
            for parameter in required:
                if parameter.is_vararg:
                    expanded_parameters.extend(replacements)
                else:
                    expanded_parameters.append(parameter)
            expanded_varargs.append(tuple(expanded_parameters))

        variables = sorted(
            set(signature.return_type.variables()).union(
                *(parameter.type.variables() for parameter in required)
            )
        )
        if len(variables) > 2:
            domains = [(cfg.OBJECT_TYPE,) for _ in variables]
        else:
            domains = [
                self.renderer.type_variable_domain(variable, model, owner=owner)
                for variable in variables
            ]
        substitutions = list(itertools.product(*domains))[:16] if variables else [()]

        result: list[ParsedSignature] = []
        for parameters in expanded_varargs:
            for values in substitutions:
                substitution = dict(zip(variables, values))
                return_type = (
                    signature.return_type.substitute(substitution)
                    if variables
                    else signature.return_type
                )
                if return_type is None:
                    continue
                substituted_parameters: list[ParsedParameter] = []
                for parameter in parameters:
                    parameter_type = (
                        parameter.type.substitute(substitution)
                        if variables
                        else parameter.type
                    )
                    if parameter_type is None:
                        break
                    substituted_parameters.append(
                        ParsedParameter(
                            parameter.name,
                            parameter_type,
                            parameter.kind,
                            parameter.has_default,
                            parameter.is_vararg,
                        )
                    )
                else:
                    result.append(
                        ParsedSignature(tuple(substituted_parameters), return_type)
                    )
        return tuple(dict.fromkeys(result))

    @staticmethod
    def _reflected_parameters(
        parameters: Sequence[ParsedParameter],
    ) -> tuple[cfg.ReflectedParameter, ...]:
        return tuple(
            cfg.ReflectedParameter(parameter.name, parameter.type, parameter.kind)
            for parameter in parameters
        )

    def _access_prefix(self, parts: Sequence[str]) -> tuple[cfg.Symbol, ...]:
        if self.options.alias is None:
            rendered_parts = tuple(parts)
        else:
            rendered_parts = (self.options.alias, *parts)
        result: list[cfg.Symbol] = []
        for index, part in enumerate(rendered_parts):
            if index:
                result.append(cfg.Token("."))
            result.append(cfg.Token(part))
        return tuple(result)

    @staticmethod
    def _extra_supporting_productions(
        productions: Iterable[cfg.Production],
    ) -> set[cfg.Production]:
        result: set[cfg.Production] = set()
        all_types = {
            nested
            for production in productions
            for type_expr in production.types()
            for nested in type_expr.walk()
        }
        for type_expr in all_types:
            if type_expr.is_variable:
                continue
            if type_expr.name == "Callable":
                result.add(
                    cfg.Production(
                        type_expr,
                        (
                            cfg.Token("lambda"),
                            cfg.Token("*"),
                            cfg.Token("args"),
                            cfg.Token(":"),
                            cfg.Token("0"),
                        ),
                    )
                )
            elif type_expr.name == "...":
                result.add(cfg.Production(type_expr, (cfg.Token("..."),)))
        return result


def _signature_nodes(
    nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    overloads = tuple(
        node
        for node in nodes
        if any(
            _annotation_name(decorator).rsplit(".", 1)[-1] == "overload"
            for decorator in node.decorator_list
        )
    )
    return overloads or tuple(nodes)


def discover_projections(
    command_line: CommandLineArguments,
) -> tuple[StubRepository, tuple[Projection, ...], str]:
    target_is_path = Path(command_line.target).suffix == ".pyi" and Path(command_line.target).is_file()
    explicit_stub = command_line.stub or (Path(command_line.target) if target_is_path else None)
    inferred_source: str | None
    if target_is_path and command_line.source_module is None:
        inferred_source = Path(command_line.target).stem
    else:
        inferred_source = command_line.source_module

    api_module = command_line.api_module or (
        command_line.target if not target_is_path else inferred_source
    )
    if api_module is None or not _is_module_name(api_module):
        raise StubDiscoveryError(
            "A dotted --api-module is required when the positional target is a path"
        )
    source_module = inferred_source or api_module
    if not _is_module_name(source_module):
        raise StubDiscoveryError("--source-module must be a dotted Python module name")

    if explicit_stub is not None:
        index = EnvironmentIndex.for_explicit_stub(explicit_stub, source_module)
    else:
        index = EnvironmentIndex.discover(command_line.python, command_line.target)
    repository = StubRepository(index)

    source = repository.module(source_module, stub_only=True)
    if source is not None:
        return (
            repository,
            (Projection(source, api_module, frozenset(source.public_names())),),
            api_module,
        )

    # Some py.typed packages, notably PyTorch, expose generated compiled APIs
    # through a TYPE_CHECKING star import but ship no package-root __init__.pyi.
    inline_module = repository.module(source_module)
    if inline_module is None:
        available = ", ".join(
            module for module in index.modules() if index.file_for(module, stub_only=True)
        )
        suffix = f" Available stub modules: {available[:500]}" if available else ""
        raise StubDiscoveryError(
            f"No .pyi source found for {source_module} in {command_line.python}.{suffix}"
        )
    projections: list[Projection] = []
    for imported_module in dict.fromkeys(inline_module.type_checking_star_imports):
        imported_stub = repository.module(imported_module, stub_only=True)
        if imported_stub is None:
            continue
        names = {
            name
            for name in imported_stub.public_names()
            if not name.startswith("_")
            or (inline_module.all_names is not None and name in inline_module.all_names)
        }
        projections.append(
            Projection(imported_stub, api_module, frozenset(names))
        )
    if not projections:
        raise StubDiscoveryError(
            f"{source_module} has no adjacent .pyi file or TYPE_CHECKING star re-export from one"
        )
    return repository, tuple(projections), api_module


def default_output_file(api_module: str, cnf: bool = False) -> Path:
    suffix = "cnf" if cnf else "cfg"
    return Path("gen", f"{api_module.replace('.', '_')}.{suffix}")


def run(
    command_line: CommandLineArguments,
) -> tuple[cfg.GeneratedGrammar, Path]:
    repository, projections, api_module = discover_projections(command_line)
    options = GeneratorOptions(
        api_module=api_module,
        alias=command_line.alias,
        max_vararg_arity=command_line.max_vararg_arity,
        normalize_chomsky_normal_form=command_line.normalize_chomsky_normal_form,
    )
    grammar = PyiCfgGenerator(repository, projections, options).generate()
    output_file = command_line.output or default_output_file(
        api_module,
        command_line.normalize_chomsky_normal_form,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(f"{grammar.text}\n", encoding="utf-8")
    for diagnostic in repository.diagnostics:
        print(f"warning: {diagnostic}", file=sys.stderr)
    source_paths = ", ".join(str(projection.source.path) for projection in projections)
    print(
        f"Wrote |P|={grammar.production_count}, |V|={grammar.nonterminal_count}, "
        f"|Σ|={grammar.terminal_count} to {output_file} from {source_paths}"
    )
    return grammar, output_file


def main(args: Sequence[str] | None = None) -> int:
    try:
        command_line = CommandLineArguments.parse(sys.argv[1:] if args is None else args)
        run(command_line)
    except (OSError, StubDiscoveryError, SyntaxError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
