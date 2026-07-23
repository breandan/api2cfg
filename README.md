# lib2cfg

Generate a context-free grammar from public Kotlin or Java APIs on the JVM classpath, or from reflected Python APIs.

### Generate a grammar

Run the generator with a package name:

```sh
./gradlew run --args='kotlin.collections'
```

By default, output is written to `gen/<package_name>.cfg`, with dots replaced by underscores. The `gen/` directory is created automatically if it does not exist. For example, the command above writes:

```text
gen/kotlin_collections.cfg
```

Output names and locations are chosen programmatically from the package name:

```sh
./gradlew run --args='java.util'
```

This writes:

```text
gen/java_util.cfg
```

To normalize the generated grammar to Chomsky normal form, add `--cnf`. CNF output defaults to a `.cnf` file:

```sh
./gradlew run --args='--cnf java.util'
```

This writes:

```text
gen/java_util.cnf
```

To emit pre-monomorphization productions with scoped type-parameter nonterminals, add `--parameterized`:

```sh
./gradlew run --args='--parameterized java.util'
```

This writes:

```text
gen/java_util.parameterized.cfg
```

Parameterized output is an ordinary CFG-shaped macro source. Type parameters are emitted as reserved `__TP_...` nonterminals with short stable hash suffixes. Implicit top bounds such as `java.lang.Object` are omitted from the readable name, while explicit bounds are rendered as punctuation-free `bound_...` fragments. Finite expansion domains are represented by unit productions through `__TP_DOMAIN_...` nonterminals. A later expansion pass can macro-expand a schema production by taking unit-closure alternatives for each `__TP_...` symbol and substituting those alternatives consistently through the production.

### Generate a Python grammar

The Python reflector has the same output convention and starts with NumPy:

```sh
python3 main.py numpy
```

This writes the monomorphic grammar to `gen/numpy.cfg`. To retain reflected type variables and parameterized NumPy dtype signatures, run:

```sh
python3 main.py --parameterized numpy
```

This writes `gen/numpy.parameterized.cfg`. `--cnf` is also supported and is mutually exclusive with `--parameterized`. The Python CLI shape is:

```text
[--cnf | --parameterized] [--alias=<identifier>] <module>
```

NumPy's library profile renders its canonical `numpy` namespace as `np`. Type nonterminals therefore use names such as `np.float32` and `np.ndarray<np.float32>`, and generated calls use the same alias. Sampled programs should run in a namespace containing `import numpy as np`. The rendered alias can be overridden with any non-keyword Python identifier:

```sh
python3 main.py --alias=npx numpy
```

For other modules, an explicit alias maps the requested import namespace to the spelling used by generated calls and qualified type names. For example, `python3 main.py --alias=jnp jax.numpy` uses generic reflection with the `jnp` spelling today and leaves room for a future JAX typing extension. The output filename continues to derive from the canonical module name, independently of its rendered alias.

The reflector scans the requested module's directly exported namespace, including aliases, compiled callables, classes, inherited public members, readable properties, and Python operator protocols. Ordinary callables use resolved runtime annotations when present, then their reflected/docstring signature, and finally a bounded variadic fallback. Optional parameters are omitted from the default productions rather than filled with unrelated values.

NumPy ufunc typing is strict. The generator asks `resolve_dtypes(..., casting="no")` about combinations from one restricted ground universe and accepts a row only when every resolved input dtype is exactly the requested dtype. It then factors those exact rows into equality schemas. For example, `divide` has one `ndarray<T>, ndarray<T> -> ndarray<T>` schema, comparisons have same-typed operands and a fixed `ndarray<bool_>` result, and intrinsically heterogeneous calls such as `ldexp(T, I) -> T` retain separate type variables. A multi-variable schema is emitted only when all combinations in the product of its variable domains are valid; correlated cases remain exact singleton schemas.

The same schemas produce infix, unary, and in-place operator spellings, so reflected `object`-typed numeric dunders cannot bypass the no-promotion rule. Bound `ufunc.outer` is retained because it has the same elementwise dtype relation. `at`, `reduce`, `accumulate`, and `reduceat` are omitted for now: their accumulator and casting rules need separate schemas, and reflecting their compiled signatures as untyped arrays would be unsound.

The default NumPy ground dtypes are `bool_`, `int32`, `int64`, `float32`, `float64`, `complex64`, and `complex128`. This deliberately excludes implicit promotions, aliases, unsigned and narrow-width integers, extended precision, strings, objects, datetimes, and timedeltas. The profile's ground universe can be replaced programmatically with `GeneratorOptions.ground_type_names`, for example `GeneratorOptions("numpy", ground_type_names=("float32", "float64"))`. A ufunc with no exact loop in the configured universe, such as `isnat` under the default set, is omitted instead of receiving an unsound `object` fallback.

Both NumPy output modes are derived from the same schemas. `--parameterized` writes scoped, role-labeled variables such as `__TP_DType_...`, `__TP_OperandDType_...`, `__TP_ResultDType_...`, and `__TP_Input2DType_...`, plus finite `__TP_DOMAIN_...` definitions whose readable portion summarizes the allowed types. The hash suffix keeps otherwise identical role names distinct across signature scopes. Repeated occurrences of one scoped variable denote one atomic choice; the file is a macro representation rather than an independently sampled CFG. The default `.cfg` consistently substitutes every allowed choice, so it is the monomorphized form of those schemas rather than a separately inferred set of dtype rows. Reflected, author-defined type-variable names remain intact rather than being replaced by NumPy roles.

Concrete NumPy scalar types have small, dtype-compatible literal domains. Typed one-dimensional arrays are emitted as `np.array([...], dtype=np.<dtype>)` and contain zero through three literal values by default; `GeneratorOptions.max_array_literal_values` controls that fixed cap. Ufunc names occur only in calls (including strict bound forms such as `np.add.outer(...)`), never as unconstrained bare values. Builtin type nonterminals use labels such as `builtins.int`, keeping them distinct from terminal call names in the line-oriented format. A submodule can be targeted independently, for example:

```sh
python3 main.py numpy.linalg
```

`LibraryProfile` is the separation point between the generic reflector and library-specific behavior. A profile records the canonical module and namespace, rendered alias, optional array type and ground type names, and an optional extension selector. The NumPy extension supplies strict dtype resolution and typed array literals; the scanner, signature renderer, alias mapping, and CFG machinery remain library-neutral. This is the intended seam for adding JAX, TensorFlow, or another library's typing rules without specializing the reflection core to that library.

### Generate a Python grammar from stubs

`main_pyi.py` is a static alternative to the runtime reflector. It discovers
installed files with `importlib.metadata`, parses `.pyi` annotations without
executing the target package, and writes the same line-oriented CFG format:

```sh
python3 main_pyi.py --alias=torch torch
```

This writes `gen/torch.cfg`. If the library is installed under a different
Python interpreter, select that environment explicitly:

```sh
python3 main_pyi.py \
  --python /opt/homebrew/anaconda3/bin/python \
  --alias=torch \
  torch
```

The scanner prefers an adjacent stub for the requested module. When a
`py.typed` package has no root stub, it can also follow a static
`TYPE_CHECKING` star re-export into a `.pyi` file. PyTorch uses this arrangement
for the generated signatures in `torch._C._VariableFunctions`, so the command
above emits public calls such as:

```text
torch.Tensor -> torch . tensor ( builtins.object )
torch.Tensor -> torch . matmul ( torch.Tensor , torch.Tensor )
```

An explicit stub can be scanned when distribution metadata is unavailable:

```sh
python3 main_pyi.py fixture.api \
  --stub /path/to/fixture/api.pyi \
  --source-module fixture.api \
  --alias=fixture
```

The supported CLI is:

```text
[--cnf] [--alias NAME] [--python PATH] [--output PATH]
[--stub PATH] [--source-module MODULE] [--api-module MODULE]
[--max-vararg-arity N] <module-or-distribution>
```

The static scanner handles overloads, positional and keyword-only parameters,
bounded variadics, constructors, declared methods and properties, constants,
type/import aliases, type variables, unions, literals, generics, and forward
references. Optional arguments are omitted, matching `main.py`'s minimal-call
policy. It does not yet perform `ty` semantic member queries, recursively scan
an entire distribution API, infer runtime-only compiled members, or merge
inherited members from dependencies.

### Generate for another library

The package must be visible on the generator runtime classpath. JDK packages are available automatically, and this project already includes the Kotlin runtime needed for Kotlin standard library packages. For a Maven dependency, add it to `build.gradle.kts`:

```kotlin
dependencies {
  implementation("com.example:example-library:1.2.3")
}
```

Then run the generator for a package exported by that dependency:

```sh
./gradlew run --args='com.example.library'
```

For a local jar, build the application distribution and run with an explicit classpath:

```sh
./gradlew installDist
java -cp 'build/install/lib2cfg/lib/*:/path/to/library.jar' org.lib2cfg.MainKt com.example.library
```

The scanner reads classes directly under the requested package from the JDK runtime image, classpath directories, and classpath jars.

### JVM CLI

The command accepts one optional flag and one required positional argument:

```text
[--cnf | --parameterized] <package>
```

Examples:

```sh
./gradlew run --args='kotlin.collections'
./gradlew run --args='--cnf kotlin.collections'
./gradlew run --args='--parameterized kotlin.collections'
```
