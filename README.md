# lib2cfg

Generate a context-free grammar from public Kotlin or Java APIs in a package on the JVM classpath.

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

### CLI

The command accepts one optional flag and one required positional argument:

```text
[--cnf] <package>
```

Examples:

```sh
./gradlew run --args='kotlin.collections'
./gradlew run --args='--cnf kotlin.collections'
```