package org.api2cfg.cpp26

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CompletableFuture

data class ClangFrontendOptions(
  val compiler: String = "clang++",
  val standard: String = "c++2c",
  val standardLibrary: String? = null,
  val extraArguments: List<String> = emptyList(),
) {
  init {
    require(standard in setOf("c++26", "c++2c")) { "Expected --std=c++26 or --std=c++2c" }
    require(standardLibrary == null || standardLibrary in setOf("libc++", "libstdc++")) {
      "Expected --stdlib=libc++ or --stdlib=libstdc++"
    }
  }
}

class ClangFrontendException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/** Runs Clang without a shell and streams its declaration AST into Kotlin. */
internal class ClangFrontend(
  private val options: ClangFrontendOptions,
  private val reader: ClangAstJsonReader = ClangAstJsonReader(),
) {
  fun scan(target: CppScanTarget): List<ClangAstNode> {
    val temporaryDirectory = Files.createTempDirectory("api2cfg-cpp26-")
    val probe = temporaryDirectory.resolve("probe.cpp")
    return try {
      Files.writeString(probe, probeSource(target))
      runClang(target, probe)
    } finally {
      runCatching { Files.deleteIfExists(probe) }
      runCatching { Files.deleteIfExists(temporaryDirectory) }
    }
  }

  private fun runClang(target: CppScanTarget, probe: Path): List<ClangAstNode> {
    val command = buildList {
      add(options.compiler)
      add("-x")
      add("c++")
      add("-std=${options.standard}")
      options.standardLibrary?.let { add("-stdlib=$it") }
      add("-fsyntax-only")
      add("-fno-color-diagnostics")
      add("-Xclang")
      add("-ast-dump=json")
      add("-Xclang")
      add("-ast-dump-filter=${target.astFilter}")
      addAll(options.extraArguments)
      add(probe.toString())
    }

    val process = try {
      ProcessBuilder(command).start()
    } catch (error: Exception) {
      throw ClangFrontendException(
        "Could not start '${options.compiler}'. Install Clang or pass --clang=<path>.",
        error,
      )
    }
    val stderr = CompletableFuture.supplyAsync { process.errorStream.bufferedReader().use { it.readText() } }
    val nodes = try {
      process.inputStream.use(reader::read)
    } catch (error: Exception) {
      process.destroyForcibly()
      throw ClangFrontendException("Could not parse Clang AST for <${target.header}>", error)
    }
    val exitCode = process.waitFor()
    val diagnostic = stderr.join().trim()
    if (exitCode != 0) {
      throw ClangFrontendException(
        buildString {
          append("Clang failed while scanning <${target.header}> with ${options.standard}")
          if (diagnostic.isNotEmpty()) append(":\n").append(diagnostic)
        },
      )
    }
    if (nodes.isEmpty()) {
      throw ClangFrontendException(
        "Clang produced no declaration named ${target.astFilter} from <${target.header}>",
      )
    }
    return nodes
  }

  private fun probeSource(target: CppScanTarget): String = buildString {
    append("#include <").append(target.header).append(">\n")
    target.instantiation?.let { append(it).append('\n') }
  }
}
