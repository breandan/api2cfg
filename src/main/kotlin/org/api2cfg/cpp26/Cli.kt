package org.api2cfg.cpp26

import java.nio.file.Files
import java.nio.file.Path

data class Cpp26CommandLine(
  val headers: List<String>,
  val normalizeChomskyNormalForm: Boolean,
  val compiler: String,
  val standard: String,
  val standardLibrary: String?,
  val output: Path?,
) {
  companion object {
    fun parse(arguments: Array<String>): Cpp26CommandLine {
      val headers = mutableListOf<String>()
      var normalizeChomskyNormalForm = false
      var compiler = "clang++"
      var standard = "c++2c"
      var standardLibrary: String? = null
      var output: Path? = null
      var index = 0

      while (index < arguments.size) {
        val argument = arguments[index]
        fun followingValue(flag: String): String {
          require(index + 1 < arguments.size) { "$flag requires a value" }
          index += 1
          return arguments[index]
        }
        when {
          argument == "--cnf" -> normalizeChomskyNormalForm = true
          argument == "--header" -> headers += splitHeaders(followingValue("--header"))
          argument.startsWith("--header=") -> headers += splitHeaders(argument.substringAfter('='))
          argument == "--headers" -> headers += splitHeaders(followingValue("--headers"))
          argument.startsWith("--headers=") -> headers += splitHeaders(argument.substringAfter('='))
          argument == "--clang" -> compiler = followingValue("--clang")
          argument.startsWith("--clang=") -> compiler = argument.substringAfter('=')
          argument == "--std" -> standard = followingValue("--std")
          argument.startsWith("--std=") -> standard = argument.substringAfter('=')
          argument == "--stdlib" -> standardLibrary = normalizeStandardLibrary(followingValue("--stdlib"))
          argument.startsWith("--stdlib=") -> standardLibrary = normalizeStandardLibrary(argument.substringAfter('='))
          argument == "--output" -> output = Path.of(followingValue("--output"))
          argument.startsWith("--output=") -> output = Path.of(argument.substringAfter('='))
          else -> require(false) { "Unknown C++26 flag: $argument" }
        }
        index += 1
      }

      require(compiler.isNotBlank()) { "--clang must not be empty" }
      require(standard in setOf("c++26", "c++2c")) { "Expected --std=c++26 or --std=c++2c" }
      val normalizedHeaders = (headers.ifEmpty { Cpp26StandardLibraryCatalog.defaultHeaders })
        .map(Cpp26StandardLibraryCatalog::normalizeHeader)
        .distinct()
        .sorted()
      return Cpp26CommandLine(
        headers = normalizedHeaders,
        normalizeChomskyNormalForm = normalizeChomskyNormalForm,
        compiler = compiler,
        standard = standard,
        standardLibrary = standardLibrary,
        output = output,
      )
    }

    private fun splitHeaders(value: String): List<String> {
      require(value.isNotBlank()) { "--header must not be empty" }
      return value.split(',').map(String::trim).onEach { header ->
        require(header.isNotEmpty()) { "--header must not contain an empty name" }
      }
    }

    private fun normalizeStandardLibrary(value: String): String? = when (value) {
      "default" -> null
      "libc++", "libstdc++" -> value
      else -> throw IllegalArgumentException("Expected --stdlib=default, libc++, or libstdc++")
    }
  }
}

fun main(args: Array<String>) {
  val commandLine = Cpp26CommandLine.parse(args)
  val options = Cpp26GeneratorOptions(
    headers = commandLine.headers,
    normalizeChomskyNormalForm = commandLine.normalizeChomskyNormalForm,
    scannerOptions = Cpp26ScannerOptions(
      ClangFrontendOptions(
        compiler = commandLine.compiler,
        standard = commandLine.standard,
        standardLibrary = commandLine.standardLibrary,
      ),
    ),
  )
  val library = Cpp26CFGGenerator(options).generateLibrary()
  val output = commandLine.output ?: defaultCpp26OutputFile(
    commandLine.headers,
    commandLine.normalizeChomskyNormalForm,
  )
  output.parent?.let(Files::createDirectories)
  Files.writeString(output, "${library.grammar.text}\n")
  println(
    "Scanned ${library.typeGraph.allTypes.size} C++26 types from " +
        commandLine.headers.joinToString { "<$it>" },
  )
  println(
    "Wrote |P|=${library.grammar.productionCount}, |V|=${library.grammar.nonterminalCount}, " +
        "|Σ|=${library.grammar.terminalCount} to $output",
  )
}

fun defaultCpp26OutputFile(headers: Collection<String>, cnf: Boolean): Path {
  val stem = headers.distinct().sorted().joinToString("_")
  val extension = if (cnf) "cnf" else "cfg"
  return Path.of("gen", "cpp26", "$stem.$extension")
}
