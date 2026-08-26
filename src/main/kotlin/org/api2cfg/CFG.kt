package org.api2cfg

enum class TargetLanguage {
  KOTLIN, JAVA, CPP;

  companion object {
    fun fromPackageName(packageName: String): TargetLanguage =
      if (packageName.startsWith("kotlin.")) KOTLIN else JAVA
  }
}

data class CFGCall(
  val result: TypeExpr,
  val receiver: TypeExpr?,
  val staticOwner: String?,
  val name: String,
  val parameters: List<TypeExpr>,
) {
  init {
    require(receiver == null || staticOwner == null) {
      "A call cannot have both an instance receiver and a static owner"
    }
  }
}

/**
 * An immutable context-free grammar over type-shaped nonterminals.
 *
 * API scanners supply grounded calls and type relationships; this class owns
 * their [productions], cleanup, start rules, Chomsky normalization, and rendering.
 */
data class CFG(
  val productions: Set<Production>,
  val start: TypeExpr = DefaultStart,
) {
  constructor(
    productions: Iterable<Production>,
    start: TypeExpr = DefaultStart,
  ) : this(productions.toSet(), start)

  fun withStartProductions(): CFG {
    val startProductions = productions
      .asSequence()
      .map { production -> production.lhs }
      .filter { type -> type != start }
      .distinct()
      .sortedBy(TypeExpr::render)
      .map { type -> Production.unit(start, type) }
      .toSet()
    return copy(productions = productions + startProductions)
  }

  fun withoutUndefinedNonterminals(): CFG {
    var remaining = productions
    while (true) {
      val definedTypes = remaining.mapTo(mutableSetOf()) { production -> production.lhs }
      val pruned = remaining
        .filter { production ->
          production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in definedTypes }
        }
        .toSet()
      if (pruned.size == remaining.size) return copy(productions = pruned)
      remaining = pruned
    }
  }

  fun withoutNonGeneratingProductions(): CFG {
    val generating = linkedSetOf<TypeExpr>()
    var changed: Boolean
    do {
      changed = false
      for (production in productions) {
        if (production.lhs in generating) continue
        if (production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in generating }) {
          changed = generating.add(production.lhs) || changed
        }
      }
    } while (changed)

    return copy(
      productions = productions.filterTo(linkedSetOf()) { production ->
        production.lhs in generating &&
            production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in generating }
      },
    )
  }

  fun toChomskyNormalForm(): CFG =
    copy(productions = ChomskyNormalFormConverter(start).convert(productions))

  fun toGeneratedGrammar(): GeneratedGrammar = GeneratedGrammar.from(productions)

  companion object {
    val DefaultStart: TypeExpr = TypeExpr.Applied("START")

    fun fromCalls(
      calls: Iterable<CFGCall>,
      targetLanguage: TargetLanguage,
      includeNullableTypes: Boolean = false,
      start: TypeExpr = DefaultStart,
      subtypeRelation: ((actual: TypeExpr, expected: TypeExpr) -> Boolean)? = null,
    ): CFG {
      val seedProductions = buildSet {
        addAll(PrimitiveLiteralRules.rules(targetLanguage, includeNullableTypes))
        calls.mapTo(this) { call -> call.toProduction() }
      }
      val mentionedTypes = seedProductions
        .flatMapTo(linkedSetOf()) { production -> production.types() }
      val subtypeProductions = if (subtypeRelation == null) {
        emptySet()
      } else {
        unitProductions(mentionedTypes, subtypeRelation)
      }
      return CFG(seedProductions + subtypeProductions, start)
    }

    fun unitProductions(
      types: Iterable<TypeExpr>,
      relation: (actual: TypeExpr, expected: TypeExpr) -> Boolean,
    ): Set<Production> {
      val distinctTypes = types.distinct()
      return buildSet {
        for (expected in distinctTypes) {
          for (actual in distinctTypes) {
            if (actual != expected && relation(actual, expected)) {
              add(Production.unit(expected, actual))
            }
          }
        }
      }
    }
  }
}

data class GeneratedGrammar(
  val text: String,
  val productionCount: Int,
  val nonterminalCount: Int,
  val terminalCount: Int,
) {
  companion object {
    fun from(productions: Iterable<Production>): GeneratedGrammar {
      val sortedProductions = productions
        .sortedWith(compareBy<Production> { it.lhs.render() }.thenBy { it.rhs.joinToString(" ") { symbol -> symbol.render() } })
      val nonterminals = linkedSetOf<TypeExpr>()
      val terminals = linkedSetOf<String>()

      for (production in sortedProductions) {
        nonterminals += production.lhs
        for (symbol in production.rhs) {
          when (symbol) {
            is Symbol.Type -> nonterminals += symbol.type
            is Symbol.Token -> terminals += symbol.value
          }
        }
      }

      return GeneratedGrammar(
        text = sortedProductions.joinToString("\n") { it.render() },
        productionCount = sortedProductions.size,
        nonterminalCount = nonterminals.size,
        terminalCount = terminals.size,
      )
    }
  }
}

data class Production(val lhs: TypeExpr, val rhs: List<Symbol>) {
  fun render(): String = "${lhs.render()} -> ${rhs.joinToString(" ") { it.render() }}"

  fun types(): Set<TypeExpr> = buildSet {
    add(lhs)
    rhs.forEach { symbol ->
      if (symbol is Symbol.Type) add(symbol.type)
    }
  }

  companion object {
    fun unit(lhs: TypeExpr, rhs: TypeExpr): Production =
      Production(lhs, listOf(Symbol.Type(rhs)))

    fun literal(type: TypeExpr, token: String): Production =
      Production(type, listOf(Symbol.Token(token)))

    fun call(
      result: TypeExpr,
      name: String,
      receiver: TypeExpr?,
      parameters: List<TypeExpr>,
      staticOwner: String?,
    ): Production = Production(
      lhs = result,
      rhs = buildList {
        if (staticOwner != null) {
          add(Symbol.Token(staticOwner))
          add(Symbol.Token("."))
        } else if (receiver != null) {
          add(Symbol.Type(receiver))
          add(Symbol.Token("."))
        }
        add(Symbol.Token(name))
        add(Symbol.Token("("))
        parameters.forEachIndexed { index, parameter ->
          if (index > 0) add(Symbol.Token(","))
          add(Symbol.Type(parameter))
        }
        add(Symbol.Token(")"))
      },
    )
  }
}

private fun CFGCall.toProduction(): Production =
  Production.call(
    result = result,
    name = name,
    receiver = receiver,
    parameters = parameters,
    staticOwner = staticOwner,
  )

sealed interface Symbol {
  fun render(): String

  data class Type(val type: TypeExpr) : Symbol {
    override fun render(): String = type.render()
  }

  data class Token(val value: String) : Symbol {
    override fun render(): String = value.cfgToken()
  }
}

sealed interface TypeExpr {
  fun render(): String

  data class Applied(
    val name: String,
    val arguments: List<TypeExpr> = emptyList(),
    val nullable: Boolean = false,
  ) : TypeExpr {
    override fun render(): String {
      val renderedArguments = if (arguments.isEmpty()) "" else arguments.joinToString(",", "<", ">") { it.render() }
      return "$name$renderedArguments${if (nullable) "?" else ""}"
    }
  }

  data class Variable(val name: String) : TypeExpr {
    override fun render(): String = name.cfgToken()
  }
}

internal fun TypeExpr.variables(): Set<String> = when (this) {
  is TypeExpr.Variable -> setOf(name)
  is TypeExpr.Applied -> arguments.flatMapTo(linkedSetOf()) { argument -> argument.variables() }
}

internal fun TypeExpr.substitute(substitution: Map<String, TypeExpr>): TypeExpr? = when (this) {
  is TypeExpr.Variable -> substitution[name]
  is TypeExpr.Applied -> copy(
    arguments = arguments.map { argument -> argument.substitute(substitution) ?: return null },
  )
}

internal fun TypeExpr.isGround(): Boolean = variables().isEmpty()

internal fun TypeExpr.depth(): Int = when (this) {
  is TypeExpr.Variable -> 0
  is TypeExpr.Applied -> if (arguments.isEmpty()) 0 else 1 + arguments.maxOf(TypeExpr::depth)
}

data class LiteralRule(val typeName: String, val literalToken: String)

object PrimitiveLiteralRules {
  fun typeNames(targetLanguage: TargetLanguage): List<String> =
    rulesFor(targetLanguage).map { rule -> rule.typeName }

  fun rules(targetLanguage: TargetLanguage, includeNullableTypes: Boolean): List<Production> {
    val nullRules = if (includeNullableTypes && targetLanguage == TargetLanguage.KOTLIN) {
      listOf(Production.literal(TypeExpr.Applied("Nothing", nullable = true), "null"))
    } else {
      emptyList()
    }

    return rulesFor(targetLanguage).flatMap { (typeName, literalToken) ->
      buildList {
        add(Production.literal(TypeExpr.Applied(typeName), literalToken))
        if (includeNullableTypes) {
          add(Production.literal(TypeExpr.Applied(typeName, nullable = true), "null"))
        }
      }
    } + nullRules
  }

  private fun rulesFor(targetLanguage: TargetLanguage): List<LiteralRule> = when (targetLanguage) {
    TargetLanguage.KOTLIN -> KotlinLiteralRules
    TargetLanguage.JAVA -> JavaLiteralRules
    TargetLanguage.CPP -> CppLiteralRules
  }
}

private val KotlinLiteralRules = listOf(
  LiteralRule("Boolean", "true"),
  LiteralRule("Byte", "0"),
  LiteralRule("Char", "'x'"),
  LiteralRule("Double", "0.0"),
  LiteralRule("Float", "0.0f"),
  LiteralRule("Int", "0"),
  LiteralRule("Long", "0L"),
  LiteralRule("Short", "0"),
  LiteralRule("String", "\"s\""),
  LiteralRule("UByte", "0u"),
  LiteralRule("UInt", "0u"),
  LiteralRule("ULong", "0UL"),
  LiteralRule("UShort", "0u"),
)

private val JavaLiteralRules = listOf(
  LiteralRule("boolean", "true"),
  LiteralRule("byte", "0"),
  LiteralRule("char", "'x'"),
  LiteralRule("double", "0.0"),
  LiteralRule("float", "0.0f"),
  LiteralRule("int", "0"),
  LiteralRule("long", "0L"),
  LiteralRule("short", "0"),
  LiteralRule("Boolean", "true"),
  LiteralRule("Byte", "0"),
  LiteralRule("Character", "'x'"),
  LiteralRule("Double", "0.0"),
  LiteralRule("Float", "0.0f"),
  LiteralRule("Integer", "0"),
  LiteralRule("Long", "0L"),
  LiteralRule("Short", "0"),
  LiteralRule("String", "\"s\""),
)

private val CppLiteralRules = listOf(
  LiteralRule("bool", "true"),
  LiteralRule("char", "'x'"),
  LiteralRule("double", "0.0"),
  LiteralRule("float", "0.0f"),
  LiteralRule("int", "0"),
  LiteralRule("long", "0L"),
  LiteralRule("long_long", "0LL"),
  LiteralRule("unsigned", "0u"),
  LiteralRule("unsigned_long", "0ul"),
  LiteralRule("unsigned_long_long", "0ull"),
)

fun toChomskyNormalForm(productions: Iterable<Production>, start: TypeExpr): Set<Production> =
  CFG(productions, start).toChomskyNormalForm().productions

private class ChomskyNormalFormConverter(private val start: TypeExpr) {
  private val terminalNonterminals = linkedMapOf<String, TypeExpr>()
  private val suffixNonterminals = linkedMapOf<List<Symbol.Type>, TypeExpr>()
  private var terminalCounter = 0
  private var suffixCounter = 0

  fun convert(productions: Iterable<Production>): Set<Production> {
    val normalized = linkedSetOf<Production>()
    productions
      .filter { it.rhs.isNotEmpty() }
      .sortedWith(compareBy<Production> { it.lhs.render() }.thenBy { it.rhs.joinToString(" ") { symbol -> symbol.render() } })
      .forEach { production -> normalized += normalizeShape(production) }

    terminalNonterminals.forEach { (terminal, nonterminal) ->
      normalized += Production.literal(nonterminal, terminal)
    }

    val withoutUnits = eliminateUnitProductions(normalized)
    val useful = pruneToUsefulProductions(withoutUnits)
    require(useful.all(::isChomskyNormalForm)) {
      "CNF conversion produced non-CNF productions"
    }
    require(allProductionsAreUseful(useful)) {
      "CNF conversion produced unreachable or non-generating productions"
    }
    return useful
  }

  private fun normalizeShape(production: Production): Set<Production> {
    val rhs = if (production.rhs.size == 1) {
      production.rhs
    } else {
      production.rhs.map { symbol ->
        when (symbol) {
          is Symbol.Type -> symbol
          is Symbol.Token -> Symbol.Type(terminalNonterminal(symbol.value))
        }
      }
    }

    return when (rhs.size) {
      1, 2 -> setOf(Production(production.lhs, rhs))
      else -> binarize(production.lhs, rhs.map { symbol -> symbol as Symbol.Type })
    }
  }

  private fun binarize(lhs: TypeExpr, rhs: List<Symbol.Type>): Set<Production> {
    val result = linkedSetOf<Production>()
    var currentLhs = lhs
    for (index in 0 until rhs.size - 2) {
      val suffix = rhs.drop(index + 1)
      val suffixType = suffixNonterminal(suffix)
      result += Production(currentLhs, listOf(rhs[index], Symbol.Type(suffixType)))
      currentLhs = suffixType
    }
    result += Production(currentLhs, rhs.takeLast(2))
    return result
  }

  private fun eliminateUnitProductions(productions: Set<Production>): Set<Production> {
    val unitTargets = productions
      .filter { production -> production.rhs.size == 1 && production.rhs.single() is Symbol.Type }
      .groupBy(
        keySelector = { production -> production.lhs },
        valueTransform = { production -> (production.rhs.single() as Symbol.Type).type },
      )
    val nonUnitProductions = productions
      .filterNot { production -> production.rhs.size == 1 && production.rhs.single() is Symbol.Type }
    val nonUnitsByLhs = nonUnitProductions.groupBy { production -> production.lhs }
    val nonterminals = productions.flatMapTo(mutableSetOf()) { production -> production.types() }

    val result = linkedSetOf<Production>()
    for (source in nonterminals.sortedBy(TypeExpr::render)) {
      val closure = unitClosure(source, unitTargets)
      for (target in closure) {
        for (production in nonUnitsByLhs[target].orEmpty()) {
          result += production.copy(lhs = source)
        }
      }
    }
    return result
  }

  private fun unitClosure(
    source: TypeExpr,
    unitTargets: Map<TypeExpr, List<TypeExpr>>,
  ): Set<TypeExpr> {
    val closure = linkedSetOf(source)
    val queue = ArrayDeque<TypeExpr>()
    queue += source
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      for (target in unitTargets[current].orEmpty()) {
        if (closure.add(target)) queue += target
      }
    }
    return closure
  }

  private fun pruneToUsefulProductions(productions: Set<Production>): Set<Production> {
    val generating = generatingNonterminals(productions)
    val generatingProductions = productions
      .filter { production ->
        production.lhs in generating &&
            production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in generating }
      }
      .toSet()
    val reachable = reachableNonterminals(generatingProductions)
    return generatingProductions
      .filter { production ->
        production.lhs in reachable &&
            production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in reachable }
      }
      .toSet()
  }

  private fun generatingNonterminals(productions: Set<Production>): Set<TypeExpr> {
    val generating = linkedSetOf<TypeExpr>()
    var changed: Boolean
    do {
      changed = false
      for (production in productions) {
        if (production.lhs in generating) continue
        if (production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in generating }) {
          changed = generating.add(production.lhs) || changed
        }
      }
    } while (changed)
    return generating
  }

  private fun reachableNonterminals(productions: Set<Production>): Set<TypeExpr> {
    val byLhs = productions.groupBy { production -> production.lhs }
    val reachable = linkedSetOf(start)
    val queue = ArrayDeque<TypeExpr>()
    queue += start
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      for (production in byLhs[current].orEmpty()) {
        for (symbol in production.rhs) {
          if (symbol is Symbol.Type && reachable.add(symbol.type)) queue += symbol.type
        }
      }
    }
    return reachable
  }

  private fun isChomskyNormalForm(production: Production): Boolean = when (production.rhs.size) {
    1 -> production.rhs.single() is Symbol.Token
    2 -> production.rhs.all { symbol -> symbol is Symbol.Type }
    else -> false
  }

  private fun allProductionsAreUseful(productions: Set<Production>): Boolean {
    val generating = generatingNonterminals(productions)
    val reachable = reachableNonterminals(productions)
    return productions.all { production ->
      production.lhs in generating &&
          production.lhs in reachable &&
          production.rhs.all { symbol ->
            symbol !is Symbol.Type || (symbol.type in generating && symbol.type in reachable)
          }
    }
  }

  private fun terminalNonterminal(terminal: String): TypeExpr =
    terminalNonterminals.getOrPut(terminal) {
      terminalCounter += 1
      TypeExpr.Applied(
        "__CNF_T_${terminal.sanitizedCnfNamePart()}_${terminalCounter.toString().padStart(4, '0')}",
      )
    }

  private fun suffixNonterminal(suffix: List<Symbol.Type>): TypeExpr =
    suffixNonterminals.getOrPut(suffix) {
      suffixCounter += 1
      TypeExpr.Applied("__CNF_N_${suffixCounter.toString().padStart(6, '0')}")
    }
}

private fun String.cfgToken(): String = replace(Regex("\\s+"), "_")

private fun String.sanitizedCnfNamePart(): String {
  val sanitized = map { character -> if (character.isLetterOrDigit()) character else '_' }.joinToString("")
    .trim('_')
    .ifBlank { "TOKEN" }
  return sanitized.take(32)
}
