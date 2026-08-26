package org.api2cfg.cpp26

import org.api2cfg.CFG
import org.api2cfg.GeneratedGrammar
import org.api2cfg.PrimitiveLiteralRules
import org.api2cfg.Production
import org.api2cfg.Symbol
import org.api2cfg.TargetLanguage
import org.api2cfg.TypeExpr

data class Cpp26GeneratorOptions(
  val headers: List<String> = Cpp26StandardLibraryCatalog.defaultHeaders,
  val normalizeChomskyNormalForm: Boolean = false,
  val maxCallArity: Int = 3,
  val scannerOptions: Cpp26ScannerOptions = Cpp26ScannerOptions(),
) {
  init {
    require(maxCallArity in 0..3) { "The C++26 fragment supports call arities zero through three" }
  }
}

data class GeneratedCpp26Library(
  val typeGraph: CppTypeGraph,
  val grammar: GeneratedGrammar,
)

/**
 * Generates the fragment
 *
 * `statement ::= expression ;`
 *
 * where expressions are literals, values, construction, named member/static
 * calls, and `operator()` calls with at most three arguments. Other operators,
 * field access, pointer syntax, variadics, and dependent signatures are omitted.
 */
class Cpp26CFGGenerator(private val options: Cpp26GeneratorOptions = Cpp26GeneratorOptions()) {
  fun generate(): GeneratedGrammar = generateLibrary().grammar

  fun generateLibrary(): GeneratedCpp26Library {
    val graph = Cpp26Scanner(options.scannerOptions).scan(options.headers)
    return GeneratedCpp26Library(graph, generate(graph))
  }

  /** Lowers an already queried or cached graph without invoking Clang again. */
  fun generate(graph: CppTypeGraph): GeneratedGrammar {
    val bodyProductions = linkedSetOf<Production>()
    bodyProductions += PrimitiveLiteralRules.rules(TargetLanguage.CPP, includeNullableTypes = false)
    graph.values.mapNotNullTo(bodyProductions, ::valueProduction)

    for (type in graph.allTypes) {
      type.declaredCallables.mapNotNullTo(bodyProductions, ::callableProduction)
      for (base in type.directBases) {
        if (base.access != CppAccess.PUBLIC) continue
        val supertype = cfgType(base.type) ?: continue
        val subtype = cfgType(type.type) ?: continue
        if (supertype != subtype) bodyProductions += Production.unit(supertype, subtype)
      }
    }

    val expressionResultTypes = bodyProductions
      .asSequence()
      .map(Production::lhs)
      .filter { type -> type != CppStatementType && type != CFG.DefaultStart }
      .distinct()
      .sortedBy(TypeExpr::render)
      .toList()
    val statementProductions = expressionResultTypes.map { resultType ->
      Production(
        CppStatementType,
        listOf(Symbol.Type(resultType), Symbol.Token(";")),
      )
    }
    val startProduction = Production.unit(CFG.DefaultStart, CppStatementType)
    // Context-bound lvalue/rvalue slots intentionally need not generate a
    // closed term: derivations may stop at them, just like Γ-slots in the
    // project's correctness statement.
    var grammar = CFG(bodyProductions + statementProductions + startProduction)
    if (options.normalizeChomskyNormalForm) grammar = grammar.toChomskyNormalForm()
    return grammar.toGeneratedGrammar()
  }

  private fun valueProduction(value: CppValueInfo): Production? {
    val type = cfgType(value.type) ?: return null
    return Production.literal(type, value.name)
  }

  private fun callableProduction(callable: CppCallableInfo): Production? {
    if (callable.access != CppAccess.PUBLIC || callable.isDeleted || callable.isVariadic) return null
    if (callable.parameterTypes.size > options.maxCallArity) return null
    if (callable.kind != CppCallableKind.INVOCATION && !callable.name.isSupportedCppIdentifier()) return null
    val result = cfgType(callable.resultType) ?: return null
    val parameters = callable.parameterTypes.map { parameter ->
      cfgParameterType(parameter) ?: return null
    }
    val owner = callable.owner?.let(::cfgType)

    return when (callable.kind) {
      CppCallableKind.CONSTRUCTOR -> {
        val constructed = owner ?: return null
        Production.call(
          result = constructed,
          name = constructed.render(),
          receiver = null,
          parameters = parameters,
          staticOwner = null,
        )
      }
      CppCallableKind.METHOD -> Production.call(
        result = result,
        name = callable.name,
        receiver = owner ?: return null,
        parameters = parameters,
        staticOwner = null,
      )
      CppCallableKind.STATIC_METHOD -> staticCallProduction(
        result,
        owner ?: return null,
        callable.name,
        parameters,
      )
      CppCallableKind.INVOCATION -> invocationProduction(
        result,
        owner ?: return null,
        parameters,
      )
      CppCallableKind.FREE_FUNCTION -> freeCallProduction(result, callable.name, parameters)
    }
  }

  private fun staticCallProduction(
    result: TypeExpr,
    owner: TypeExpr,
    name: String,
    parameters: List<TypeExpr>,
  ): Production = callLikeProduction(
    result = result,
    prefix = listOf(Symbol.Token(owner.render()), Symbol.Token("::"), Symbol.Token(name)),
    parameters = parameters,
  )

  private fun freeCallProduction(
    result: TypeExpr,
    name: String,
    parameters: List<TypeExpr>,
  ): Production = callLikeProduction(
    result = result,
    prefix = listOf(Symbol.Token(name)),
    parameters = parameters,
  )

  private fun invocationProduction(
    result: TypeExpr,
    receiver: TypeExpr,
    parameters: List<TypeExpr>,
  ): Production = callLikeProduction(
    result = result,
    prefix = listOf(Symbol.Type(receiver)),
    parameters = parameters,
  )

  private fun callLikeProduction(
    result: TypeExpr,
    prefix: List<Symbol>,
    parameters: List<TypeExpr>,
  ): Production = Production(
    result,
    buildList {
      addAll(prefix)
      add(Symbol.Token("("))
      parameters.forEachIndexed { index, parameter ->
        if (index > 0) add(Symbol.Token(","))
        add(Symbol.Type(parameter))
      }
      add(Symbol.Token(")"))
    },
  )

  private fun cfgParameterType(type: CppTypeRef): TypeExpr? {
    val slotType = when (type.reference) {
      // A const lvalue reference can bind both values and lvalues. Volatile
      // references remain exact because ordinary values cannot bind to them.
      CppReferenceKind.LVALUE -> if (type.isConst && !type.isVolatile) {
        type.withoutTopLevelCvAndReference()
      } else {
        type.copy(isConst = false, isVolatile = false)
      }
      // Rvalue and non-const lvalue references are value categories, not just
      // aliases for the referred-to static type. Preserve them as Γ slots.
      CppReferenceKind.RVALUE -> type.copy(isConst = false, isVolatile = false)
      null -> type.copy(isConst = false, isVolatile = false)
    }
    return cfgType(slotType)
  }

  private fun cfgType(type: CppTypeRef): TypeExpr? {
    if (!type.isSupportedInStatementFragment()) return null
    return TypeExpr.Applied(type.render())
  }

  private fun CppTypeRef.isSupportedInStatementFragment(): Boolean {
    val rendered = render()
    if (rendered.isEmpty() || rendered.any(Char::isWhitespace)) return false
    if (
      rendered.contains("<dependent") ||
      rendered.contains("type-parameter") ||
      rendered.contains("NULL_TYPE") ||
      rendered.contains("...") ||
      rendered.contains("decltype(") ||
      rendered.any { character -> character in "()[]" } ||
      name == "auto" ||
      name.startsWith("_") ||
      name.contains("::__")
    ) {
      return false
    }
    if (name.matches(Regex("[A-Z_]*_[A-Z0-9_]*")) || name.matches(Regex("_[A-Za-z0-9_]+"))) {
      return false
    }
    return arguments.all { argument -> argument.isSupportedInStatementFragment() }
  }

  private fun String.isSupportedCppIdentifier(): Boolean =
    isNotEmpty() &&
        (first() == '_' || first().isLetter()) &&
        drop(1).all { character -> character == '_' || character.isLetterOrDigit() } &&
        !startsWith("__")

  companion object {
    val CppStatementType: TypeExpr = TypeExpr.Applied("CPP_STATEMENT")
  }
}
