package org.lib2cfg

import java.io.File
import java.lang.reflect.Modifier
import java.net.URI
import java.nio.file.Files
import java.nio.file.FileSystems
import java.util.concurrent.ConcurrentHashMap
import java.util.jar.JarFile
import kotlin.reflect.KCallable
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.KParameter
import kotlin.reflect.KProperty1
import kotlin.reflect.KType
import kotlin.reflect.KTypeParameter
import kotlin.reflect.KTypeProjection
import kotlin.reflect.KVariance
import kotlin.reflect.KVisibility
import kotlin.reflect.full.declaredMemberFunctions
import kotlin.reflect.full.declaredMemberProperties
import kotlin.reflect.full.memberFunctions
import kotlin.reflect.full.memberProperties
import kotlin.reflect.full.staticFunctions
import kotlin.reflect.jvm.kotlinFunction
import kotlin.reflect.jvm.jvmErasure

private const val TRACK_NULLABILITY_ANNOTATIONS = false
private const val INCLUDE_NULLABLE_TYPES = false

fun main(args: Array<String>) {
  val commandLine = CommandLineArguments.parse(args)
  val options = GeneratorOptions(
    packageName = commandLine.packageName,
    normalizeChomskyNormalForm = commandLine.normalizeChomskyNormalForm,
  )
  val outputFile = defaultOutputFile(commandLine.packageName, commandLine.normalizeChomskyNormalForm)
  outputFile.parentFile?.mkdirs()
  val grammar = CfgGenerator(options).generate()
  Files.writeString(outputFile.toPath(), "${grammar.text}\n")
  println(
    "Wrote |P|=${grammar.productionCount}, |V|=${grammar.nonterminalCount}, |Σ|=${grammar.terminalCount} to ${outputFile.path}",
  )
}

private fun defaultOutputFile(packageName: String, normalizeChomskyNormalForm: Boolean): File {
  val extension = if (normalizeChomskyNormalForm) "cnf" else "cfg"
  return File("gen", "${packageName.replace('.', '_')}.$extension")
}

private data class CommandLineArguments(
  val packageName: String,
  val normalizeChomskyNormalForm: Boolean,
) {
  companion object {
    fun parse(args: Array<String>): CommandLineArguments {
      val positional = mutableListOf<String>()
      var normalizeChomskyNormalForm = false

      for (arg in args) {
        when (arg) {
          "--cnf" -> normalizeChomskyNormalForm = true
          else -> {
            require(!arg.startsWith("-")) { "Unknown flag: $arg" }
            positional += arg
          }
        }
      }
      require(positional.size == 1) { "Expected exactly one positional argument: <package>" }

      return CommandLineArguments(positional.single(), normalizeChomskyNormalForm)
    }
  }
}

data class GeneratorOptions(
  val packageName: String,
  val monomorphizationDepth: Int = 2,
  val maxSubtypeAlternativesPerType: Int = 80,
  val maxTypeArgumentsPerVariable: Int = 26,
  val maxTypeVariablesPerCallable: Int = 2,
  val maxVarargArity: Int = 3,
  val trackNullabilityAnnotations: Boolean = TRACK_NULLABILITY_ANNOTATIONS,
  val includeNullableTypes: Boolean = INCLUDE_NULLABLE_TYPES,
  val normalizeChomskyNormalForm: Boolean = false,
  val targetLanguage: TargetLanguage = TargetLanguage.fromPackageName(packageName),
)

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

enum class TargetLanguage {
  KOTLIN,
  JAVA;

  companion object {
    fun fromPackageName(packageName: String): TargetLanguage = if (packageName.startsWith("kotlin.")) KOTLIN else JAVA
  }
}

private fun <T, R> Iterable<T>.parallelFlatMap(transform: (T) -> Iterable<R>): List<R> =
  materializedCollection().parallelStream()
    .flatMap { item -> transform(item).materializedCollection().stream() }
    .toList()

@Suppress("UNCHECKED_CAST")
private fun <T> Iterable<T>.materializedCollection(): Collection<T> =
  if (this is Collection<*>) this as Collection<T> else toList()

class CfgGenerator(private val options: GeneratorOptions) {
  private val scanner = ClassPathPackageScanner()
  private lateinit var typeRenderer: TypeRenderer
  private val memberFunctionsByClass = ConcurrentHashMap<KClass<*>, List<KFunction<*>>>()
  private val memberPropertiesByClass = ConcurrentHashMap<KClass<*>, List<KProperty1<out Any, *>>>()
  private val staticFunctionsByClass = ConcurrentHashMap<KClass<*>, List<KFunction<*>>>()
  private val receiverCandidateClassesByClass = ConcurrentHashMap<KClass<*>, List<KClass<*>>>()

  fun generate(): GeneratedGrammar {
    val scannedClasses = scanner.scan(options.packageName)
    val topLevelFunctionHolders = scannedClasses
      .filter(::isPublicTopLevelFunctionHolder)
      .sortedBy { it.name }
    val topLevelFunctions = topLevelFunctionHolders.parallelFlatMap(::topLevelFunctions)
    val extensionFunctions = topLevelFunctions.filter { function ->
      function.parameters.any { it.kind == KParameter.Kind.EXTENSION_RECEIVER }
    }
    val ordinaryTopLevelFunctions = topLevelFunctions.filter { function ->
      function.parameters.none { it.kind == KParameter.Kind.EXTENSION_RECEIVER }
    }
    val scannedTypeClasses = scannedClasses
      .filter(::isPublicTypeClass)
      .mapNotNull { it.safeKotlinClass() }
    val signatureTypeClasses = topLevelFunctions.parallelFlatMap(::signatureTypeClasses)
    val typeClasses = closeRelevantTypeClasses(scannedTypeClasses + signatureTypeClasses)
    typeRenderer = TypeRenderer(
      targetLanguage = options.targetLanguage,
      trackNullabilityAnnotations = options.trackNullabilityAnnotations,
      qualifiedTypeNames = collidingTypeQualifiedNames(typeClasses, topLevelFunctions),
    )

    val subtypeIndex = SubtypeIndex(
      templates = typeClasses.parallelFlatMap(::directSubtypeTemplates),
      maxAlternativesPerType = options.maxSubtypeAlternativesPerType,
      topTypeName = options.topTypeName(),
    )
    val groundTypes = buildGroundTypes(typeClasses)
    val typeArgumentTypes = typeArgumentTypes(groundTypes)
    val productions = linkedSetOf<Production>()
    val typeSlots = linkedSetOf<TypeExpr>()

    PrimitiveLiteralRules.rules(options.targetLanguage, options.includeNullableTypes).forEach { productions += it }

    addProductions(
      typeClasses.parallelFlatMap { typeClass ->
        constructorProductions(typeClass, groundTypes, typeArgumentTypes, subtypeIndex) +
          memberProductions(typeClass, groundTypes, typeArgumentTypes, subtypeIndex)
      },
      productions,
      typeSlots,
    )

    addProductions(
      extensionFunctionProductions(extensionFunctions, groundTypes, typeArgumentTypes, subtypeIndex),
      productions,
      typeSlots,
    )
    addProductions(
      topLevelFunctionProductions(ordinaryTopLevelFunctions, groundTypes, typeArgumentTypes, subtypeIndex),
      productions,
      typeSlots,
    )

    addProductions(groundSubtypeProductions(subtypeIndex, groundTypes), productions, typeSlots)
    addProductions(erasedSubtypeProductions(subtypeIndex, groundTypes, typeSlots), productions, typeSlots)

    val erasedSlots = typeSlots.mapNotNullTo(mutableSetOf()) { type -> type.asErasedSlot() }
    addProductions(
      typeSlots.toList().parallelFlatMap { slot ->
        groundTypes.mapNotNull { alternative ->
          if (
            alternative != slot &&
            !(slot == options.topType() && alternative.hasErasedCover(erasedSlots)) &&
            subtypeIndex.isSubtypeOf(alternative, slot, options.monomorphizationDepth + 4)
          ) {
            Production(slot, listOf(Symbol.Type(alternative)))
          } else {
            null
          }
        }
      },
      productions,
      typeSlots,
    )
    topTypeProductions(typeSlots, subtypeIndex).forEach { production ->
      if (productions.add(production)) {
        typeSlots += production.types()
      }
    }

    if (options.includeNullableTypes) {
      nullableLiteralProductions(productions.flatMap { it.types() }).forEach { productions += it }
    }
    val bodyProductions = pruneUndefinedNonterminals(productions)
    val allProductions = bodyProductions + startProductions(bodyProductions)
    val finalProductions = if (options.normalizeChomskyNormalForm) {
      toChomskyNormalForm(allProductions, StartType)
    } else {
      allProductions
    }

    return GeneratedGrammar.from(finalProductions)
  }

  private fun constructorProductions(
    typeClass: KClass<*>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production> {
    val resultPattern = classType(typeClass)
    return typeClass.constructors
      .asSequence()
      .filter { it.isPublicCallable() }
      .filterNot { Modifier.isAbstract(typeClass.java.modifiers) }
      .filter { it.parameters.all { parameter -> parameter.kind == KParameter.Kind.VALUE } }
      .flatMap { constructor ->
        val parameterAlternatives = callParameterTypeAlternatives(constructor.parameters) ?: return@flatMap emptySequence()
        parameterAlternatives.asSequence().flatMap { parameters ->
          val parameterPatterns = parameters.map(typeRenderer::render)
          monomorphizeCall(
            resultPattern = resultPattern,
            receiverPattern = null,
            name = typeClass.tokenName(),
            parameterPatterns = parameterPatterns,
            groundTypes = groundTypes,
            typeArgumentTypes = typeArgumentTypes,
            typeParameterBounds = typeParameterBounds(typeClass.typeParameters + constructor.typeParameters),
            subtypeIndex = subtypeIndex,
          ).asSequence()
        }
      }
      .distinct()
      .toList()
  }

  private fun memberProductions(
    typeClass: KClass<*>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production> {
    val functions = cachedMemberFunctions(typeClass).mapNotNull { function ->
      memberFunctionProductions(typeClass, function, groundTypes, typeArgumentTypes, subtypeIndex)
    }.flatten()
    val properties = cachedMemberProperties(typeClass).mapNotNull { property ->
      memberPropertyProductions(typeClass, property, groundTypes, typeArgumentTypes, subtypeIndex)
    }.flatten()
    val staticFunctions = cachedStaticFunctions(typeClass).mapNotNull { function ->
      staticFunctionProductions(typeClass.tokenName(), function, groundTypes, typeArgumentTypes, subtypeIndex)
    }.flatten()
    return functions + properties + staticFunctions
  }

  private fun memberFunctionProductions(
    receiverClass: KClass<*>,
    function: KFunction<*>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production>? {
    if (!function.isUsablePublicCallable()) return null
    val valueParameters = function.parameters.filter { it.kind == KParameter.Kind.VALUE }
    val parameterAlternatives = callParameterTypeAlternatives(valueParameters) ?: return null
    if (!isSupportedType(function.returnType)) return null
    return parameterAlternatives.flatMap { parameterTypes ->
      val call = liftedMemberFunctionCall(receiverClass, function, parameterTypes)
      monomorphizeCall(
        resultPattern = call.resultPattern,
        receiverPattern = call.receiverPattern,
        name = function.name,
        parameterPatterns = call.parameterPatterns,
        groundTypes = groundTypes,
        typeArgumentTypes = typeArgumentTypes,
        typeParameterBounds = call.typeParameterBounds,
        subtypeIndex = subtypeIndex,
      )
    }
  }

  private fun memberPropertyProductions(
    receiverClass: KClass<*>,
    property: KProperty1<out Any, *>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production>? {
    if (!property.isUsablePublicCallable() || !isSupportedType(property.returnType)) return null
    val call = liftedMemberPropertyCall(receiverClass, property)
    return monomorphizeCall(
      resultPattern = call.resultPattern,
      receiverPattern = call.receiverPattern,
      name = property.name,
      parameterPatterns = emptyList(),
      groundTypes = groundTypes,
      typeArgumentTypes = typeArgumentTypes,
      typeParameterBounds = call.typeParameterBounds,
      subtypeIndex = subtypeIndex,
      propertyAccess = true,
    )
  }

  private fun staticFunctionProductions(
    ownerToken: String,
    function: KFunction<*>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production>? {
    if (!function.isUsablePublicCallable()) return null
    if (function.parameters.any { it.kind != KParameter.Kind.VALUE }) return null
    if (!isSupportedType(function.returnType)) return null
    val parameterAlternatives = callParameterTypeAlternatives(function.parameters) ?: return null
    return parameterAlternatives.flatMap { parameterTypes ->
      monomorphizeCall(
        resultPattern = typeRenderer.render(function.returnType),
        receiverPattern = null,
        name = function.name,
        parameterPatterns = parameterTypes.map(typeRenderer::render),
        groundTypes = groundTypes,
        typeArgumentTypes = typeArgumentTypes,
        typeParameterBounds = typeParameterBounds(function.typeParameters),
        subtypeIndex = subtypeIndex,
        staticOwnerToken = ownerToken,
      )
    }
  }

  private fun topLevelFunctions(holder: Class<*>): List<KFunction<*>> {
    return holder.declaredMethods
      .asSequence()
      .filter { method -> Modifier.isPublic(method.modifiers) && Modifier.isStatic(method.modifiers) }
      .filterNot { method -> method.name.isGeneratedName() }
      .mapNotNull { method -> method.kotlinFunction }
      .distinct()
      .toList()
  }

  private fun extensionFunctionProductions(
    functions: List<KFunction<*>>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production> = functions.parallelFlatMap { function ->
    extensionFunctionProductions(function, groundTypes, typeArgumentTypes, subtypeIndex).orEmpty()
  }.distinct()

  private fun extensionFunctionProductions(
    function: KFunction<*>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production>? {
    if (!function.isUsablePublicCallable()) return null
    val extensionReceiver = function.parameters.firstOrNull { it.kind == KParameter.Kind.EXTENSION_RECEIVER } ?: return null
    val valueParameters = function.parameters.filter { it.kind == KParameter.Kind.VALUE }
    val parameterAlternatives = callParameterTypeAlternatives(valueParameters) ?: return null
    if (!isSupportedType(extensionReceiver.type) || !isSupportedType(function.returnType)) {
      return null
    }
    return parameterAlternatives.flatMap { parameterTypes ->
      monomorphizeCall(
        resultPattern = typeRenderer.render(function.returnType),
        receiverPattern = typeRenderer.render(extensionReceiver.type),
        name = function.name,
        parameterPatterns = parameterTypes.map(typeRenderer::render),
        groundTypes = groundTypes,
        typeArgumentTypes = typeArgumentTypes,
        typeParameterBounds = typeParameterBounds(function.typeParameters),
        subtypeIndex = subtypeIndex,
      )
    }
  }

  private fun topLevelFunctionProductions(
    functions: List<KFunction<*>>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production> =
    functions.parallelFlatMap { function ->
      topLevelFunctionProduction(function, groundTypes, typeArgumentTypes, subtypeIndex).orEmpty()
    }.distinct()

  private fun topLevelFunctionProduction(
    function: KFunction<*>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
  ): List<Production>? {
    if (!function.isUsablePublicCallable()) return null
    if (function.parameters.any { it.kind == KParameter.Kind.EXTENSION_RECEIVER }) return null
    if (!isSupportedType(function.returnType)) return null
    val valueParameters = function.parameters.filter { it.kind == KParameter.Kind.VALUE }
    val parameterAlternatives = callParameterTypeAlternatives(valueParameters) ?: return null
    return parameterAlternatives.flatMap { parameterTypes ->
      monomorphizeCall(
        resultPattern = typeRenderer.render(function.returnType),
        receiverPattern = null,
        name = function.name,
        parameterPatterns = parameterTypes.map(typeRenderer::render),
        groundTypes = groundTypes,
        typeArgumentTypes = typeArgumentTypes,
        typeParameterBounds = typeParameterBounds(function.typeParameters),
        subtypeIndex = subtypeIndex,
      )
    }
  }

  private fun signatureTypeClasses(function: KFunction<*>): List<KClass<*>> =
    (function.parameters.map { it.type } + function.returnType)
    .flatMap { it.erasedClasses() }
    .filter { it.isAdmissibleTypeClass(options.packageName, options.targetLanguage) }

  private fun closeRelevantTypeClasses(seedClasses: List<KClass<*>>): List<KClass<*>> {
    val byName = linkedMapOf<String, KClass<*>>()
    val queue = ArrayDeque<KClass<*>>()

    fun add(typeClass: KClass<*>) {
      if (!typeClass.isAdmissibleTypeClass(options.packageName, options.targetLanguage)) return
      val key = typeClass.qualifiedName ?: typeClass.typeName(options.targetLanguage)
      if (byName.putIfAbsent(key, typeClass) == null) {
        queue += typeClass
      }
    }

    seedClasses.forEach(::add)
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      current.safeSupertypes()
        .flatMap { it.erasedClasses() }
        .forEach(::add)
    }

    return byName.values.sortedBy { it.qualifiedName ?: it.simpleName ?: "" }
  }

  private fun buildGroundTypes(typeClasses: List<KClass<*>>): Set<TypeExpr> {
    val constructors = typeClasses
      .map { typeClass -> GroundTypeConstructor(typeRenderer.renderClassName(typeClass), typeClass.typeParameters.size) }
      .filter { constructor -> constructor.arity <= 2 && constructor.name.isGroundTypeConstructorName() }
      .distinct()
      .sortedWith(compareBy<GroundTypeConstructor> { it.arity }.thenBy { it.name })

    val result = linkedSetOf<TypeExpr>()
    result += primitiveGroundTypes(options.targetLanguage, options.includeNullableTypes)
    constructors
      .filter { it.arity == 0 }
      .mapTo(result) { TypeExpr.Applied(it.name) }

    for (depth in 1..options.monomorphizationDepth) {
      val argumentPool = result
        .filter { it.depth() < depth && it.isTypeArgumentCandidate(options.targetLanguage) }
        .sortedWith(compareBy<TypeExpr> { it.depth() }.thenBy { it.render() })
        .take(options.maxTypeArgumentsPerVariable)

      for (constructor in constructors) {
        when (constructor.arity) {
          1 -> argumentPool
            .map { argument -> TypeExpr.Applied(constructor.name, listOf(argument)) }
            .filterTo(result) { it.depth() <= options.monomorphizationDepth }

          2 -> {
            for (left in argumentPool) {
              for (right in argumentPool) {
                val type = TypeExpr.Applied(constructor.name, listOf(left, right))
                if (type.depth() <= options.monomorphizationDepth) {
                  result += type
                }
              }
            }
          }
        }
      }
    }

    return result
  }

  private fun typeArgumentTypes(groundTypes: Set<TypeExpr>): List<TypeExpr> = groundTypes
    .filter { it.depth() < options.monomorphizationDepth && it.isTypeArgumentCandidate(options.targetLanguage) }
    .sortedWith(compareBy<TypeExpr> { it.typeArgumentPriority(options.targetLanguage) }.thenBy { it.depth() }.thenBy { it.render() })
    .take(options.maxTypeArgumentsPerVariable)

  private fun groundSubtypeProductions(subtypeIndex: SubtypeIndex, groundTypes: Set<TypeExpr>): List<Production> {
    val types = groundTypes.toList()
    val topTypes = setOf(options.topType(), options.nullableTopType())
    return types.parallelFlatMap { expected ->
      if (expected in topTypes) return@parallelFlatMap emptyList()
      types.mapNotNull { actual ->
        if (actual != expected && subtypeIndex.isSubtypeOf(actual, expected, options.monomorphizationDepth + 4)) {
          Production(expected, listOf(Symbol.Type(actual)))
        } else {
          null
        }
      }
    }
  }

  private fun erasedSubtypeProductions(
    subtypeIndex: SubtypeIndex,
    groundTypes: Set<TypeExpr>,
    typeSlots: Set<TypeExpr>,
  ): List<Production> {
    val parameterizedTypes = (groundTypes.asSequence() + typeSlots.asSequence())
      .mapNotNull { type -> (type as? TypeExpr.Applied)?.takeIf { it.arguments.isNotEmpty() } }
      .distinct()
      .toList()
    val neededErasedTypes = typeSlots
      .mapNotNullTo(linkedSetOf()) { type -> type.asErasedSlot() }

    parameterizedTypes
      .mapNotNull { type -> type.erasedApplied() }
      .forEach { erasedType -> neededErasedTypes += erasedType }

    val erasedEdges = subtypeIndex.templates
      .mapNotNull { template ->
        val supertype = template.supertype.erasedApplied() ?: return@mapNotNull null
        val subtype = template.subtype.erasedApplied() ?: return@mapNotNull null
        if (supertype == subtype) null else supertype to subtype
      }
      .distinct()
    val childrenBySupertype = erasedEdges.groupBy({ it.first }, { it.second })

    val queue = ArrayDeque<TypeExpr.Applied>()
    neededErasedTypes.forEach { erasedType -> queue += erasedType }
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      for (child in childrenBySupertype[current].orEmpty()) {
        if (neededErasedTypes.add(child)) {
          queue += child
        }
      }
    }

    val productions = linkedSetOf<Production>()
    for ((supertype, subtype) in erasedEdges) {
      if (supertype in neededErasedTypes && subtype in neededErasedTypes) {
        productions += Production(supertype, listOf(Symbol.Type(subtype)))
      }
    }
    for (type in parameterizedTypes) {
      val erasedType = type.erasedApplied() ?: continue
      if (erasedType in neededErasedTypes) {
        productions += Production(erasedType, listOf(Symbol.Type(type)))
      }
    }
    return productions.toList()
  }

  private fun topTypeProductions(typeSlots: Set<TypeExpr>, subtypeIndex: SubtypeIndex): List<Production> {
    val topType = options.topType()
    val nullableTopType = options.nullableTopType()
    val erasedSlots = typeSlots.mapNotNullTo(mutableSetOf()) { type -> type.asErasedSlot() }
    return typeSlots.parallelFlatMap { type ->
      listOfNotNull(
        if (
          type != topType &&
          !type.hasErasedCover(erasedSlots) &&
          subtypeIndex.isSubtypeOf(type, topType, options.monomorphizationDepth + 4)
        ) {
          Production(topType, listOf(Symbol.Type(type)))
        } else {
          null
        },
        if (
          options.includeNullableTypes &&
          type != nullableTopType &&
          subtypeIndex.isSubtypeOf(type, nullableTopType, options.monomorphizationDepth + 4)
        ) {
          Production(nullableTopType, listOf(Symbol.Type(type)))
        } else {
          null
        },
      )
    }
  }

  private fun addProductions(
    newProductions: Iterable<Production>,
    productions: MutableSet<Production>,
    typeSlots: MutableSet<TypeExpr>,
  ) {
    for (production in newProductions) if (productions.add(production)) typeSlots += production.types()
  }

  private fun cachedMemberFunctions(typeClass: KClass<*>): List<KFunction<*>> =
    memberFunctionsByClass.computeIfAbsent(typeClass) { safeMemberFunctions(it, options.packageName) }

  private fun cachedMemberProperties(typeClass: KClass<*>): List<KProperty1<out Any, *>> =
    memberPropertiesByClass.computeIfAbsent(typeClass) { safeMemberProperties(it, options.packageName) }

  private fun cachedStaticFunctions(typeClass: KClass<*>): List<KFunction<*>> =
    staticFunctionsByClass.computeIfAbsent(typeClass) { safeStaticFunctions(it, options.packageName) }

  private fun liftedMemberFunctionCall(
    receiverClass: KClass<*>,
    function: KFunction<*>,
    parameterTypes: List<KType>,
  ): LiftedMemberCall {
    val targetShape = normalizedCallableShape(
      function.name,
      typeRenderer.render(function.returnType),
      parameterTypes.map(typeRenderer::render),
    )

    for (candidateClass in receiverCandidateClasses(receiverClass)) {
      for (candidateFunction in cachedMemberFunctions(candidateClass)) {
        if (!candidateFunction.isUsablePublicCallable() || candidateFunction.name != function.name) continue
        if (!isSupportedType(candidateFunction.returnType)) continue
        val valueParameters = candidateFunction.parameters.filter { it.kind == KParameter.Kind.VALUE }
        val candidateParameterAlternatives = callParameterTypeAlternatives(valueParameters) ?: continue
        for (candidateParameterTypes in candidateParameterAlternatives) {
          val resultPattern = typeRenderer.render(candidateFunction.returnType)
          val parameterPatterns = candidateParameterTypes.map(typeRenderer::render)
          val candidateShape = normalizedCallableShape(candidateFunction.name, resultPattern, parameterPatterns)
          if (candidateShape != targetShape) continue
          return LiftedMemberCall(
            resultPattern = resultPattern,
            receiverPattern = liftedReceiverPattern(candidateClass, resultPattern, parameterPatterns),
            parameterPatterns = parameterPatterns,
            typeParameterBounds = typeParameterBounds(candidateFunction.typeParameters),
          )
        }
      }
    }

    val resultPattern = typeRenderer.render(function.returnType)
    val parameterPatterns = parameterTypes.map(typeRenderer::render)
    return LiftedMemberCall(
      resultPattern = resultPattern,
      receiverPattern = liftedReceiverPattern(receiverClass, resultPattern, parameterPatterns),
      parameterPatterns = parameterPatterns,
      typeParameterBounds = typeParameterBounds(function.typeParameters),
    )
  }

  private fun liftedMemberPropertyCall(
    receiverClass: KClass<*>,
    property: KProperty1<out Any, *>,
  ): LiftedMemberCall {
    val targetShape = normalizedCallableShape(property.name, typeRenderer.render(property.returnType), emptyList())

    for (candidateClass in receiverCandidateClasses(receiverClass)) {
      for (candidateProperty in cachedMemberProperties(candidateClass)) {
        if (!candidateProperty.isUsablePublicCallable() || candidateProperty.name != property.name) continue
        if (!isSupportedType(candidateProperty.returnType)) continue
        val resultPattern = typeRenderer.render(candidateProperty.returnType)
        val candidateShape = normalizedCallableShape(candidateProperty.name, resultPattern, emptyList())
        if (candidateShape != targetShape) continue
        return LiftedMemberCall(
          resultPattern = resultPattern,
          receiverPattern = liftedReceiverPattern(candidateClass, resultPattern, emptyList()),
          parameterPatterns = emptyList(),
          typeParameterBounds = emptyMap(),
        )
      }
    }

    val resultPattern = typeRenderer.render(property.returnType)
    return LiftedMemberCall(
      resultPattern = resultPattern,
      receiverPattern = liftedReceiverPattern(receiverClass, resultPattern, emptyList()),
      parameterPatterns = emptyList(),
      typeParameterBounds = emptyMap(),
    )
  }

  private fun receiverCandidateClasses(typeClass: KClass<*>): List<KClass<*>> =
    receiverCandidateClassesByClass.computeIfAbsent(typeClass) {
      val result = linkedMapOf<String, KClass<*>>()
      val visited = mutableSetOf<String>()

      fun key(candidate: KClass<*>): String = candidate.qualifiedName ?: candidate.java.name

      fun add(candidate: KClass<*>) {
        if (!candidate.isMemberReceiverCandidate()) return
        result.putIfAbsent(key(candidate), candidate)
      }

      fun visit(candidate: KClass<*>) {
        val candidateKey = key(candidate)
        if (!visited.add(candidateKey)) return
        candidate.safeSupertypes()
          .mapNotNull { supertype -> supertype.erasedClass() }
          .forEach(::visit)
        add(candidate)
      }

      add(Any::class)
      visit(typeClass)
      result.values.toList()
    }

  private fun KClass<*>.isMemberReceiverCandidate(): Boolean =
    isTopTypeClass() || isAdmissibleTypeClass(options.packageName, options.targetLanguage)

  private fun liftedReceiverPattern(
    receiverClass: KClass<*>,
    resultPattern: TypeExpr,
    parameterPatterns: List<TypeExpr>,
  ): TypeExpr {
    val parameterizedReceiver = classType(receiverClass)
    val receiverVariables = parameterizedReceiver.variables()
    if (receiverVariables.isEmpty()) return parameterizedReceiver

    val signatureVariables = (listOf(resultPattern) + parameterPatterns)
      .flatMapTo(mutableSetOf()) { pattern -> pattern.variables() }
    return if (receiverVariables.none { variable -> variable in signatureVariables }) {
      parameterizedReceiver.erased()
    } else {
      parameterizedReceiver
    }
  }

  private fun normalizedCallableShape(
    name: String,
    resultPattern: TypeExpr,
    parameterPatterns: List<TypeExpr>,
  ): MemberCallableShape {
    val normalizedTypes = normalizeVariables(listOf(resultPattern) + parameterPatterns)
    return MemberCallableShape(name, normalizedTypes.first(), normalizedTypes.drop(1))
  }

  private fun collidingTypeQualifiedNames(
    typeClasses: List<KClass<*>>,
    topLevelFunctions: List<KFunction<*>>,
  ): Set<String> {
    val terminalTokens = linkedSetOf<String>()

    topLevelFunctions
      .filter { function -> function.isUsablePublicCallable() }
      .mapTo(terminalTokens) { function -> function.name.noWhitespaceToken() }

    for (typeClass in typeClasses) {
      if (hasPublicConstructorTerminal(typeClass)) {
        terminalTokens += typeClass.tokenName()
      }
      if (cachedStaticFunctions(typeClass).any { function -> function.isUsablePublicCallable() }) {
        terminalTokens += typeClass.tokenName()
      }
      cachedMemberFunctions(typeClass)
        .filter { function -> function.isUsablePublicCallable() }
        .mapTo(terminalTokens) { function -> function.name.noWhitespaceToken() }
      cachedMemberProperties(typeClass)
        .filter { property -> property.isUsablePublicCallable() }
        .mapTo(terminalTokens) { property -> property.name.noWhitespaceToken() }
    }

    return typeClasses
      .mapNotNull { typeClass ->
        val qualified = typeClass.qualifiedName ?: return@mapNotNull null
        val typeName = typeClass.typeName(options.targetLanguage)
        if (typeName in terminalTokens && typeName !in AlwaysUnqualifiedTypeNames) qualified else null
      }
      .toSet()
  }

  private fun hasPublicConstructorTerminal(typeClass: KClass<*>): Boolean =
    !Modifier.isAbstract(typeClass.java.modifiers) &&
      typeClass.constructors.any { constructor ->
        constructor.isPublicCallable() &&
          constructor.parameters.all { parameter -> parameter.kind == KParameter.Kind.VALUE }
      }

  private fun directSubtypeTemplates(typeClass: KClass<*>): List<SubtypeTemplate> {
    val subtype = classType(typeClass)
    return typeClass.safeSupertypes()
      .mapNotNull { supertype ->
        if (!isSupportedType(supertype)) return@mapNotNull null
        val renderedSupertype = typeRenderer.render(supertype)
        if (renderedSupertype == options.topType() || renderedSupertype == subtype) null else SubtypeTemplate(renderedSupertype, subtype)
      }
  }

  private fun classType(typeClass: KClass<*>): TypeExpr =
    TypeExpr.Applied(typeRenderer.renderClassName(typeClass), typeClass.typeParameters.map { TypeExpr.Variable(it.name) })

  private fun isSupportedType(type: KType): Boolean =
    type.containsOnlySupportedArguments() && runCatching { typeRenderer.render(type) }.isSuccess

  private fun callParameterTypeAlternatives(parameters: List<KParameter>): List<List<KType>>? {
    val varargIndex = parameters.indexOfFirst { it.isVararg }
    if (varargIndex < 0) {
      return parameters
        .takeIf { values -> values.all { parameter -> isSupportedType(parameter.type) } }
        ?.let { values -> listOf(values.map { parameter -> parameter.type }) }
    }

    if (parameters.indexOfLast { it.isVararg } != varargIndex) return null
    if (varargIndex != parameters.lastIndex) return null

    val fixedParameters = parameters.take(varargIndex)
    if (fixedParameters.any { parameter -> !isSupportedType(parameter.type) }) return null

    val varargElementType = parameters[varargIndex].type.varargElementType() ?: return null
    if (!isSupportedType(varargElementType)) return null

    val fixedTypes = fixedParameters.map { parameter -> parameter.type }
    return (0..options.maxVarargArity).map { arity ->
      fixedTypes + List(arity) { varargElementType }
    }
  }

  private fun typeParameterBounds(typeParameters: List<KTypeParameter>): Map<String, List<TypeExpr>> =
    typeParameters.associate { typeParameter ->
      typeParameter.name to typeParameter.upperBounds
        .mapNotNull { bound -> runCatching { typeRenderer.render(bound) }.getOrNull() }
        .filter { bound -> bound != options.nullableTopType() }
    }

  private fun monomorphizeCall(
    resultPattern: TypeExpr,
    receiverPattern: TypeExpr?,
    name: String,
    parameterPatterns: List<TypeExpr>,
    groundTypes: Set<TypeExpr>,
    typeArgumentTypes: List<TypeExpr>,
    typeParameterBounds: Map<String, List<TypeExpr>>,
    subtypeIndex: SubtypeIndex,
    propertyAccess: Boolean = false,
    staticOwnerToken: String? = null,
  ): List<Production> {
    val patterns = listOfNotNull(resultPattern, receiverPattern) + parameterPatterns
    val variables = patterns.flatMap { it.variables() }.distinct().sorted()
    if (variables.size > options.maxTypeVariablesPerCallable) return emptyList()

    return substitutions(variables, typeArgumentTypes)
      .mapNotNull { substitution ->
        if (!substitutionSatisfiesBounds(substitution, typeParameterBounds, subtypeIndex)) {
          return@mapNotNull null
        }
        val resultType = resultPattern.substitute(substitution) ?: return@mapNotNull null
        val receiverType = receiverPattern?.substitute(substitution)
        val parameterTypes = parameterPatterns.map { pattern -> pattern.substitute(substitution) ?: return@mapNotNull null }

        if (!resultType.isGround() || resultType.depth() > options.monomorphizationDepth) return@mapNotNull null
        if (receiverType != null && (!receiverType.isGround() || receiverType.depth() > options.monomorphizationDepth)) {
          return@mapNotNull null
        }
        if (parameterTypes.any { !it.isGround() || it.depth() > options.monomorphizationDepth }) return@mapNotNull null

        val rhs = if (propertyAccess) {
          checkNotNull(receiverType)
          listOf(Symbol.Type(receiverType), Symbol.Token("."), Symbol.Token(name))
        } else {
          buildCallRhs(name, receiverType, parameterTypes, staticOwnerToken)
        }
        Production(resultType, rhs)
      }
      .distinct()
      .toList()
  }

  private fun substitutionSatisfiesBounds(
    substitution: Map<String, TypeExpr>,
    typeParameterBounds: Map<String, List<TypeExpr>>,
    subtypeIndex: SubtypeIndex,
  ): Boolean = substitution.all { (variable, actualType) ->
    typeParameterBounds[variable]
      .orEmpty()
      .all { boundPattern ->
        val bound = boundPattern.substitute(substitution) ?: return@all false
        bound == options.nullableTopType() || subtypeIndex.isSubtypeOf(actualType, bound, options.monomorphizationDepth + 4)
      }
  }

  private fun substitutions(variables: List<String>, typeArgumentTypes: List<TypeExpr>): Sequence<Map<String, TypeExpr>> {
    if (variables.isEmpty()) return sequenceOf(emptyMap())
    return variables.fold(listOf(emptyMap<String, TypeExpr>())) { partialSubstitutions, variable ->
      partialSubstitutions.flatMap { partial ->
        typeArgumentTypes.map { type -> partial + (variable to type) }
      }
    }
      .asSequence()
  }

  private fun buildCallRhs(
    name: String,
    receiver: TypeExpr?,
    parameters: List<TypeExpr>,
    staticOwnerToken: String? = null,
  ): List<Symbol> {
    val symbols = mutableListOf<Symbol>()
    if (staticOwnerToken != null) {
      symbols += Symbol.Token(staticOwnerToken)
      symbols += Symbol.Token(".")
    } else if (receiver != null) {
      symbols += Symbol.Type(receiver)
      symbols += Symbol.Token(".")
    }
    symbols += Symbol.Token(name)
    symbols += Symbol.Token("(")
    parameters.forEachIndexed { index, parameter ->
      if (index > 0) symbols += Symbol.Token(",")
      symbols += Symbol.Type(parameter)
    }
    symbols += Symbol.Token(")")
    return symbols
  }

  private fun nullableLiteralProductions(types: Iterable<TypeExpr>): List<Production> {
    val nullableTypes = types
      .filter { it.isNullableType() }
      .toMutableSet()
    nullableTypes += TypeExpr.Applied("Nothing", nullable = true)

    return nullableTypes
      .sortedBy { it.render() }
      .map { type -> Production(type, listOf(Symbol.Token("null"))) }
  }

  private fun startProductions(productions: Iterable<Production>): List<Production> =
    productions.map { it.lhs }.filter { it != StartType }.toSet().sortedBy { it.render() }
      .map { type -> Production(StartType, listOf(Symbol.Type(type))) }

  private fun pruneUndefinedNonterminals(productions: Iterable<Production>): Set<Production> {
    var remaining = productions.toSet()
    while (true) {
      val definedTypes = remaining.mapTo(mutableSetOf()) { it.lhs }
      val pruned = remaining
        .filter { production ->
          production.rhs.all { symbol -> symbol !is Symbol.Type || symbol.type in definedTypes }
        }
        .toSet()
      if (pruned.size == remaining.size) return pruned
      remaining = pruned
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
}

sealed interface Symbol {
  fun render(): String

  data class Type(val type: TypeExpr) : Symbol { override fun render(): String = type.render() }
  data class Token(val value: String) : Symbol { override fun render(): String = value.noWhitespaceToken() }
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

  data class Variable(val name: String) : TypeExpr { override fun render(): String = name.noWhitespaceToken() }
}

data class GroundTypeConstructor(val name: String, val arity: Int)

private data class LiftedMemberCall(
  val resultPattern: TypeExpr,
  val receiverPattern: TypeExpr,
  val parameterPatterns: List<TypeExpr>,
  val typeParameterBounds: Map<String, List<TypeExpr>>,
)

private data class MemberCallableShape(
  val name: String,
  val resultPattern: TypeExpr,
  val parameterPatterns: List<TypeExpr>,
)

private fun primitiveGroundTypes(targetLanguage: TargetLanguage, includeNullableTypes: Boolean): Set<TypeExpr> {
  val primitiveNames = when (targetLanguage) {
    TargetLanguage.KOTLIN -> KotlinLiteralRules.map { it.typeName } + listOf("Unit")
    TargetLanguage.JAVA -> JavaLiteralRules.map { it.typeName }
  }
  return buildSet {
    primitiveNames.forEach { name ->
      add(TypeExpr.Applied(name))
      if (includeNullableTypes) {
        add(TypeExpr.Applied(name, nullable = true))
      }
    }
    add(targetLanguage.topType())
    if (targetLanguage == TargetLanguage.KOTLIN) {
      add(TypeExpr.Applied("Nothing"))
    }
    if (includeNullableTypes) {
      add(targetLanguage.nullableTopType())
      if (targetLanguage == TargetLanguage.KOTLIN) {
        add(TypeExpr.Applied("Nothing", nullable = true))
      }
    }
  }
}

private fun TypeExpr.variables(): Set<String> = when (this) {
  is TypeExpr.Variable -> setOf(name)
  is TypeExpr.Applied -> arguments.flatMapTo(mutableSetOf()) { it.variables() }
}

private fun TypeExpr.erased(): TypeExpr = when (this) {
  is TypeExpr.Variable -> this
  is TypeExpr.Applied -> copy(arguments = emptyList())
}

private fun TypeExpr.erasedApplied(): TypeExpr.Applied? = when (this) {
  is TypeExpr.Variable -> null
  is TypeExpr.Applied -> copy(arguments = emptyList())
}

private fun TypeExpr.asErasedSlot(): TypeExpr.Applied? =
  (this as? TypeExpr.Applied)?.takeIf { it.arguments.isEmpty() }

private fun TypeExpr.hasErasedCover(erasedSlots: Set<TypeExpr.Applied>): Boolean =
  this is TypeExpr.Applied && arguments.isNotEmpty() && copy(arguments = emptyList()) in erasedSlots

private fun normalizeVariables(types: List<TypeExpr>): List<TypeExpr> {
  val names = linkedMapOf<String, String>()

  fun normalize(type: TypeExpr): TypeExpr = when (type) {
    is TypeExpr.Variable -> {
      val normalizedName = names.getOrPut(type.name) { "T${names.size}" }
      TypeExpr.Variable(normalizedName)
    }

    is TypeExpr.Applied -> type.copy(arguments = type.arguments.map(::normalize))
  }

  return types.map(::normalize)
}

private fun TypeExpr.substitute(substitution: Map<String, TypeExpr>): TypeExpr? = when (this) {
  is TypeExpr.Variable -> substitution[name]
  is TypeExpr.Applied -> copy(arguments = arguments.map { argument -> argument.substitute(substitution) ?: return null })
}

private fun TypeExpr.isGround(): Boolean = variables().isEmpty()

private fun TypeExpr.depth(): Int = when (this) {
  is TypeExpr.Variable -> 0
  is TypeExpr.Applied -> if (arguments.isEmpty()) 0 else 1 + arguments.maxOf { it.depth() }
}

private fun TypeExpr.isTypeArgumentCandidate(targetLanguage: TargetLanguage): Boolean {
  val rendered = render()
  return !(targetLanguage == TargetLanguage.JAVA && rendered in JavaNonGenericArgumentTypes) &&
    !rendered.startsWith("Function") &&
    !rendered.startsWith("Comparator") &&
    !rendered.startsWith("stream.") &&
    !rendered.contains("Spliterator") &&
    !rendered.endsWith("Array") &&
    !rendered.endsWith("Iterator")
}

private fun TypeExpr.typeArgumentPriority(targetLanguage: TargetLanguage): Int {
  val priority = when (targetLanguage) {
    TargetLanguage.KOTLIN -> KotlinTypeArgumentPriority
    TargetLanguage.JAVA -> JavaTypeArgumentPriority
  }
  return priority[render()] ?: 100
}

private fun String.isGroundTypeConstructorName(): Boolean =
  !startsWith("Function") &&
  !startsWith("Comparator") &&
  !startsWith("stream.") &&
  !contains("Spliterator") &&
  !contains("[]")

data class LiteralRule(val typeName: String, val literalToken: String)

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

private val JavaNonGenericArgumentTypes = setOf("boolean", "byte", "char", "double", "float", "int", "long", "short", "void")

private val KotlinTypeArgumentPriority = linkedMapOf(
  "Int" to 0,
  "String" to 1,
  "Boolean" to 2,
  "Char" to 3,
  "Long" to 4,
  "Double" to 5,
  "Float" to 6,
  "Short" to 7,
  "Byte" to 8,
  "UInt" to 9,
  "ULong" to 10,
  "UShort" to 11,
  "UByte" to 12,
  KotlinTopTypeName to 13,
  "$KotlinTopTypeName?" to 14,
  "Nothing" to 15,
  "Nothing?" to 16,
  "Int?" to 17,
  "String?" to 18,
  "Boolean?" to 19,
  "Char?" to 20,
  "Long?" to 21,
  "Double?" to 22,
  "Float?" to 23,
  "Short?" to 24,
  "Byte?" to 25,
  "List<Int>" to 26,
  "List<String>" to 27,
  "Set<Int>" to 28,
  "Iterable<Int>" to 29,
  "Array<Int>" to 30,
  "Pair<Int,Int>" to 31,
)

private val JavaTypeArgumentPriority = linkedMapOf(
  "Integer" to 0,
  "String" to 1,
  "Boolean" to 2,
  "Character" to 3,
  "Long" to 4,
  "Double" to 5,
  "Float" to 6,
  "Short" to 7,
  "Byte" to 8,
  JavaTopTypeName to 9,
  "List<Integer>" to 10,
  "List<String>" to 11,
  "Set<Integer>" to 12,
  "Collection<Integer>" to 13,
)

fun toChomskyNormalForm(productions: Iterable<Production>, start: TypeExpr): Set<Production> =
  ChomskyNormalFormConverter(start).convert(productions)

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
      normalized += Production(nonterminal, listOf(Symbol.Token(terminal)))
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
      else -> binarize(production.lhs, rhs.map { it as Symbol.Type })
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
      .filter { it.rhs.size == 1 && it.rhs.single() is Symbol.Type }
      .groupBy({ it.lhs }, { (it.rhs.single() as Symbol.Type).type })
    val nonUnitProductions = productions
      .filterNot { it.rhs.size == 1 && it.rhs.single() is Symbol.Type }
    val nonUnitsByLhs = nonUnitProductions.groupBy { it.lhs }
    val nonterminals = productions.flatMapTo(mutableSetOf()) { production ->
      production.types()
    }

    val result = linkedSetOf<Production>()
    for (source in nonterminals.sortedBy { it.render() }) {
      val closure = unitClosure(source, unitTargets)
      for (target in closure) {
        for (production in nonUnitsByLhs[target].orEmpty()) {
          result += production.copy(lhs = source)
        }
      }
    }
    return result
  }

  private fun unitClosure(source: TypeExpr, unitTargets: Map<TypeExpr, List<TypeExpr>>): Set<TypeExpr> {
    val closure = linkedSetOf(source)
    val queue = ArrayDeque<TypeExpr>()
    queue += source
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      for (target in unitTargets[current].orEmpty()) {
        if (closure.add(target)) {
          queue += target
        }
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
    val byLhs = productions.groupBy { it.lhs }
    val reachable = linkedSetOf(start)
    val queue = ArrayDeque<TypeExpr>()
    queue += start
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      for (production in byLhs[current].orEmpty()) {
        for (symbol in production.rhs) {
          if (symbol is Symbol.Type && reachable.add(symbol.type)) {
            queue += symbol.type
          }
        }
      }
    }
    return reachable
  }

  private fun isChomskyNormalForm(production: Production): Boolean =
    when (production.rhs.size) {
      1 -> production.rhs.single() is Symbol.Token
      2 -> production.rhs.all { it is Symbol.Type }
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
      TypeExpr.Applied("__CNF_T_${terminal.sanitizedCnfNamePart()}_${terminalCounter.toString().padStart(4, '0')}")
    }

  private fun suffixNonterminal(suffix: List<Symbol.Type>): TypeExpr =
    suffixNonterminals.getOrPut(suffix) {
      suffixCounter += 1
      TypeExpr.Applied("__CNF_N_${suffixCounter.toString().padStart(6, '0')}")
    }
}

class TypeRenderer(
  private val targetLanguage: TargetLanguage,
  private val trackNullabilityAnnotations: Boolean,
  private val qualifiedTypeNames: Set<String> = emptySet(),
) {
  fun renderClassName(typeClass: KClass<*>): String =
    typeClass.typeName(targetLanguage, forceQualified = typeClass.qualifiedName in qualifiedTypeNames)

  fun render(type: KType): TypeExpr = when (val classifier = type.classifier) {
    is KClass<*> -> {
      val arguments = type.arguments.map(::renderProjection)
      TypeExpr.Applied(renderClassName(classifier), arguments, trackNullabilityAnnotations && type.isMarkedNullable)
    }

    is KTypeParameter -> {
      val suffix = if (trackNullabilityAnnotations && type.isMarkedNullable) "?" else ""
      TypeExpr.Variable("${classifier.name}$suffix".noWhitespaceToken())
    }

    else -> error("Unsupported type classifier: $classifier")
  }

  private fun renderProjection(projection: KTypeProjection): TypeExpr =
    render(projection.type ?: error("Star projections are not supported"))
}

data class SubtypeTemplate(val supertype: TypeExpr, val subtype: TypeExpr)

class SubtypeIndex(
  val templates: List<SubtypeTemplate>,
  private val maxAlternativesPerType: Int,
  private val topTypeName: String,
) {
  private val templatesBySuperName = templates
    .filter { it.supertype is TypeExpr.Applied && it.subtype is TypeExpr.Applied }
    .groupBy { (it.supertype as TypeExpr.Applied).name }
  private val templatesBySubName = templates
    .filter { it.supertype is TypeExpr.Applied && it.subtype is TypeExpr.Applied }
    .groupBy { (it.subtype as TypeExpr.Applied).name }

  fun fittingAlternatives(type: TypeExpr, depth: Int): Set<TypeExpr> {
    if (depth <= 0) return setOf(type)
    return when (type) {
      is TypeExpr.Variable -> setOf(type)
      is TypeExpr.Applied -> fittingAppliedAlternatives(type, depth)
    }
      .filter { candidate -> isSubtypeOf(candidate, type, depth + 2) }
      .take(maxAlternativesPerType)
      .toSet()
  }

  fun isSubtypeOf(actual: TypeExpr, expected: TypeExpr, depth: Int): Boolean {
    return isSubtypeOf(actual, expected, depth, mutableSetOf())
  }

  private fun fittingAppliedAlternatives(type: TypeExpr.Applied, depth: Int): Set<TypeExpr> {
    val alternatives = linkedSetOf<TypeExpr>()
    alternatives += type

    if (type.arguments.isNotEmpty()) {
      for (argumentCombination in combinations(type.arguments.map { fittingAlternatives(it, depth - 1).take(10).toList() })) {
        alternatives += type.copy(arguments = argumentCombination)
      }
    }

    for (template in templatesBySuperName[type.name].orEmpty()) {
      val bindings = mutableMapOf<String, TypeExpr>()
      if (!unify(template.supertype, type, bindings)) continue
      alternatives += substitute(template.subtype, bindings, depth - 1)
    }

    if (depth > 1) {
      val directAlternatives = alternatives.toList()
      for (alternative in directAlternatives) {
        if (alternative != type) {
          alternatives += fittingAlternatives(alternative, depth - 1)
        }
      }
    }

    return alternatives.take(maxAlternativesPerType).toSet()
  }

  private fun isSubtypeOf(
    actual: TypeExpr,
    expected: TypeExpr,
    depth: Int,
    seen: MutableSet<Pair<TypeExpr, TypeExpr>>,
  ): Boolean {
    if (actual == expected) return true
    if (depth <= 0 || !seen.add(actual to expected)) return false

    if (expected is TypeExpr.Applied && expected.name == topTypeName && expected.nullable) {
      return true
    }
    if (expected is TypeExpr.Applied && expected.name == topTypeName && !expected.nullable) {
      return actual is TypeExpr.Applied && !actual.nullable
    }

    return when {
      actual is TypeExpr.Variable || expected is TypeExpr.Variable -> {
        typeVariableFits(actual, expected)
      }

      actual is TypeExpr.Applied && expected is TypeExpr.Applied -> {
        nullableFits(actual, expected) &&
          (
            sameConstructorSubtype(actual, expected) ||
              directSupertypes(actual).any { supertype ->
                isSubtypeOf(supertype, expected, depth - 1, seen)
              }
            )
      }

      else -> false
    }
  }

  private fun typeVariableFits(actual: TypeExpr, expected: TypeExpr): Boolean =
    actual == expected ||
    actual is TypeExpr.Variable &&
    expected is TypeExpr.Variable &&
    actual.name.removeSuffix("?") == expected.name.removeSuffix("?") &&
    (!actual.name.endsWith("?") || expected.name.endsWith("?"))

  private fun nullableFits(actual: TypeExpr.Applied, expected: TypeExpr.Applied): Boolean = !actual.nullable || expected.nullable

  private fun sameConstructorSubtype(
    actual: TypeExpr.Applied,
    expected: TypeExpr.Applied,
  ): Boolean = actual.name == expected.name &&
    actual.arguments.size == expected.arguments.size &&
    actual.arguments == expected.arguments

  private fun directSupertypes(type: TypeExpr.Applied): Set<TypeExpr> =
    templatesBySubName[type.name].orEmpty()
    .mapNotNull { template ->
      val bindings = mutableMapOf<String, TypeExpr>()
      if (unify(template.subtype, type, bindings)) substituteExact(template.supertype, bindings) else null
    }.toSet()

  private fun substitute(template: TypeExpr, bindings: Map<String, TypeExpr>, depth: Int): Set<TypeExpr> = when (template) {
    is TypeExpr.Variable -> {
      val bound = bindings[template.name] ?: template
      fittingAlternatives(bound, depth)
    }

    is TypeExpr.Applied -> {
      val argumentAlternatives = template.arguments.map { substitute(it, bindings, depth).take(10).toList() }
      combinations(argumentAlternatives)
        .mapTo(linkedSetOf()) { arguments -> template.copy(arguments = arguments) }
    }
  }

  private fun substituteExact(template: TypeExpr, bindings: Map<String, TypeExpr>): TypeExpr = when (template) {
    is TypeExpr.Variable -> bindings[template.name] ?: template
    is TypeExpr.Applied -> template.copy(arguments = template.arguments.map { substituteExact(it, bindings) })
  }

  private fun unify(pattern: TypeExpr, actual: TypeExpr, bindings: MutableMap<String, TypeExpr>): Boolean = when {
    pattern is TypeExpr.Variable -> {
      val previous = bindings[pattern.name]
      if (previous == null) {
        bindings[pattern.name] = actual
        true
      } else {
        previous == actual
      }
    }

    pattern is TypeExpr.Applied && actual is TypeExpr.Applied -> {
      pattern.name == actual.name &&
        pattern.nullable == actual.nullable &&
        pattern.arguments.size == actual.arguments.size &&
        pattern.arguments.zip(actual.arguments).all { (left, right) -> unify(left, right, bindings) }
    }

    else -> false
  }

  private fun <T> combinations(lists: List<List<T>>): List<List<T>> {
    if (lists.isEmpty()) return listOf(emptyList())
    return lists.fold(listOf(emptyList())) { accumulator, values ->
      accumulator.flatMap { prefix -> values.map { value -> prefix + value } }
    }
  }
}

object PrimitiveLiteralRules {
  fun rules(targetLanguage: TargetLanguage, includeNullableTypes: Boolean): List<Production> {
    val primitiveRules = when (targetLanguage) {
      TargetLanguage.KOTLIN -> KotlinLiteralRules
      TargetLanguage.JAVA -> JavaLiteralRules
    }
    val nullRules = if (includeNullableTypes && targetLanguage == TargetLanguage.KOTLIN) {
      listOf(Production(TypeExpr.Applied("Nothing", nullable = true), listOf(Symbol.Token("null"))))
    } else {
      emptyList()
    }

    return primitiveRules.flatMap { (typeName, literalToken) ->
      buildList {
        add(Production(TypeExpr.Applied(typeName), listOf(Symbol.Token(literalToken))))
        if (includeNullableTypes) {
          add(Production(TypeExpr.Applied(typeName, nullable = true), listOf(Symbol.Token("null"))))
        }
      }
    } + nullRules
  }
}

class ClassPathPackageScanner {
  fun scan(packageName: String): List<Class<*>> {
    val packagePath = packageName.replace('.', '/')
    val classNames = linkedSetOf<String>()
    scanRuntimeImage(packagePath, classNames)
    System.getProperty("java.class.path")
      .split(File.pathSeparator)
      .filter { it.isNotBlank() }
      .map(::File)
      .forEach { classPathEntry ->
        when {
          classPathEntry.isDirectory -> scanDirectory(classPathEntry, packagePath, classNames)
          classPathEntry.isFile && classPathEntry.extension == "jar" -> scanJar(classPathEntry, packagePath, classNames)
        }
      }

    val classLoader = Thread.currentThread().contextClassLoader
    return classNames.mapNotNull { className ->
      runCatching { Class.forName(className, false, classLoader) }.getOrNull()
    }
  }

  private fun scanRuntimeImage(packagePath: String, classNames: MutableSet<String>) {
    val fileSystem = runCatching { FileSystems.getFileSystem(URI.create("jrt:/")) }.getOrNull() ?: return
    val modulesRoot = fileSystem.getPath("/modules")
    if (!Files.isDirectory(modulesRoot)) return

    Files.list(modulesRoot).use { modules ->
      modules.forEach { moduleRoot ->
        val packageRoot = moduleRoot.resolve(packagePath)
        if (!Files.isDirectory(packageRoot)) return@forEach
        Files.list(packageRoot).use { files ->
          files
            .filter { file -> Files.isRegularFile(file) && file.fileName.toString().endsWith(".class") }
            .filter { file -> !file.fileName.toString().removeSuffix(".class").contains('$') }
            .forEach { file ->
              val simpleName = file.fileName.toString().removeSuffix(".class")
              classNames += "${packagePath.replace('/', '.')}.$simpleName"
            }
        }
      }
    }
  }

  private fun scanDirectory(root: File, packagePath: String, classNames: MutableSet<String>) {
    val packageRoot = root.resolve(packagePath)
    if (!packageRoot.exists()) return
    packageRoot.listFiles()
      .orEmpty()
      .asSequence()
      .filter { it.isFile && it.extension == "class" }
      .forEach { file ->
        val relativePath = root.toPath().relativize(file.toPath()).toString()
        classNames += relativePath.removeSuffix(".class").replace(File.separatorChar, '.')
      }
  }

  private fun scanJar(jarFile: File, packagePath: String, classNames: MutableSet<String>) {
    runCatching {
      JarFile(jarFile).use { jar ->
        jar.entries().asSequence()
          .filter { entry ->
            !entry.isDirectory &&
              entry.name.startsWith("$packagePath/") &&
              entry.name.endsWith(".class") &&
              entry.name.removePrefix("$packagePath/").contains('/').not()
          }
          .forEach { entry -> classNames += entry.name.removeSuffix(".class").replace('/', '.') }
      }
    }
  }
}

class KotlinCompilerProbe(private val compilerCommand: String = "kotlinc") {
  fun isAssignable(actualType: String, expectedType: String): Boolean? {
    val tempDir = Files.createTempDirectory("lib2cfg-probe-")
    return try {
      val source = tempDir.resolve("Probe.kt")
      Files.writeString(
        source,
        """
        |@Suppress("UNUSED_VARIABLE", "UNUSED_PARAMETER")
        |fun <T> probe(value: $actualType) {
        |  val expected: $expectedType = value
        |}
        |""".trimMargin(),
      )
      val process = ProcessBuilder(compilerCommand, source.toString(), "-d", tempDir.resolve("out").toString())
        .redirectErrorStream(true)
        .start()
      process.inputStream.bufferedReader().readText()
      process.waitFor() == 0
    } catch (_: Exception) {
      null
    } finally {
      tempDir.toFile().deleteRecursively()
    }
  }
}

private const val KotlinTopTypeName = "kotlin.Any"
private const val JavaTopTypeName = "java.lang.Object"
private val JavaBackedKotlinTypeAliases = mapOf(
  KotlinTopTypeName to JavaTopTypeName,
  "kotlin.Boolean" to "boolean",
  "kotlin.Byte" to "byte",
  "kotlin.Char" to "char",
  "kotlin.Double" to "double",
  "kotlin.Float" to "float",
  "kotlin.Int" to "int",
  "kotlin.Long" to "long",
  "kotlin.Short" to "short",
  "kotlin.String" to "String",
  "kotlin.Throwable" to "java.lang.Throwable",
  "kotlin.Unit" to "void",
  "kotlin.Nothing" to "java.lang.Void",
)
private val StartType = TypeExpr.Applied("START")
private val AlwaysUnqualifiedTypeNames = setOf(
  "Any",
  "Nothing",
  "Boolean",
  "Byte",
  "Char",
  "Double",
  "Float",
  "Int",
  "Long",
  "Short",
  "String",
  "UByte",
  "UInt",
  "ULong",
  "UShort",
  "Unit",
)
private val UnsupportedReflectionCallableNames = setOf("clone", "finalize")

private fun TargetLanguage.topTypeName(): String = when (this) {
  TargetLanguage.KOTLIN -> KotlinTopTypeName
  TargetLanguage.JAVA -> JavaTopTypeName
}

private fun TargetLanguage.topType(): TypeExpr = TypeExpr.Applied(topTypeName())

private fun TargetLanguage.nullableTopType(): TypeExpr = TypeExpr.Applied(topTypeName(), nullable = true)

private fun GeneratorOptions.topTypeName(): String = targetLanguage.topTypeName()

private fun GeneratorOptions.topType(): TypeExpr = targetLanguage.topType()

private fun GeneratorOptions.nullableTopType(): TypeExpr = targetLanguage.nullableTopType()

private fun KCallable<*>.isPublicCallable(): Boolean =
  runCatching { visibility == KVisibility.PUBLIC }.getOrDefault(false)

private fun KCallable<*>.isSupportedCallableName(): Boolean =
  name !in UnsupportedReflectionCallableNames && !name.isGeneratedName()

private fun KCallable<*>.isUsablePublicCallable(): Boolean =
  isSupportedCallableName() && isPublicCallable()

private fun KClass<*>.hasPublicVisibility(): Boolean =
  runCatching { visibility == KVisibility.PUBLIC }.getOrDefault(false)

private fun KType.containsOnlySupportedArguments(): Boolean = arguments.all { projection ->
  val projectedType = projection.type ?: return@all false
  (projection.variance == null || projection.variance == KVariance.INVARIANT || projection.variance == KVariance.OUT) &&
    projectedType.containsOnlySupportedArguments()
}

private fun KClass<*>.safeSupertypes(): List<KType> = runCatching { supertypes }.getOrElse { emptyList() }

private fun KType.erasedClass(): KClass<*>? = runCatching { jvmErasure }.getOrNull()

private fun KType.erasedClasses(): List<KClass<*>> = buildList {
  runCatching { jvmErasure }.getOrNull()?.let(::add)
  arguments.mapNotNull { it.type }.forEach { nestedType ->
    addAll(nestedType.erasedClasses())
  }
}

private fun KType.varargElementType(): KType? = arguments.singleOrNull()?.type

private fun TypeExpr.isNullableType(): Boolean = when (this) {
  is TypeExpr.Applied -> nullable
  is TypeExpr.Variable -> name.endsWith("?")
}

private fun safeMemberFunctions(typeClass: KClass<*>, packageName: String): List<KFunction<*>> {
  if (!typeClass.shouldReflectMembers(packageName)) return emptyList()
  val functions = try {
    typeClass.memberFunctions.toList()
  } catch (_: Throwable) {
    try {
      typeClass.declaredMemberFunctions.toList()
    } catch (_: Throwable) {
      emptyList()
    }
  }
  return functions.filter { it.isSupportedCallableName() }
}

private fun safeMemberProperties(typeClass: KClass<*>, packageName: String): List<KProperty1<out Any, *>> {
  if (!typeClass.shouldReflectMembers(packageName)) return emptyList()
  val properties = try {
    typeClass.memberProperties.toList()
  } catch (_: Throwable) {
    try {
      typeClass.declaredMemberProperties.toList()
    } catch (_: Throwable) {
      emptyList()
    }
  }
  return properties.filter { it.isSupportedCallableName() }
}

private fun KClass<*>.shouldReflectMembers(packageName: String): Boolean =
  isTopTypeClass() ||
    java.hasKotlinMetadata() ||
    (qualifiedName?.startsWith("$packageName.") == true && !java.hasFragileJavaCloneMember())

private fun KClass<*>.isTopTypeClass(): Boolean =
  qualifiedName == "kotlin.Any" || qualifiedName == "java.lang.Object"

private fun Class<*>.hasKotlinMetadata(): Boolean =
  getAnnotation(Metadata::class.java) != null

private fun Class<*>.hasFragileJavaCloneMember(): Boolean =
  !hasKotlinMetadata() &&
    java.lang.Cloneable::class.java.isAssignableFrom(this) &&
    methods.any { method -> method.name == "clone" }

private fun safeStaticFunctions(typeClass: KClass<*>, packageName: String): List<KFunction<*>> =
  if (!typeClass.shouldReflectMembers(packageName)) {
    emptyList()
  } else {
    safeStaticFunctions(typeClass)
  }

private fun safeStaticFunctions(typeClass: KClass<*>): List<KFunction<*>> =
  try {
    typeClass.staticFunctions.toList().filter { it.isSupportedCallableName() }
  } catch (_: Throwable) {
    emptyList()
  }

private fun Class<*>.safeKotlinClass(): KClass<*>? = runCatching { kotlin }.getOrNull()

private fun isPublicTypeClass(javaClass: Class<*>): Boolean {
  val kotlinClass = javaClass.safeKotlinClass() ?: return false
  return Modifier.isPublic(javaClass.modifiers) &&
    !javaClass.name.contains('$') &&
    kotlinClass.isRelevantPublicTypeClass(kotlinClass.qualifiedName?.substringBeforeLast('.') ?: "")
}

private fun KClass<*>.isRelevantPublicTypeClass(packageName: String): Boolean {
  val qualified = qualifiedName ?: return false
  val simple = simpleName ?: return false
  return qualified.startsWith("$packageName.") &&
    hasPublicVisibility() &&
    !java.isSynthetic &&
    !java.isAnnotation &&
    !java.name.contains('$') &&
    simple != "Companion" &&
    !simple.isGeneratedName() &&
    !simple.contains("Kt") &&
    !simple.endsWith("DefaultImpls")
}

private fun KClass<*>.isAdmissibleTypeClass(packageName: String, targetLanguage: TargetLanguage): Boolean =
  isRelevantPublicTypeClass(packageName) || isConcretePublicSignatureClass(targetLanguage)

private fun KClass<*>.isConcretePublicSignatureClass(targetLanguage: TargetLanguage): Boolean {
  val qualified = qualifiedName ?: return false
  val simple = simpleName ?: return false
  if (
    targetLanguage == TargetLanguage.JAVA &&
    qualified.startsWith("kotlin.") &&
    qualified !in JavaBackedKotlinTypeAliases
  ) {
    return false
  }
  return hasPublicVisibility() &&
    !java.isSynthetic &&
    !java.isAnnotation &&
    !java.isArray &&
    !java.isPrimitive &&
    !java.isInterface &&
    !Modifier.isAbstract(java.modifiers) &&
    !java.name.contains('$') &&
    simple != "Companion" &&
    !simple.isGeneratedName() &&
    !simple.contains("Kt") &&
    !qualified.startsWith("kotlin.jvm.functions.") &&
    qualified != "kotlin.String"
}

private fun isPublicTopLevelFunctionHolder(javaClass: Class<*>): Boolean {
  val simpleName = javaClass.simpleName
  return !javaClass.isSynthetic &&
    !javaClass.name.contains('$') &&
    simpleName.contains("Kt") &&
    !simpleName.isGeneratedName()
}

private fun KClass<*>.typeName(targetLanguage: TargetLanguage = TargetLanguage.KOTLIN, forceQualified: Boolean = false): String {
  val qualified = qualifiedName
  val rendered = when {
    qualified == null -> simpleName ?: "Anonymous"
    targetLanguage == TargetLanguage.JAVA && qualified in JavaBackedKotlinTypeAliases -> JavaBackedKotlinTypeAliases.getValue(qualified)
    targetLanguage == TargetLanguage.JAVA && qualified == "java.lang.Object" -> JavaTopTypeName
    forceQualified -> qualified
    qualified == KotlinTopTypeName -> qualified
    qualified.startsWith("kotlin.collections.") -> qualified.removePrefix("kotlin.collections.")
    qualified.startsWith("kotlin.ranges.") -> qualified.removePrefix("kotlin.ranges.")
    qualified.startsWith("kotlin.sequences.") -> qualified.removePrefix("kotlin.sequences.")
    qualified == "kotlin.random.Random" -> qualified
    qualified.startsWith("kotlin.") -> qualified.removePrefix("kotlin.")
    qualified.startsWith("java.lang.") -> qualified.removePrefix("java.lang.")
    qualified == "java.util.Random" -> qualified
    qualified.startsWith("java.util.stream.") -> qualified.removePrefix("java.util.stream.")
    qualified.startsWith("java.util.") -> qualified.removePrefix("java.util.")
    else -> qualified
  }
  return rendered.replace('$', '.').noWhitespaceToken()
}

private fun KClass<*>.tokenName(): String = (simpleName ?: typeName()).noWhitespaceToken()

private fun String.isGeneratedName(): Boolean = contains('$') || startsWith("<") || endsWith("DefaultImpls")

private fun String.noWhitespaceToken(): String = replace(Regex("\\s+"), "_")

private fun String.sanitizedCnfNamePart(): String {
  val sanitized = map { character -> if (character.isLetterOrDigit()) character else '_' }.joinToString("")
    .trim('_')
    .ifBlank { "TOKEN" }
  return sanitized.take(32)
}
