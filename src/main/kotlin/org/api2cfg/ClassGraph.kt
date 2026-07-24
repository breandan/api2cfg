package org.api2cfg

import io.github.classgraph.BaseTypeSignature
import io.github.classgraph.ClassGraph
import io.github.classgraph.ClassInfo
import io.github.classgraph.ClassRefTypeSignature
import io.github.classgraph.ClassTypeSignature
import io.github.classgraph.MethodInfo
import io.github.classgraph.ScanResult
import io.github.classgraph.TypeArgument
import io.github.classgraph.TypeParameter
import io.github.classgraph.TypeSignature as ClassGraphTypeSignature
import io.github.classgraph.TypeVariableSignature
import java.io.File
import java.nio.file.Files
import java.util.IdentityHashMap

private const val CLASS_GRAPH_MAX_CALL_ARITY = 3
private const val CLASS_GRAPH_MAX_TYPE_ARGUMENTS = 2
private const val CLASS_GRAPH_SUBTYPE_DEPTH = 8
private const val CLASS_GRAPH_TOP_TYPE = "java.lang.Object"

private val DefaultClassGraphGroundClasses = listOf(
  "java.lang.Boolean",
  "java.lang.Integer",
  "java.lang.Long",
  "java.lang.Double",
  "java.lang.String",
)

private val JavaPrimitiveTypeNames = setOf(
  "boolean",
  "byte",
  "char",
  "double",
  "float",
  "int",
  "long",
  "short",
  "void",
)

/**
 * Deliberately small options for the bytecode-only generator.
 *
 * [groundClassNames] is the finite domain used to instantiate class and method
 * type variables. Concrete, non-generic classes in the requested package fill
 * any remaining slots up to [maxGroundTypes].
 */
data class ClassGraphGeneratorOptions(
  val packageName: String,
  val normalizeChomskyNormalForm: Boolean = false,
  val groundClassNames: List<String> = DefaultClassGraphGroundClasses,
  val maxGroundTypes: Int = 8,
  val maxTypeVariablesPerCallable: Int = 2,
  val maxTypeDepth: Int = 2,
) {
  init {
    require(packageName.isNotBlank()) { "Package name must not be blank" }
    require(maxGroundTypes > 0) { "maxGroundTypes must be positive" }
    require(maxTypeVariablesPerCallable >= 0) { "maxTypeVariablesPerCallable must not be negative" }
    require(maxTypeDepth >= 0) { "maxTypeDepth must not be negative" }
  }
}

/**
 * A ClassGraph-backed, JVM-language-agnostic API-to-CFG generator.
 *
 * It intentionally supports only public constructors and methods, call arities
 * zero through three, invariant class types, type variables with upper bounds,
 * and a finite ground universe. Unsupported bytecode signatures are omitted.
 */
class ClassGraphCFGGenerator(private val options: ClassGraphGeneratorOptions) {
  fun generate(): GeneratedGrammar {
    val graph = ClassGraph()
      .enableClassInfo()
      .enableMethodInfo()
      .enableSystemJarsAndModules()
      .acceptPackagesNonRecursive(options.packageName)

    if (options.groundClassNames.isNotEmpty()) {
      graph.acceptClasses(*options.groundClassNames.toTypedArray())
    }

    return graph.scan().use(::generate)
  }

  private fun generate(scanResult: ScanResult): GeneratedGrammar {
    val targetInfos = scanResult.allClasses
      .asSequence()
      .filter { it.packageName == options.packageName }
      .filter(::isSupportedApiClass)
      .sortedBy { it.name }
      .toList()
    val supportInfos = options.groundClassNames
      .mapNotNull(scanResult::getClassInfo)
      .filter(::isSupportedMetadataClass)

    val modelsByName = linkedMapOf<String, ClassGraphClassModel>()
    (targetInfos + supportInfos).forEach { info ->
      classModel(info)?.let { model -> modelsByName.putIfAbsent(info.name, model) }
    }
    val targetModels = targetInfos.mapNotNull { info -> modelsByName[info.name] }
    val metadataModels = modelsByName.values.toList()

    val subtypeTemplates = metadataModels.flatMap(::directSubtypeTemplates).distinct()
    val subtypeIndex = SubtypeIndex(
      templates = subtypeTemplates,
      maxAlternativesPerType = 80,
      topTypeName = CLASS_GRAPH_TOP_TYPE,
    )
    val erasedSubtypeIndex = ErasedClassGraphSubtypeIndex(subtypeTemplates)
    val groundTypes = groundTypes(targetModels, metadataModels)
    val ambiguousOverloads = ambiguousOverloads(targetModels, erasedSubtypeIndex)

    val calls = targetModels
      .flatMap { model ->
        callPatterns(model, ambiguousOverloads[model.info.name].orEmpty())
      }
      .flatMap { pattern ->
        monomorphize(pattern, groundTypes, subtypeIndex, erasedSubtypeIndex)
      }
      .distinct()
    val body = CFG.fromCalls(
      calls = calls,
      targetLanguage = TargetLanguage.JAVA,
      subtypeRelation = { actual, expected ->
        isClassGraphSubtype(actual, expected, subtypeIndex, erasedSubtypeIndex)
      },
    ).withoutNonGeneratingProductions()
    val withStart = body.withStartProductions()
    val finalGrammar = if (options.normalizeChomskyNormalForm) {
      withStart.toChomskyNormalForm()
    } else {
      withStart
    }

    return finalGrammar.toGeneratedGrammar()
  }

  private fun classModel(info: ClassInfo): ClassGraphClassModel? {
    val signature = runCatching { info.typeSignatureOrTypeDescriptor }.getOrNull() ?: return null
    if (signature.typeParameters.size > options.maxTypeVariablesPerCallable) return null
    return ClassGraphClassModel(info, signature)
  }

  private fun directSubtypeTemplates(model: ClassGraphClassModel): List<SubtypeTemplate> {
    val renderer = ClassGraphTypeRenderer(model.signature.typeParameters)
    val subtype = renderer.classPattern(model.info.name)
    val directSupertypes = listOfNotNull(model.signature.superclassSignature) +
        model.signature.superinterfaceSignatures

    return directSupertypes.mapNotNull { signature ->
      val supertype = renderer.render(signature) ?: return@mapNotNull null
      if (supertype == subtype || supertype == TypeExpr.Applied(CLASS_GRAPH_TOP_TYPE)) {
        null
      } else {
        SubtypeTemplate(supertype, subtype)
      }
    }
  }

  private fun groundTypes(
    targetModels: List<ClassGraphClassModel>,
    metadataModels: List<ClassGraphClassModel>,
  ): List<TypeExpr> {
    val metadataByName = metadataModels.associateBy { model -> model.info.name }
    val result = linkedSetOf<TypeExpr>()
    options.groundClassNames
      .asSequence()
      .mapNotNull(metadataByName::get)
      .filter { model -> model.signature.typeParameters.isEmpty() }
      .map { model -> TypeExpr.Applied(renderClassGraphClassName(model.info.name)) }
      .take(options.maxGroundTypes)
      .forEach(result::add)

    targetModels
      .asSequence()
      .filter { model -> model.signature.typeParameters.isEmpty() }
      .filter { model -> !model.info.isAbstract && !model.info.isInterface }
      .map { model -> TypeExpr.Applied(renderClassGraphClassName(model.info.name)) }
      .take(options.maxGroundTypes - result.size)
      .forEach(result::add)

    return result.toList()
  }

  private fun ambiguousOverloads(
    targetModels: List<ClassGraphClassModel>,
    erasedSubtypeIndex: ErasedClassGraphSubtypeIndex,
  ): Map<String, Set<ClassGraphMethodKey>> = buildMap {
    for (receiver in targetModels) {
      val receiverName = renderClassGraphClassName(receiver.info.name)
      val descriptorsByKey = linkedMapOf<ClassGraphMethodKey, MutableSet<String>>()
      targetModels
        .asSequence()
        .filter { candidate ->
          erasedSubtypeIndex.isSubtypeOf(
            renderClassGraphClassName(candidate.info.name),
            receiverName,
          )
        }
        .flatMap { candidate ->
          runCatching { candidate.info.methodInfo.asSequence() }.getOrDefault(emptySequence())
        }
        .filter(::isOverloadCandidate)
        .forEach { method ->
          val key = ClassGraphMethodKey(method.name, method.parameterInfo.size)
          descriptorsByKey.getOrPut(key, ::linkedSetOf) +=
            method.typeDescriptorStr.substringBefore(')') + ")"
        }

      val ambiguous = descriptorsByKey
        .filterValues { descriptors -> descriptors.size > 1 }
        .keys
      if (ambiguous.isNotEmpty()) put(receiver.info.name, ambiguous)
    }
  }

  private fun callPatterns(model: ClassGraphClassModel, ambiguousOverloads: Set<ClassGraphMethodKey>): List<ClassGraphCallPattern> = buildList {
    if (!model.info.isAbstract && !model.info.isInterface) {
      model.info.declaredConstructorInfo
        .asSequence()
        .filter(::isSupportedConstructor)
        .mapNotNull { constructor -> constructorPattern(model, constructor) }
        .forEach(::add)
    }

    model.info.declaredMethodInfo
      .asSequence()
      .filter { method -> ClassGraphMethodKey(method.name, method.parameterInfo.size) !in ambiguousOverloads }
      .filter(::isSupportedMethod)
      .mapNotNull { method -> methodPattern(model, method) }
      .forEach(::add)
  }

  private fun constructorPattern(
    model: ClassGraphClassModel,
    constructor: MethodInfo,
  ): ClassGraphCallPattern? = runCatching {
    val signature = constructor.typeSignatureOrTypeDescriptor
    val renderer = ClassGraphTypeRenderer(model.signature.typeParameters, signature.typeParameters)
    val parameters = constructor.parameterInfo.map { parameter ->
      renderer.render(parameter.typeSignatureOrTypeDescriptor) ?: return null
    }
    val bounds = renderer.bounds() ?: return null
    ClassGraphCallPattern(
      result = renderer.classPattern(model.info.name),
      receiver = null,
      staticOwner = null,
      name = model.info.simpleName,
      parameters = parameters,
      bounds = bounds,
    )
  }.getOrNull()

  private fun methodPattern(
    model: ClassGraphClassModel,
    method: MethodInfo,
  ): ClassGraphCallPattern? = runCatching {
    val signature = method.typeSignatureOrTypeDescriptor
    val renderer = ClassGraphTypeRenderer(model.signature.typeParameters, signature.typeParameters)
    val result = renderer.render(signature.resultType) ?: return null
    if (result == TypeExpr.Applied("void")) return null
    val parameters = method.parameterInfo.map { parameter ->
      renderer.render(parameter.typeSignatureOrTypeDescriptor) ?: return null
    }
    val bounds = renderer.bounds() ?: return null
    ClassGraphCallPattern(
      result = result,
      receiver = if (method.isStatic) null else renderer.classPattern(model.info.name),
      staticOwner = if (method.isStatic) model.info.simpleName else null,
      name = method.name,
      parameters = parameters,
      bounds = bounds,
    )
  }.getOrNull()

  private fun monomorphize(
    pattern: ClassGraphCallPattern,
    groundTypes: List<TypeExpr>,
    subtypeIndex: SubtypeIndex,
    erasedSubtypeIndex: ErasedClassGraphSubtypeIndex,
  ): List<CFGCall> {
    val types = listOfNotNull(pattern.result, pattern.receiver) + pattern.parameters
    val variables = types.flatMap { type -> type.variables() }.distinct().sorted()
    if (variables.size > options.maxTypeVariablesPerCallable) return emptyList()

    return classGraphSubstitutions(variables, groundTypes).mapNotNull { substitution ->
      if (!substitutionSatisfiesBounds(pattern.bounds, substitution, subtypeIndex, erasedSubtypeIndex)) {
        return@mapNotNull null
      }

      val result = pattern.result.substitute(substitution) ?: return@mapNotNull null
      val receiver = pattern.receiver?.substitute(substitution)
      val parameters = pattern.parameters.map { parameter ->
        parameter.substitute(substitution) ?: return@mapNotNull null
      }
      val instantiatedTypes = listOfNotNull(result, receiver) + parameters
      if (instantiatedTypes.any { type -> !type.isGround() || type.depth() > options.maxTypeDepth }) {
        return@mapNotNull null
      }

      CFGCall(
        result = result,
        receiver = receiver,
        staticOwner = pattern.staticOwner,
        name = pattern.name,
        parameters = parameters,
      )
    }.distinct().toList()
  }

  private fun substitutionSatisfiesBounds(
    bounds: Map<String, List<TypeExpr>>,
    substitution: Map<String, TypeExpr>,
    subtypeIndex: SubtypeIndex,
    erasedSubtypeIndex: ErasedClassGraphSubtypeIndex,
  ): Boolean = substitution.all { (variable, actual) ->
    bounds[variable].orEmpty().all { boundPattern ->
      val bound = boundPattern.substitute(substitution) ?: return@all false
      isClassGraphSubtype(actual, bound, subtypeIndex, erasedSubtypeIndex)
    }
  }

  private fun isClassGraphSubtype(
    actual: TypeExpr,
    expected: TypeExpr,
    subtypeIndex: SubtypeIndex,
    erasedSubtypeIndex: ErasedClassGraphSubtypeIndex,
  ): Boolean {
    if (actual == expected) return true
    val actualApplied = actual as? TypeExpr.Applied ?: return false
    val expectedApplied = expected as? TypeExpr.Applied ?: return false
    if (!actualApplied.isClassGraphReferenceType() || !expectedApplied.isClassGraphReferenceType()) return false
    if (subtypeIndex.isSubtypeOf(actual, expected, CLASS_GRAPH_SUBTYPE_DEPTH)) return true
    return expectedApplied.arguments.isEmpty() &&
        erasedSubtypeIndex.isSubtypeOf(actualApplied.name, expectedApplied.name)
  }

  private fun isSupportedApiClass(info: ClassInfo): Boolean =
    isSupportedMetadataClass(info) &&
        info.isPublic &&
        !info.isSynthetic &&
        !info.isAnnotation &&
        !info.isInnerClass &&
        '$' !in info.name

  private fun isSupportedMetadataClass(info: ClassInfo): Boolean =
    !info.isArrayClass && '$' !in info.name

  private fun isSupportedConstructor(constructor: MethodInfo): Boolean = runCatching {
    constructor.isConstructor &&
        constructor.isPublic &&
        !constructor.isSynthetic &&
        !constructor.isVarArgs &&
        constructor.parameterInfo.size <= CLASS_GRAPH_MAX_CALL_ARITY &&
        constructor.typeSignatureOrTypeDescriptor.typeParameters.size <= options.maxTypeVariablesPerCallable
  }.getOrDefault(false)

  private fun isSupportedMethod(method: MethodInfo): Boolean = runCatching {
    isOverloadCandidate(method) &&
        method.typeSignatureOrTypeDescriptor.typeParameters.size <= options.maxTypeVariablesPerCallable
  }.getOrDefault(false)

  private fun isOverloadCandidate(method: MethodInfo): Boolean =
    !method.isConstructor &&
        method.isPublic &&
        !method.isSynthetic &&
        !method.isBridge &&
        !method.isVarArgs &&
        method.parameterInfo.size <= CLASS_GRAPH_MAX_CALL_ARITY &&
        method.name.isClassGraphIdentifier()
}

private data class ClassGraphClassModel(val info: ClassInfo, val signature: ClassTypeSignature)

private data class ClassGraphCallPattern(
  val result: TypeExpr,
  val receiver: TypeExpr?,
  val staticOwner: String?,
  val name: String,
  val parameters: List<TypeExpr>,
  val bounds: Map<String, List<TypeExpr>>,
)

private data class ClassGraphMethodKey(val name: String, val arity: Int)

private class ClassGraphTypeRenderer(
  private val classParameters: List<TypeParameter>,
  private val methodParameters: List<TypeParameter> = emptyList(),
) {
  private val namesByParameter = IdentityHashMap<TypeParameter, String>()

  init {
    classParameters.forEachIndexed { index, parameter -> namesByParameter[parameter] = "C$index" }
    methodParameters.forEachIndexed { index, parameter -> namesByParameter[parameter] = "M$index" }
  }

  fun classPattern(className: String): TypeExpr =
    TypeExpr.Applied(
      name = renderClassGraphClassName(className),
      arguments = classParameters.map { parameter -> TypeExpr.Variable(namesByParameter.getValue(parameter)) },
    )

  fun render(signature: ClassGraphTypeSignature): TypeExpr? = when (signature) {
    is BaseTypeSignature -> TypeExpr.Applied(signature.typeStr)
    is ClassRefTypeSignature -> renderClassReference(signature)
    is TypeVariableSignature -> variableName(signature)?.let(TypeExpr::Variable)
    else -> null
  }

  fun bounds(): Map<String, List<TypeExpr>>? {
    val result = linkedMapOf<String, List<TypeExpr>>()
    for (parameter in classParameters + methodParameters) {
      val signatures = listOfNotNull(parameter.classBound) + parameter.interfaceBounds
      val rendered = signatures.map { signature -> render(signature) ?: return null }
        .filter { bound -> bound != TypeExpr.Applied(CLASS_GRAPH_TOP_TYPE) }
      result[namesByParameter.getValue(parameter)] = rendered
    }
    return result
  }

  private fun renderClassReference(signature: ClassRefTypeSignature): TypeExpr? {
    if (signature.suffixes.isNotEmpty()) return null
    if ('$' in signature.fullyQualifiedClassName) return null
    if (signature.typeArguments.size > CLASS_GRAPH_MAX_TYPE_ARGUMENTS) return null
    val arguments = signature.typeArguments.map { argument ->
      if (argument.wildcard != TypeArgument.Wildcard.NONE) return null
      val argumentSignature = argument.typeSignature ?: return null
      render(argumentSignature) ?: return null
    }
    return TypeExpr.Applied(renderClassGraphClassName(signature.fullyQualifiedClassName), arguments)
  }

  private fun variableName(signature: TypeVariableSignature): String? {
    val resolved = runCatching(signature::resolve).getOrNull()
    if (resolved != null) namesByParameter[resolved]?.let { return it }
    methodParameters.firstOrNull { parameter -> parameter.name == signature.name }
      ?.let { parameter -> return namesByParameter[parameter] }
    classParameters.firstOrNull { parameter -> parameter.name == signature.name }
      ?.let { parameter -> return namesByParameter[parameter] }
    return null
  }
}

private class ErasedClassGraphSubtypeIndex(templates: List<SubtypeTemplate>) {
  private val directSupertypes = templates
    .mapNotNull { template ->
      val supertype = template.supertype as? TypeExpr.Applied ?: return@mapNotNull null
      val subtype = template.subtype as? TypeExpr.Applied ?: return@mapNotNull null
      subtype.name to supertype.name
    }
    .groupBy({ edge -> edge.first }, { edge -> edge.second })

  fun isSubtypeOf(actualName: String, expectedName: String): Boolean {
    if (actualName == expectedName) return true
    if (expectedName == CLASS_GRAPH_TOP_TYPE) return true

    val seen = linkedSetOf(actualName)
    val queue = ArrayDeque<String>()
    queue += actualName
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      for (supertype in directSupertypes[current].orEmpty()) {
        if (supertype == expectedName) return true
        if (seen.add(supertype)) queue += supertype
      }
    }
    return false
  }
}

private fun classGraphSubstitutions(
  variables: List<String>,
  groundTypes: List<TypeExpr>,
): Sequence<Map<String, TypeExpr>> =
  if (variables.isEmpty()) sequenceOf(emptyMap())
  else variables.fold(listOf(emptyMap<String, TypeExpr>())) { substitutions, variable ->
    substitutions.flatMap { substitution ->
      groundTypes.map { type -> substitution + (variable to type) }
    }
  }.asSequence()

private fun TypeExpr.Applied.isClassGraphReferenceType(): Boolean = name !in JavaPrimitiveTypeNames

private fun renderClassGraphClassName(className: String): String = when {
  className == CLASS_GRAPH_TOP_TYPE -> className
  className.startsWith("java.lang.") -> className.removePrefix("java.lang.")
  else -> className
}

private fun String.isClassGraphIdentifier(): Boolean =
  isNotEmpty() && Character.isJavaIdentifierStart(first()) && drop(1).all(Character::isJavaIdentifierPart)

private data class ClassGraphCommandLine(val packageName: String, val normalizeChomskyNormalForm: Boolean) {
  companion object {
    fun parse(args: Array<String>): ClassGraphCommandLine {
      val positional = mutableListOf<String>()
      var normalizeChomskyNormalForm = false
      for (argument in args) {
        when (argument) {
          "--cnf" -> normalizeChomskyNormalForm = true
          else -> {
            require(!argument.startsWith("-")) { "Unknown flag: $argument" }
            positional += argument
          }
        }
      }
      require(positional.size == 1) { "Expected exactly one positional argument: <package>" }
      return ClassGraphCommandLine(positional.single(), normalizeChomskyNormalForm)
    }
  }
}

/**
 * Bytecode entry point: `org.api2cfg.ClassGraphKt`.
 *
 * The Kotlin name is different from Main.kt's source-level `main`, while
 * @JvmName exposes the conventional JVM main method on this file facade.
 */
@JvmName("main")
fun classGraphMain(args: Array<String>) {
  val commandLine = ClassGraphCommandLine.parse(args)
  val outputFile = classGraphOutputFile(
    packageName = commandLine.packageName,
    normalizeChomskyNormalForm = commandLine.normalizeChomskyNormalForm,
  )
  val grammar = ClassGraphCFGGenerator(
    ClassGraphGeneratorOptions(
      packageName = commandLine.packageName,
      normalizeChomskyNormalForm = commandLine.normalizeChomskyNormalForm,
    ),
  ).generate()

  outputFile.parentFile?.mkdirs()
  Files.writeString(outputFile.toPath(), "${grammar.text}\n")
  println(
    "Wrote |P|=${grammar.productionCount}, |V|=${grammar.nonterminalCount}, " +
        "|Σ|=${grammar.terminalCount} to ${outputFile.path}",
  )
}

private fun classGraphOutputFile(
  packageName: String,
  normalizeChomskyNormalForm: Boolean,
): File {
  val suffix = if (normalizeChomskyNormalForm) "classgraph.cnf" else "classgraph.cfg"
  return File("gen", "${packageName.replace('.', '_')}.$suffix")
}
