package org.api2cfg.cpp26

/** A grammar-safe, structural representation of a C++ type spelling. */
data class CppTypeRef(
  val name: String,
  val arguments: List<CppTypeRef> = emptyList(),
  val isConst: Boolean = false,
  val isVolatile: Boolean = false,
  val pointers: List<CppPointerQualifier> = emptyList(),
  val reference: CppReferenceKind? = null,
) {
  init {
    require(name.isNotBlank()) { "C++ type name must not be blank" }
  }

  fun render(): String = buildString {
    if (isConst) append("const_")
    if (isVolatile) append("volatile_")
    append(normalizeCppName(name))
    if (arguments.isNotEmpty()) {
      append(arguments.joinToString(",", "<", ">", transform = CppTypeRef::render))
    }
    pointers.forEach { pointer ->
      append('*')
      if (pointer.isConst) append("_const")
      if (pointer.isVolatile) append("_volatile")
    }
    when (reference) {
      CppReferenceKind.LVALUE -> append('&')
      CppReferenceKind.RVALUE -> append("&&")
      null -> Unit
    }
  }

  fun withoutTopLevelCvAndReference(): CppTypeRef =
    copy(isConst = false, isVolatile = false, reference = null)

  fun withoutReference(): CppTypeRef = copy(reference = null)

  companion object {
    fun parse(spelling: String): CppTypeRef = CppTypeParser.parse(spelling)
  }
}

data class CppPointerQualifier(
  val isConst: Boolean = false,
  val isVolatile: Boolean = false,
)

enum class CppReferenceKind { LVALUE, RVALUE }

enum class CppTypeKind { CLASS, STRUCT, UNION, ENUM, EXTERNAL }

enum class CppAccess { PUBLIC, PROTECTED, PRIVATE }

enum class CppCallableKind { CONSTRUCTOR, METHOD, STATIC_METHOD, INVOCATION, FREE_FUNCTION }

data class CppBaseInfo(
  val type: CppTypeRef,
  val access: CppAccess = CppAccess.PUBLIC,
  val isVirtual: Boolean = false,
)

data class CppCallableInfo(
  val owner: CppTypeRef?,
  val name: String,
  val kind: CppCallableKind,
  val resultType: CppTypeRef,
  val parameterTypes: List<CppTypeRef>,
  val access: CppAccess = CppAccess.PUBLIC,
  val header: String,
  val isConst: Boolean = false,
  val isDeleted: Boolean = false,
  val isVariadic: Boolean = false,
) {
  val signature: String
    get() = buildString {
      append(owner?.render().orEmpty()).append('#').append(name).append('(')
      append(parameterTypes.joinToString(",", transform = CppTypeRef::render)).append(')')
      if (isConst) append("_const")
    }
}

data class CppTypeInfo(
  val type: CppTypeRef,
  val canonicalType: CppTypeRef = type,
  val kind: CppTypeKind,
  val header: String,
  val templateParameters: List<String> = emptyList(),
  val directBases: List<CppBaseInfo> = emptyList(),
  val constructors: List<CppCallableInfo> = emptyList(),
  val methods: List<CppCallableInfo> = emptyList(),
  val aliases: Map<String, CppTypeRef> = emptyMap(),
) {
  val name: String get() = type.render()
  val canonicalName: String get() = canonicalType.render()
  val declaredCallables: List<CppCallableInfo>
    get() = (constructors + methods).distinctBy(CppCallableInfo::signature).sortedBy(CppCallableInfo::signature)
}

data class CppValueInfo(
  val name: String,
  val type: CppTypeRef,
  val header: String,
)

data class CppScanMetadata(
  val languageVersion: String = "c++2c",
  val compiler: String = "clang++",
  val headers: List<String> = emptyList(),
)

/**
 * Immutable query surface for C++ declarations, analogous to a ClassGraph
 * [io.github.classgraph.ScanResult]. Display and canonical names are indexed.
 */
class CppTypeGraph(
  types: Iterable<CppTypeInfo>,
  values: Iterable<CppValueInfo> = emptyList(),
  val metadata: CppScanMetadata = CppScanMetadata(),
) {
  val allTypes: List<CppTypeInfo> = types
    .distinctBy { info -> info.name }
    .sortedBy(CppTypeInfo::name)
  val allClasses: List<CppTypeInfo> = allTypes.filter { info ->
    info.kind in setOf(CppTypeKind.CLASS, CppTypeKind.STRUCT, CppTypeKind.UNION, CppTypeKind.EXTERNAL)
  }
  val values: List<CppValueInfo> = values.distinctBy(CppValueInfo::name).sortedBy(CppValueInfo::name)

  private val typesByName: Map<String, CppTypeInfo> = buildMap {
    allTypes.forEach { info ->
      put(info.name, info)
      put(info.canonicalName, info)
    }
  }

  fun getTypeInfo(name: String): CppTypeInfo? =
    runCatching { CppTypeRef.parse(name).render() }.getOrNull()?.let(typesByName::get)

  fun getTypeInfo(type: CppTypeRef): CppTypeInfo? = typesByName[type.render()]

  fun getClassInfo(name: String): CppTypeInfo? = getTypeInfo(name)?.takeIf { it in allClasses }

  fun getValueInfo(name: String): CppValueInfo? = values.firstOrNull { value -> value.name == name }

  fun typesForHeader(header: String): List<CppTypeInfo> {
    val normalized = header.trim().removeSurrounding("<", ">")
    return allTypes.filter { info -> info.header == normalized }
  }

  fun valuesForHeader(header: String): List<CppValueInfo> {
    val normalized = header.trim().removeSurrounding("<", ">")
    return values.filter { value -> value.header == normalized }
  }

  fun aliasesOf(name: String): Map<String, CppTypeRef> = getTypeInfo(name)?.aliases.orEmpty()

  fun constructorsOf(name: String, publicOnly: Boolean = true): List<CppCallableInfo> =
    getTypeInfo(name)?.constructors.orEmpty()
      .filter { constructor -> !publicOnly || constructor.access == CppAccess.PUBLIC }
      .sortedBy(CppCallableInfo::signature)

  fun directSupertypeRefsOf(type: CppTypeRef, publicOnly: Boolean = true): List<CppTypeRef> =
    getTypeInfo(type)?.directBases.orEmpty()
      .filter { base -> !publicOnly || base.access == CppAccess.PUBLIC }
      .map(CppBaseInfo::type)
      .distinctBy(CppTypeRef::render)
      .sortedBy(CppTypeRef::render)

  fun directSupertypesOf(name: String, publicOnly: Boolean = true): List<CppTypeInfo> =
    getTypeInfo(name)?.let { directSupertypesOf(it.type, publicOnly) }.orEmpty()

  fun directSupertypesOf(type: CppTypeRef, publicOnly: Boolean = true): List<CppTypeInfo> =
    directSupertypeRefsOf(type, publicOnly).mapNotNull(::getTypeInfo)

  fun allSupertypesOf(name: String, publicOnly: Boolean = true): List<CppTypeInfo> =
    getTypeInfo(name)?.let { allSupertypesOf(it.type, publicOnly) }.orEmpty()

  fun allSupertypesOf(type: CppTypeRef, publicOnly: Boolean = true): List<CppTypeInfo> {
    val result = linkedMapOf<String, CppTypeInfo>()
    val queue = ArrayDeque<CppTypeRef>()
    queue += type
    while (queue.isNotEmpty()) {
      val current = queue.removeFirst()
      directSupertypesOf(current, publicOnly).forEach { supertype ->
        if (result.putIfAbsent(supertype.name, supertype) == null) queue += supertype.type
      }
    }
    result.remove(getTypeInfo(type)?.name)
    return result.values.sortedBy(CppTypeInfo::name)
  }

  fun subtypesOf(name: String, transitive: Boolean = true, publicOnly: Boolean = true): List<CppTypeInfo> {
    val expected = getTypeInfo(name) ?: return emptyList()
    return allTypes.filter { candidate ->
      if (candidate == expected) return@filter false
      val direct = directSupertypesOf(candidate.type, publicOnly)
      if (transitive) {
        direct.any { it == expected } || allSupertypesOf(candidate.type, publicOnly).any { it == expected }
      } else {
        direct.any { it == expected }
      }
    }
  }

  fun methodsOf(
    name: String,
    inherited: Boolean = true,
    publicOnly: Boolean = true,
  ): List<CppCallableInfo> = getTypeInfo(name)?.let { methodsOf(it.type, inherited, publicOnly) }.orEmpty()

  fun methodsOf(
    type: CppTypeRef,
    inherited: Boolean = true,
    publicOnly: Boolean = true,
  ): List<CppCallableInfo> {
    val info = getTypeInfo(type) ?: return emptyList()
    val owners = if (inherited) listOf(info) + allSupertypesOf(type, publicOnly) else listOf(info)
    return owners.asSequence()
      .flatMap { owner -> owner.methods.asSequence() }
      .filter { method -> !publicOnly || method.access == CppAccess.PUBLIC }
      .distinctBy(CppCallableInfo::signature)
      .sortedBy(CppCallableInfo::signature)
      .toList()
  }

  fun callablesOf(
    name: String,
    inheritedMethods: Boolean = true,
    publicOnly: Boolean = true,
  ): List<CppCallableInfo> =
    (constructorsOf(name, publicOnly) + methodsOf(name, inheritedMethods, publicOnly))
      .distinctBy(CppCallableInfo::signature)
      .sortedBy(CppCallableInfo::signature)

  fun getCallable(signature: String): CppCallableInfo? = allTypes.asSequence()
    .flatMap { type -> type.declaredCallables.asSequence() }
    .firstOrNull { callable -> callable.signature == signature }
}

private object CppTypeParser {
  fun parse(rawSpelling: String): CppTypeRef {
    var spelling = rawSpelling.trim()
      .replace(Regex("\\b(std::)?__(1|cxx11)::"), "std::")
      .replace(Regex("^(typename|class|struct|enum)\\s+"), "")
    require(spelling.isNotEmpty()) { "C++ type spelling must not be empty" }

    val reference = when {
      spelling.endsWith("&&") -> CppReferenceKind.RVALUE.also { spelling = spelling.dropLast(2).trimEnd() }
      spelling.endsWith('&') -> CppReferenceKind.LVALUE.also { spelling = spelling.dropLast(1).trimEnd() }
      else -> null
    }

    val pointerSegments = splitPointers(spelling)
    spelling = pointerSegments.first
    val pointers = pointerSegments.second.map { segment ->
      CppPointerQualifier(
        isConst = Regex("(?:^|\\s)const(?:\\s|$)").containsMatchIn(segment),
        isVolatile = Regex("(?:^|\\s)volatile(?:\\s|$)").containsMatchIn(segment),
      )
    }

    val words = spelling.split(Regex("\\s+")).toMutableList()
    val isConst = words.remove("const")
    val isVolatile = words.remove("volatile")
    spelling = words.joinToString(" ").trim()

    val templateStart = spelling.indexOfTopLevel('<')
    val (name, arguments) = if (templateStart >= 0 && spelling.endsWith('>')) {
      val baseName = spelling.substring(0, templateStart).trim()
      val argumentText = spelling.substring(templateStart + 1, spelling.length - 1)
      baseName to splitTopLevel(argumentText, ',').filter(String::isNotBlank).map(::parse)
    } else {
      spelling to emptyList()
    }
    return CppTypeRef(
      name = canonicalPrimitive(name),
      arguments = arguments,
      isConst = isConst,
      isVolatile = isVolatile,
      pointers = pointers,
      reference = reference,
    )
  }

  private fun splitPointers(spelling: String): Pair<String, List<String>> {
    var angleDepth = 0
    var parenDepth = 0
    val stars = mutableListOf<Int>()
    spelling.forEachIndexed { index, character ->
      when (character) {
        '<' -> angleDepth += 1
        '>' -> if (angleDepth > 0) angleDepth -= 1
        '(' -> parenDepth += 1
        ')' -> if (parenDepth > 0) parenDepth -= 1
        '*' -> if (angleDepth == 0 && parenDepth == 0) stars += index
      }
    }
    if (stars.isEmpty()) return spelling to emptyList()
    val base = spelling.substring(0, stars.first()).trimEnd()
    val qualifiers = stars.mapIndexed { index, star ->
      val end = stars.getOrNull(index + 1) ?: spelling.length
      spelling.substring(star + 1, end).trim()
    }
    return base to qualifiers
  }

  private fun String.indexOfTopLevel(needle: Char): Int {
    var parenDepth = 0
    forEachIndexed { index, character ->
      when (character) {
        '(' -> parenDepth += 1
        ')' -> if (parenDepth > 0) parenDepth -= 1
        needle -> if (parenDepth == 0) return index
      }
    }
    return -1
  }

  private fun splitTopLevel(text: String, delimiter: Char): List<String> {
    val result = mutableListOf<String>()
    var angleDepth = 0
    var parenDepth = 0
    var bracketDepth = 0
    var start = 0
    text.forEachIndexed { index, character ->
      when (character) {
        '<' -> angleDepth += 1
        '>' -> if (angleDepth > 0) angleDepth -= 1
        '(' -> parenDepth += 1
        ')' -> if (parenDepth > 0) parenDepth -= 1
        '[' -> bracketDepth += 1
        ']' -> if (bracketDepth > 0) bracketDepth -= 1
        delimiter -> if (angleDepth == 0 && parenDepth == 0 && bracketDepth == 0) {
          result += text.substring(start, index).trim()
          start = index + 1
        }
      }
    }
    result += text.substring(start).trim()
    return result
  }

  private fun canonicalPrimitive(name: String): String = when (name.trim()) {
    "signed int" -> "int"
    "unsigned int" -> "unsigned"
    "long int", "signed long", "signed long int" -> "long"
    "unsigned long int", "long unsigned int" -> "unsigned_long"
    "long long", "long long int", "signed long long", "signed long long int" -> "long_long"
    "unsigned long long", "unsigned long long int", "long long unsigned int" -> "unsigned_long_long"
    else -> name.trim()
  }
}

internal fun normalizeCppName(name: String): String = name.trim()
  .replace("std::__1::", "std::")
  .replace("std::__cxx11::", "std::")
  .replace(Regex("\\s+"), "_")
