package org.api2cfg.cpp26

data class Cpp26ScannerOptions(
  val frontend: ClangFrontendOptions = ClangFrontendOptions(),
)

/** Discovers the selected installed C++26 standard-library declarations. */
class Cpp26Scanner(private val options: Cpp26ScannerOptions = Cpp26ScannerOptions()) {
  fun scan(headers: Collection<String> = Cpp26StandardLibraryCatalog.defaultHeaders): CppTypeGraph {
    val normalizedHeaders = headers
      .ifEmpty { Cpp26StandardLibraryCatalog.defaultHeaders }
      .map(Cpp26StandardLibraryCatalog::normalizeHeader)
      .distinct()
      .sorted()
    val frontend = ClangFrontend(options.frontend)
    val types = mutableListOf<CppTypeInfo>()
    val values = mutableListOf<CppValueInfo>()

    for (target in Cpp26StandardLibraryCatalog.targets(normalizedHeaders)) {
      val nodes = frontend.scan(target)
      if (target.valueName != null) {
        values += requireNotNull(extractValue(target, nodes)) {
          "Could not extract ${target.valueName} from Clang's <${target.header}> AST"
        }
      } else {
        types += requireNotNull(extractType(target, nodes)) {
          "Could not extract ${target.typeName} from Clang's <${target.header}> AST; roots were " +
              nodes.joinToString(limit = 12) { node ->
                "${node.kind}:${node.name}:complete=${node.completeDefinition}"
              }
        }
      }
    }

    val knownNames = types.flatMapTo(mutableSetOf()) { type -> listOf(type.name, type.canonicalName) }
    val externalBaseTypes = types
      .flatMap { type -> type.directBases.map(CppBaseInfo::type) }
      .distinctBy(CppTypeRef::render)
      .filter { base -> base.render() !in knownNames }
      .map { base ->
        CppTypeInfo(
          type = base,
          kind = CppTypeKind.EXTERNAL,
          header = normalizedHeaders.firstOrNull { header ->
            types.any { type -> type.header == header && type.directBases.any { it.type == base } }
          }.orEmpty(),
        )
      }

    return CppTypeGraph(
      types = types + externalBaseTypes,
      values = values,
      metadata = CppScanMetadata(
        languageVersion = options.frontend.standard,
        compiler = options.frontend.compiler,
        headers = normalizedHeaders,
      ),
    )
  }

  private fun extractValue(target: CppScanTarget, nodes: List<ClangAstNode>): CppValueInfo? {
    val simpleName = target.valueName?.substringAfterLast("::") ?: return null
    val declaration = nodes.lastOrNull { node -> node.kind == "VarDecl" && node.name == simpleName }
      ?: return null
    val type = declaration.type?.bestSpelling()?.let(CppTypeRef::parse) ?: return null
    return CppValueInfo(target.valueName, type, target.header)
  }

  private fun extractType(target: CppScanTarget, nodes: List<ClangAstNode>): CppTypeInfo? {
    val displayType = target.typeName?.let(CppTypeRef::parse) ?: return null
    val canonicalType = target.canonicalTypeName?.let(CppTypeRef::parse) ?: displayType
    val simpleName = target.astFilter.substringAfterLast("::")
    val record = selectDefinition(
      nodes,
      simpleName,
      target.instantiation != null,
      canonicalType.arguments,
    ) ?: return null
    val aliases = extractAliases(record, target, displayType, canonicalType)
    val directBases = record.bases.mapNotNull { base ->
      val type = parseMemberType(base.type.bestSpelling(), aliases, target, displayType, canonicalType)
        ?: return@mapNotNull null
      CppBaseInfo(
        type = type,
        access = base.access.toCppAccess(default = defaultBaseAccess(record.tagUsed)),
        isVirtual = base.virtual,
      )
    }
    val callables = extractCallables(record, target, displayType, canonicalType, aliases)
    return CppTypeInfo(
      type = displayType,
      canonicalType = canonicalType,
      kind = record.tagUsed.toCppTypeKind(),
      header = target.header,
      templateParameters = record.inner
        .filter { child -> child.kind == "TemplateTypeParmDecl" }
        .mapNotNull(ClangAstNode::name),
      directBases = directBases,
      constructors = callables.filter { callable -> callable.kind == CppCallableKind.CONSTRUCTOR },
      methods = callables.filter { callable -> callable.kind != CppCallableKind.CONSTRUCTOR },
      aliases = aliases.toSortedMap(),
    )
  }

  private fun selectDefinition(
    nodes: List<ClangAstNode>,
    simpleName: String,
    explicitInstantiation: Boolean,
    expectedArguments: List<CppTypeRef>,
  ): ClangAstNode? {
    if (explicitInstantiation) {
      nodes.lastOrNull { node ->
        node.kind == "ClassTemplateSpecializationDecl" &&
            node.name == simpleName &&
            node.completeDefinition &&
            node.locationFile?.endsWith("probe.cpp") == true
      }?.let { return it }
    }
    val definitions = nodes.flatMap { root -> root.recordDefinitions(simpleName) }
    return definitions.lastOrNull { definition ->
      val actualArguments = definition.inner
        .filter { child -> child.kind == "TemplateArgument" }
        .mapNotNull { child -> child.type?.bestSpelling()?.let(CppTypeRef::parse) }
      expectedArguments.isEmpty() ||
          actualArguments.take(expectedArguments.size).map(CppTypeRef::render) ==
          expectedArguments.map(CppTypeRef::render)
    } ?: definitions.lastOrNull()
  }

  private fun ClangAstNode.recordDefinitions(simpleName: String): List<ClangAstNode> = buildList {
    if (
      kind in setOf("CXXRecordDecl", "ClassTemplateSpecializationDecl") &&
      name == simpleName && completeDefinition
    ) add(this@recordDefinitions)
    inner.forEach { child -> addAll(child.recordDefinitions(simpleName)) }
  }

  private fun extractAliases(
    record: ClangAstNode,
    target: CppScanTarget,
    displayType: CppTypeRef,
    canonicalType: CppTypeRef,
  ): Map<String, CppTypeRef> = buildMap {
    for (node in record.inner) {
      if (node.kind == "CXXRecordDecl") {
        val nestedName = node.name ?: continue
        if (
          !node.implicit &&
          !nestedName.startsWith("_") &&
          nestedName != target.astFilter.substringAfterLast("::") &&
          nestedName != displayType.name.substringAfterLast("::")
        ) {
          put(nestedName, CppTypeRef("${displayType.render()}::$nestedName"))
        }
        continue
      }
      if (node.kind != "TypeAliasDecl" && node.kind != "TypedefDecl") continue
      val aliasName = node.name ?: continue
      val spelling = node.type?.bestSpelling() ?: continue
      val parsed = runCatching { CppTypeRef.parse(spelling) }.getOrNull() ?: continue
      put(aliasName, normalizeSelfType(parsed, target, displayType, canonicalType))
    }
  }

  private fun extractCallables(
    record: ClangAstNode,
    target: CppScanTarget,
    displayType: CppTypeRef,
    canonicalType: CppTypeRef,
    aliases: Map<String, CppTypeRef>,
  ): List<CppCallableInfo> {
    val result = mutableListOf<CppCallableInfo>()
    var access = defaultMemberAccess(record.tagUsed)
    for (node in record.inner) {
      when (node.kind) {
        "AccessSpecDecl" -> access = node.access.toCppAccess(access)
        "CXXConstructorDecl", "CXXMethodDecl" -> callable(
          node,
          access,
          target,
          displayType,
          canonicalType,
          aliases,
        )?.let(result::add)
        "FunctionTemplateDecl" -> node.inner
          .filter { child -> child.kind in setOf("CXXConstructorDecl", "CXXMethodDecl") }
          .mapNotNull { child ->
            callable(child, access, target, displayType, canonicalType, aliases)
          }
          .forEach(result::add)
      }
    }
    return result.distinctBy(CppCallableInfo::signature).sortedBy(CppCallableInfo::signature)
  }

  private fun callable(
    node: ClangAstNode,
    access: CppAccess,
    target: CppScanTarget,
    displayType: CppTypeRef,
    canonicalType: CppTypeRef,
    aliases: Map<String, CppTypeRef>,
  ): CppCallableInfo? {
    if (node.implicit) return null
    val name = node.name ?: return null
    val functionSpelling = node.type?.spelling ?: return null
    val parameterTypes = node.inner
      .filter { child -> child.kind == "ParmVarDecl" }
      .map { parameter ->
        val spelling = parameter.type?.bestSpelling() ?: return null
        parseMemberType(spelling, aliases, target, displayType, canonicalType) ?: return null
      }
    val isConstructor = node.kind == "CXXConstructorDecl"
    val resultType = if (isConstructor) {
      displayType
    } else {
      val spelling = functionResultSpelling(functionSpelling) ?: return null
      parseMemberType(spelling, aliases, target, displayType, canonicalType) ?: return null
    }
    val kind = when {
      isConstructor -> CppCallableKind.CONSTRUCTOR
      name == "operator()" -> CppCallableKind.INVOCATION
      node.storageClass == "static" -> CppCallableKind.STATIC_METHOD
      else -> CppCallableKind.METHOD
    }
    return CppCallableInfo(
      owner = displayType,
      name = name,
      kind = kind,
      resultType = resultType,
      parameterTypes = parameterTypes,
      access = access,
      header = target.header,
      isConst = Regex("\\)\\s*const(?:\\s|$)").containsMatchIn(functionSpelling),
      isDeleted = node.deleted,
      isVariadic = node.variadic || "..." in functionSpelling,
    )
  }

  private fun parseMemberType(
    spelling: String,
    aliases: Map<String, CppTypeRef>,
    target: CppScanTarget,
    displayType: CppTypeRef,
    canonicalType: CppTypeRef,
  ): CppTypeRef? {
    var parsed = runCatching { CppTypeRef.parse(spelling) }.getOrNull() ?: return null
    parsed = parsed.copy(arguments = parsed.arguments.map { argument -> resolveAliases(argument, aliases) })
    parsed = resolveAliases(parsed, aliases)
    parsed = normalizeSelfType(parsed, target, displayType, canonicalType)
    parsed = Cpp26StandardLibraryCatalog.preferredType(parsed)
    return qualifyStandardType(parsed)
  }

  private fun resolveAliases(type: CppTypeRef, aliases: Map<String, CppTypeRef>): CppTypeRef {
    var resolved = type.copy(arguments = type.arguments.map { argument -> resolveAliases(argument, aliases) })
    repeat(8) {
      val aliasName = resolved.name.substringAfterLast("::")
      val replacement = aliases[aliasName] ?: return@repeat
      val next = replacement.copy(
        arguments = replacement.arguments.map { argument -> resolveAliases(argument, aliases) },
        isConst = resolved.isConst || replacement.isConst,
        isVolatile = resolved.isVolatile || replacement.isVolatile,
        pointers = replacement.pointers + resolved.pointers,
        reference = resolved.reference ?: replacement.reference,
      )
      if (next == resolved) return resolved
      resolved = next
    }
    return resolved
  }

  private fun qualifyStandardType(type: CppTypeRef): CppTypeRef {
    val qualifiedArguments = type.arguments.map(::qualifyStandardType)
    val name = type.name
    val qualifiedName = if (
      !name.startsWith("std::") &&
      !name.startsWith("_") &&
      name !in CppBuiltinTypeNames &&
      !name.matches(Regex("[-+]?((0[xX][0-9a-fA-F]+)|([0-9]+))([uUlL]*)")) &&
      name.matches(Regex("[A-Za-z][A-Za-z0-9_:]*"))
    ) {
      "std::$name"
    } else {
      name
    }
    return type.copy(name = qualifiedName, arguments = qualifiedArguments)
  }

  private companion object {
    val CppBuiltinTypeNames = setOf(
      "bool", "char", "char8_t", "char16_t", "char32_t", "wchar_t",
      "short", "unsigned_short", "int", "unsigned", "long", "unsigned_long",
      "long_long", "unsigned_long_long", "float", "double", "long_double", "void",
    )
  }

  private fun normalizeSelfType(
    type: CppTypeRef,
    target: CppScanTarget,
    displayType: CppTypeRef,
    canonicalType: CppTypeRef,
  ): CppTypeRef {
    if (type.pointers.isNotEmpty()) return type
    val baseName = type.name.substringAfterLast("::")
    val targetName = target.astFilter.substringAfterLast("::")
    val displayName = displayType.name.substringAfterLast("::")
    val renderedBase = type.copy(isConst = false, isVolatile = false, reference = null).render()
    val isSelf = baseName == targetName || baseName == displayName ||
        renderedBase == displayType.render() ||
        renderedBase == canonicalType.render()
    return if (isSelf) {
      displayType.copy(
        isConst = type.isConst,
        isVolatile = type.isVolatile,
        reference = type.reference,
      )
    } else {
      type
    }
  }

  private fun functionResultSpelling(functionType: String): String? {
    var angleDepth = 0
    var bracketDepth = 0
    functionType.forEachIndexed { index, character ->
      when (character) {
        '<' -> angleDepth += 1
        '>' -> if (angleDepth > 0) angleDepth -= 1
        '[' -> bracketDepth += 1
        ']' -> if (bracketDepth > 0) bracketDepth -= 1
        '(' -> if (angleDepth == 0 && bracketDepth == 0) {
          return functionType.substring(0, index).trim().takeIf(String::isNotEmpty)
        }
      }
    }
    return null
  }

  private fun defaultMemberAccess(tagUsed: String?): CppAccess =
    if (tagUsed == "class") CppAccess.PRIVATE else CppAccess.PUBLIC

  private fun defaultBaseAccess(tagUsed: String?): CppAccess = defaultMemberAccess(tagUsed)

  private fun String?.toCppAccess(default: CppAccess): CppAccess = when (this) {
    "public" -> CppAccess.PUBLIC
    "protected" -> CppAccess.PROTECTED
    "private" -> CppAccess.PRIVATE
    else -> default
  }

  private fun String?.toCppTypeKind(): CppTypeKind = when (this) {
    "struct" -> CppTypeKind.STRUCT
    "union" -> CppTypeKind.UNION
    "enum" -> CppTypeKind.ENUM
    else -> CppTypeKind.CLASS
  }
}
