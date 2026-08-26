package org.api2cfg.cpp26

import com.fasterxml.jackson.core.JsonFactory
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.core.JsonToken
import java.io.InputStream

/** The small declaration-only projection that we retain from Clang's AST. */
internal data class ClangAstNode(
  val kind: String,
  val name: String? = null,
  val tagUsed: String? = null,
  val type: ClangAstType? = null,
  val locationFile: String? = null,
  val completeDefinition: Boolean = false,
  val implicit: Boolean = false,
  val deleted: Boolean = false,
  val variadic: Boolean = false,
  val storageClass: String? = null,
  val access: String? = null,
  val bases: List<ClangAstBase> = emptyList(),
  val inner: List<ClangAstNode> = emptyList(),
)

internal data class ClangAstType(
  val spelling: String,
  val desugaredSpelling: String? = null,
) {
  fun bestSpelling(): String = desugaredSpelling ?: spelling
}

internal data class ClangAstBase(
  val access: String?,
  val type: ClangAstType,
  val virtual: Boolean = false,
)

/**
 * Reads the JSON value sequence emitted by `-ast-dump=json -ast-dump-filter=...`.
 *
 * Filtered Clang output is deliberately not a JSON array: every matching
 * declaration is a separate root value. This parser also skips statement and
 * expression subtrees, which keeps memory proportional to declarations even
 * when a standard-library method has a large inline body.
 */
internal class ClangAstJsonReader(
  private val factory: JsonFactory = JsonFactory.builder().build(),
) {
  fun read(input: InputStream): List<ClangAstNode> = buildList {
    factory.createParser(input).use { parser ->
      while (parser.nextToken() != null) {
        if (parser.currentToken() == JsonToken.START_OBJECT) {
          val node = readNode(parser)
          if (node.kind in RetainedDeclarationKinds) add(node)
        } else {
          parser.skipChildren()
        }
      }
    }
  }

  private fun readNode(parser: JsonParser): ClangAstNode {
    check(parser.currentToken() == JsonToken.START_OBJECT)
    var kind = ""
    var name: String? = null
    var tagUsed: String? = null
    var type: ClangAstType? = null
    var locationFile: String? = null
    var completeDefinition = false
    var implicit = false
    var deleted = false
    var variadic = false
    var storageClass: String? = null
    var access: String? = null
    var bases = emptyList<ClangAstBase>()
    var inner = emptyList<ClangAstNode>()

    while (parser.nextToken() != JsonToken.END_OBJECT) {
      val fieldName = parser.currentName()
      parser.nextToken()
      when (fieldName) {
        "kind" -> kind = parser.stringValueOrEmpty()
        "name" -> name = parser.stringValueOrNull()
        "tagUsed" -> tagUsed = parser.stringValueOrNull()
        "type" -> type = readType(parser)
        "loc" -> locationFile = readLocationFile(parser)
        "completeDefinition" -> completeDefinition = parser.booleanValueOrFalse()
        "isImplicit" -> implicit = parser.booleanValueOrFalse()
        "explicitlyDeleted", "isDeleted" -> deleted = parser.booleanValueOrFalse()
        "variadic", "isVariadic" -> variadic = parser.booleanValueOrFalse()
        "storageClass" -> storageClass = parser.stringValueOrNull()
        "access" -> access = parser.stringValueOrNull()
        "bases" -> bases = readBases(parser)
        "inner" -> inner = if (kind in DeclarationContainers) readInner(parser) else {
          parser.skipChildren()
          emptyList()
        }
        else -> parser.skipChildren()
      }
    }

    return ClangAstNode(
      kind = kind,
      name = name,
      tagUsed = tagUsed,
      type = type,
      locationFile = locationFile,
      completeDefinition = completeDefinition,
      implicit = implicit,
      deleted = deleted,
      variadic = variadic,
      storageClass = storageClass,
      access = access,
      bases = bases,
      inner = inner,
    )
  }

  private fun readInner(parser: JsonParser): List<ClangAstNode> {
    if (parser.currentToken() != JsonToken.START_ARRAY) {
      parser.skipChildren()
      return emptyList()
    }
    return buildList {
      while (parser.nextToken() != JsonToken.END_ARRAY) {
        if (parser.currentToken() == JsonToken.START_OBJECT) {
          val node = readNode(parser)
          if (node.kind in RetainedDeclarationKinds) add(node)
        } else {
          parser.skipChildren()
        }
      }
    }
  }

  private fun readType(parser: JsonParser): ClangAstType? {
    if (parser.currentToken() != JsonToken.START_OBJECT) {
      parser.skipChildren()
      return null
    }
    var spelling: String? = null
    var desugared: String? = null
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      val fieldName = parser.currentName()
      parser.nextToken()
      when (fieldName) {
        "qualType" -> spelling = parser.stringValueOrNull()
        "desugaredQualType" -> desugared = parser.stringValueOrNull()
        else -> parser.skipChildren()
      }
    }
    return spelling?.let { ClangAstType(it, desugared) }
  }

  private fun readBases(parser: JsonParser): List<ClangAstBase> {
    if (parser.currentToken() != JsonToken.START_ARRAY) {
      parser.skipChildren()
      return emptyList()
    }
    return buildList {
      while (parser.nextToken() != JsonToken.END_ARRAY) {
        if (parser.currentToken() != JsonToken.START_OBJECT) {
          parser.skipChildren()
          continue
        }
        var access: String? = null
        var type: ClangAstType? = null
        var virtual = false
        while (parser.nextToken() != JsonToken.END_OBJECT) {
          val fieldName = parser.currentName()
          parser.nextToken()
          when (fieldName) {
            "access", "writtenAccess" -> access = parser.stringValueOrNull() ?: access
            "type" -> type = readType(parser)
            "isVirtual" -> virtual = parser.booleanValueOrFalse()
            else -> parser.skipChildren()
          }
        }
        type?.let { add(ClangAstBase(access, it, virtual)) }
      }
    }
  }

  private fun readLocationFile(parser: JsonParser): String? {
    if (parser.currentToken() != JsonToken.START_OBJECT) {
      parser.skipChildren()
      return null
    }
    var file: String? = null
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      val fieldName = parser.currentName()
      parser.nextToken()
      when {
        fieldName == "file" -> file = parser.stringValueOrNull() ?: file
        parser.currentToken() == JsonToken.START_OBJECT -> {
          val nestedFile = readLocationFile(parser)
          if (file == null) file = nestedFile
        }
        else -> parser.skipChildren()
      }
    }
    return file
  }

  private fun JsonParser.stringValueOrNull(): String? =
    if (currentToken() == JsonToken.VALUE_STRING) valueAsString else null

  private fun JsonParser.stringValueOrEmpty(): String = stringValueOrNull().orEmpty()

  private fun JsonParser.booleanValueOrFalse(): Boolean =
    currentToken() == JsonToken.VALUE_TRUE

  private companion object {
    val DeclarationContainers = setOf(
      "ClassTemplateDecl",
      "ClassTemplateSpecializationDecl",
      "ClassTemplatePartialSpecializationDecl",
      "CXXRecordDecl",
      "FunctionTemplateDecl",
      "CXXConstructorDecl",
      "CXXMethodDecl",
      "FunctionDecl",
    )
    val RetainedDeclarationKinds = DeclarationContainers + setOf(
      "TemplateTypeParmDecl",
      "TemplateArgument",
      "AccessSpecDecl",
      "TypeAliasDecl",
      "TypedefDecl",
      "ParmVarDecl",
      "VarDecl",
    )
  }
}
