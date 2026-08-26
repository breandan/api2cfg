package org.api2cfg.cpp26

/** A deliberately bounded, useful C++26 standard-library surface. */
object Cpp26StandardLibraryCatalog {
  val defaultHeaders: List<String> = listOf("vector", "random", "iostream")
  val supportedHeaders: Set<String> = defaultHeaders.toSet()

  internal fun targets(headers: Collection<String>): List<CppScanTarget> =
    headers.flatMap { header -> TargetsByHeader.getValue(header) }

  internal fun preferredType(type: CppTypeRef): CppTypeRef {
    if (type.pointers.isNotEmpty()) return type
    val unqualified = type.copy(isConst = false, isVolatile = false, reference = null)
    val preferredName = TargetsByHeader.values.flatten().firstNotNullOfOrNull { target ->
      val display = target.typeName ?: return@firstNotNullOfOrNull null
      val canonicalMatches = target.canonicalTypeName?.let(CppTypeRef::parse)?.render() == unqualified.render()
      if (
        canonicalMatches ||
        (target.preferAnySpecialization && unqualified.name == target.astFilter)
      ) display else null
    } ?: return type
    return CppTypeRef.parse(preferredName).copy(
      isConst = type.isConst,
      isVolatile = type.isVolatile,
      reference = type.reference,
    )
  }

  fun normalizeHeader(header: String): String {
    val normalized = header.trim().removeSurrounding("<", ">")
    require(normalized.isNotEmpty()) { "Header name must not be empty" }
    require(normalized in supportedHeaders) {
      "Unsupported C++26 header <$normalized>; supported headers: " +
          supportedHeaders.sorted().joinToString { "<$it>" }
    }
    return normalized
  }

  private val TargetsByHeader = mapOf(
    "vector" to listOf(
      CppScanTarget(
        header = "vector",
        astFilter = "std::vector",
        typeName = "std::vector<int>",
        canonicalTypeName = "std::vector<int,std::allocator<int>>",
        instantiation = "template class std::vector<int>;",
      ),
      CppScanTarget(
        header = "vector",
        astFilter = "std::vector",
        typeName = "std::vector<double>",
        canonicalTypeName = "std::vector<double,std::allocator<double>>",
        instantiation = "template class std::vector<double>;",
      ),
    ),
    "random" to listOf(
      CppScanTarget(
        header = "random",
        astFilter = "std::random_device",
        typeName = "std::random_device",
      ),
      CppScanTarget(
        header = "random",
        astFilter = "std::mersenne_twister_engine",
        typeName = "std::mt19937",
        canonicalTypeName = null,
        preferAnySpecialization = true,
        instantiation =
          "template class std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31, " +
              "0x9908b0dfUL, 11, 0xffffffffUL, 7, 0x9d2c5680UL, 15, 0xefc60000UL, 18, 1812433253UL>;",
      ),
      CppScanTarget(
        header = "random",
        astFilter = "std::uniform_int_distribution",
        typeName = "std::uniform_int_distribution<int>",
        instantiation =
          "template class std::uniform_int_distribution<int>;\n" +
              "template int std::uniform_int_distribution<int>::operator()<std::mt19937>(std::mt19937&);",
      ),
      CppScanTarget(
        header = "random",
        astFilter = "std::uniform_real_distribution",
        typeName = "std::uniform_real_distribution<double>",
        instantiation =
          "template class std::uniform_real_distribution<double>;\n" +
              "template double std::uniform_real_distribution<double>::operator()<std::mt19937>(std::mt19937&);",
      ),
    ),
    "iostream" to listOf(
      CppScanTarget(
        header = "iostream",
        astFilter = "std::ios_base",
        typeName = "std::ios_base",
      ),
      CppScanTarget(
        header = "iostream",
        astFilter = "std::basic_ios",
        typeName = "std::ios",
        canonicalTypeName = "std::basic_ios<char,std::char_traits<char>>",
        instantiation = "template class std::basic_ios<char>;",
      ),
      CppScanTarget(
        header = "iostream",
        astFilter = "std::basic_ostream",
        typeName = "std::ostream",
        canonicalTypeName = "std::basic_ostream<char,std::char_traits<char>>",
        instantiation = "template class std::basic_ostream<char>;",
      ),
      CppScanTarget(
        header = "iostream",
        astFilter = "std::basic_istream",
        typeName = "std::istream",
        canonicalTypeName = "std::basic_istream<char,std::char_traits<char>>",
        instantiation = "template class std::basic_istream<char>;",
      ),
      CppScanTarget(
        header = "iostream",
        astFilter = "std::basic_iostream",
        typeName = "std::iostream",
        canonicalTypeName = "std::basic_iostream<char,std::char_traits<char>>",
        instantiation = "template class std::basic_iostream<char>;",
      ),
      CppScanTarget("iostream", "std::cin", valueName = "std::cin"),
      CppScanTarget("iostream", "std::cout", valueName = "std::cout"),
      CppScanTarget("iostream", "std::cerr", valueName = "std::cerr"),
      CppScanTarget("iostream", "std::clog", valueName = "std::clog"),
    ),
  )
}

internal data class CppScanTarget(
  val header: String,
  val astFilter: String,
  val typeName: String? = null,
  val canonicalTypeName: String? = typeName,
  val valueName: String? = null,
  val instantiation: String? = null,
  val preferAnySpecialization: Boolean = false,
)
