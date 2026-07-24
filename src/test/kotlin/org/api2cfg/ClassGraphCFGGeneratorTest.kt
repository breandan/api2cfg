package org.api2cfg

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ClassGraphCFGGeneratorTest {
  private val packageName = "org.api2cfg.classgraphfixture"

  @Test
  fun `emits bounded generic calls subtyping and chains`() {
    val grammar = ClassGraphCFGGenerator(ClassGraphGeneratorOptions(packageName)).generate().text

    assertContains(grammar, "org.api2cfg.classgraphfixture.FluentBox<Integer> -> FluentBox ( Integer )")
    assertContains(
      grammar,
      "org.api2cfg.classgraphfixture.FluentBox<Integer> -> " +
          "org.api2cfg.classgraphfixture.FluentBox<Integer> . next ( Integer )",
    )
    assertContains(grammar, "Integer -> org.api2cfg.classgraphfixture.FluentBox<Integer> . value ( )")
    assertContains(
      grammar,
      "org.api2cfg.classgraphfixture.FluentBox<Integer> -> FluentBox . create ( )",
    )
    assertContains(
      grammar,
      "org.api2cfg.classgraphfixture.FluentBox<Integer> -> FluentBox . pair ( Integer , Integer )",
    )
    assertContains(
      grammar,
      "org.api2cfg.classgraphfixture.FluentBox<Integer> -> " +
          "org.api2cfg.classgraphfixture.FluentBox<Integer> . join ( Integer , Integer , Integer )",
    )
    assertContains(grammar, "Integer -> org.api2cfg.classgraphfixture.FluentBox<Integer> . echo ( Integer )")
    assertContains(
      grammar,
      "org.api2cfg.classgraphfixture.FluentBox<Integer> -> org.api2cfg.classgraphfixture.IntegerBox",
    )
    assertFalse("FluentBox<Boolean>" in grammar)
    assertFalse("FluentBox<String>" in grammar)
    assertFalse(". echo ( Boolean )" in grammar)
    assertFalse(". echo ( String )" in grammar)
    assertFalse("choose" in grammar)
    assertFalse("tooWide" in grammar)

    val callRightHandSides = grammar.lineSequence()
      .map { production -> production.substringAfter(" -> ") }
      .filter { rhs -> " ( " in rhs }
      .toList()
    val supportedCall = Regex("""^(?:\S+ \. )?\S+ \( (?:\S+(?: , \S+){0,2} )?\)$""")
    assertTrue(callRightHandSides.isNotEmpty())
    assertTrue(callRightHandSides.all(supportedCall::matches))
  }

  @Test
  fun `delegates CNF conversion to the shared implementation`() {
    val productions = ClassGraphCFGGenerator(
      ClassGraphGeneratorOptions(packageName, normalizeChomskyNormalForm = true),
    ).generate().text.lineSequence()
      .filter(String::isNotBlank)
      .map { production ->
        production.substringBefore(" -> ") to production.substringAfter(" -> ").split(" ")
      }
      .toList()
    val nonterminals = productions.mapTo(mutableSetOf()) { production -> production.first }

    assertTrue(productions.all { (_, rhs) ->
      when (rhs.size) {
        1 -> rhs.single() !in nonterminals
        2 -> rhs.all { symbol -> symbol in nonterminals }
        else -> false
      }
    })
  }
}
