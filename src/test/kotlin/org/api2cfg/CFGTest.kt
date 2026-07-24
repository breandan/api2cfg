package org.api2cfg

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CFGTest {
  @Test
  fun `owns pruning and start production assembly`() {
    val a = TypeExpr.Applied("A")
    val b = TypeExpr.Applied("B")
    val c = TypeExpr.Applied("C")
    val d = TypeExpr.Applied("D")
    val body = setOf(
      Production.literal(a, "a"),
      Production.unit(b, a),
      Production.unit(c, d),
      Production.unit(d, c),
    )

    val pruned = CFG(body).withoutNonGeneratingProductions()
    assertEquals(setOf(Production.literal(a, "a"), Production.unit(b, a)), pruned.productions)

    val withStart = pruned.withStartProductions()
    assertEquals(
      setOf(Production.unit(CFG.DefaultStart, a), Production.unit(CFG.DefaultStart, b)),
      withStart.productions.filter { production -> production.lhs == CFG.DefaultStart }.toSet(),
    )
    assertEquals(withStart, withStart.withStartProductions())
  }

  @Test
  fun `owns Chomsky normalization`() {
    val a = TypeExpr.Applied("A")
    val b = TypeExpr.Applied("B")
    val expression = TypeExpr.Applied("Expression")
    val grammar = CFG(
      setOf(
        Production.literal(b, "b"),
        Production.unit(a, b),
        Production(
          expression,
          listOf(Symbol.Token("f"), Symbol.Type(a), Symbol.Token(")")),
        ),
      ),
    ).withStartProductions().toChomskyNormalForm()

    assertTrue(grammar.productions.isNotEmpty())
    assertTrue(grammar.productions.all { production ->
      when (production.rhs.size) {
        1 -> production.rhs.single() is Symbol.Token
        2 -> production.rhs.all { symbol -> symbol is Symbol.Type }
        else -> false
      }
    })
    assertTrue(grammar.productions.any { production -> production.lhs == CFG.DefaultStart })
  }

  @Test
  fun `owns constructor instance and static call productions`() {
    val box = TypeExpr.Applied("Box")
    val string = TypeExpr.Applied("String")
    val grammar = CFG.fromCalls(
      calls = listOf(
        CFGCall(box, receiver = null, staticOwner = null, name = "Box", parameters = listOf(string)),
        CFGCall(box, receiver = box, staticOwner = null, name = "next", parameters = listOf(string, string)),
        CFGCall(box, receiver = null, staticOwner = "Box", name = "create", parameters = emptyList()),
      ),
      targetLanguage = TargetLanguage.JAVA,
    )
    val rendered = grammar.productions.mapTo(mutableSetOf(), Production::render)

    assertTrue("Box -> Box ( String )" in rendered)
    assertTrue("Box -> Box . next ( String , String )" in rendered)
    assertTrue("Box -> Box . create ( )" in rendered)
  }
}
