# api2cfg

`api2cfg` generates a context-free grammar of type-safe expressions from a library API.

## Motivation

We aim to be sound and complete w.r.t. respect to the surface grammar, $G_0$:

```text
S    -> ID | ID ( ) | ID ( ARGS ) | S . S
ARGS -> S | S , S | S , S , S
```

Here, $\mathtt{ID}$ may be any identifier. The fragment contains names, calls with zero through three arguments, and arbitrary member chaining.

Let $e$ be a concrete surface term. We use two syntax-directed projections.
The surface projection $\pi_{\Sigma}(e)$ replaces every
identifier spelling by $\mathtt{ID}$ and preserves punctuation.

An identifier spelling need not determine a unique binding or type. Treat the
ambient environment as a finite relation among spellings, binding identities,
and complete static types:

$$
\Gamma\subseteq
  \mathtt{ID}\times\mathsf{Binding}\times\mathsf{Type},
\qquad
\mathsf{Ty}_{\Gamma}(x)
=\{\tau\mid\exists b.\ (x,b,\tau)\in\Gamma\}.
$$

The binding identity is retained because two declarations may have the same
spelling and type but different kinds or origins. An overload's type is its
whole correlated signature, not independent sets of parameter and result
types.

Fix arbitrary environments $\Delta_{\texttt{api}}$ and $\Gamma$. Let
$\mathcal{R}(e)$ be the set of legal resolved elaborations of $e$ under these
environments. A resolved term $\sigma$ records the binding, overload and type
instantiation, and static type selected at every identifier occurrence.
Resolution is performed for the whole term according to the language's
namespace, shadowing, overload, and typing rules; occurrence choices are not
assumed to form an unrestricted Cartesian product.

For a resolved term, $\pi_{\Gamma}(\sigma)$ replaces each context-bound
identifier occurrence of selected type $\tau$ by $\langle\tau\rangle$, while
preserving fixed API names and punctuation. It extends homomorphically over
the term. The type-directed projection of unresolved surface syntax is the set

$$
\Pi_{\Gamma}(e)
=\{\pi_{\Gamma}(\sigma)\mid
  \sigma\in\mathcal{R}(e)\}.
$$

For example, if $e=\texttt{x.f(y)}$ has a unique resolution in which
$\texttt{x}:X$ and $\texttt{y}:Y$, then

$$
\begin{aligned}
\pi_{\Sigma}(e) &= \texttt{ID . ID ( ID )},\\
\Pi_{\Gamma}(e)
  &=\{\langle X\rangle \texttt{ . f ( }\langle Y\rangle \texttt{ )}\}.
\end{aligned}
$$

If a function and a variable named $\texttt{abc}$ have types $F$ and $V$ in
the same scope, then distinct bindings $b_f$ and $b_v$ give

$$
(\texttt{abc},b_f,F),(\texttt{abc},b_v,V)\in\Gamma,
\qquad
\mathsf{Ty}_{\Gamma}(\texttt{abc})=\{F,V\}.
$$

Separate occurrences, such as those in $\texttt{pair(abc,abc())}$, may select
different bindings when the language permits it; the spelling alone does not
choose between them. Likewise, one concrete binding may admit the static
types $C$, $I_1$, and $I_2$ when $C$ is known to satisfy both contracts.

Thus $e$ matches the surface fragment exactly when
$\pi_{\Sigma}(e)\in\mathcal{L}(G_0)$. For resolved terms, type-directed
identifier equivalence remains equality under $\pi_{\Gamma}$:

$$
\sigma\cong_{\Gamma}\sigma'
\quad\stackrel{\mathrm{def}}{\Longleftrightarrow}\quad
\pi_{\Gamma}(\sigma)
=\pi_{\Gamma}(\sigma').
$$

Unresolved surface terms are type-compatible when some pair of legal
elaborations are equivalent:

$$
e\mathrel{\bowtie_{\Gamma}}e'
\quad\stackrel{\mathrm{def}}{\Longleftrightarrow}\quad
\Pi_{\Gamma}(e)
\cap\Pi_{\Gamma}(e')
\ne\varnothing.
$$

Compatibility is not generally transitive, so it is not itself an
equivalence relation. Different names sharing an admissible type nevertheless
match automatically, one or many occurrences at a time, while fixed API names
remain significant.

Let $G_{\texttt{api}}$ be the generated grammar. Write
$\Delta_{\texttt{api}};\Gamma\vdash e:\tau$ when some
$\sigma\in\mathcal{R}(e)$ has type $\tau$, and write $\langle\tau\rangle$ for
the grammar nonterminal representing type $\tau$. The
soundness-and-completeness target is:

$$
\forall e,\tau.\quad
\pi_{\Sigma}(e)\in\mathcal{L}(G_0)
\quad\Longrightarrow\quad
\left(
  \Delta_{\texttt{api}};\Gamma\vdash e:\tau
  \quad\Longleftrightarrow\quad
  \exists\varphi\in\Pi_{\Gamma}(e).\;
  \langle\tau\rangle
    \Rightarrow^{*}_{G_{\texttt{api}}}
    \varphi
\right).
$$

The right-to-left direction is **soundness**; the left-to-right direction is
**completeness**. A form
$\varphi\in\Pi_{\Gamma}(e)$ may retain type nonterminals, so it is a
sentential form rather than, in general, a terminal word in
$\mathcal{L}(G_{\texttt{api}})$. Ordinary language membership requires
continuing the derivation until every remaining type slot has been expanded.

When runtime or static metadata is insufficient, a backend should preserve soundness by omitting uncertain productions rather than inventing types.

## Backends

There are three backend families:

| Backend | API source | Example |
| --- | --- | --- |
| JVM reflection | Classes loaded on the runtime classpath | `./gradlew run --args='java.util'` |
| ClassGraph | JVM bytecode and generic-signature metadata | `./gradlew runClassGraph --args='java.util.function'` |
| Python | Runtime reflection or static `.pyi` files | `python3 main.py numpy`<br>`python3 main_pyi.py --alias=torch torch` |

The JVM reflection and Python reflection backends also support `--parameterized`. Every backend supports `--cnf`.

Generated grammars are written under `gen/` as `.cfg` files, or `.cnf` files when Chomsky normalization is requested.
