# api2cfg

`api2cfg` generates a context-free grammar of type-safe expressions from a library API.

## Motivation

We aim to be sound and complete w.r.t. respect to the surface grammar, $G_0$:

```text
S    -> ID | ID ( ) | ID ( ARGS ) | S . S
ARGS -> S | S , S | S , S , S
```

Here, $\mathtt{ID}$ may be any identifier. The fragment contains names, calls with zero through three arguments, and arbitrary member chaining.

Let $\sigma$ be a resolved concrete term. We use two syntax-directed projections:

- $\pi_{\Sigma}(\sigma)$ replaces every identifier spelling by
  $\mathtt{ID}$ and preserves punctuation.
- $\pi_{\Gamma}(\sigma)$ replaces each context-bound occurrence $x$ by the
  type slot $\langle\tau\rangle$ when $\Gamma(x)=\tau$, while preserving API
  names and punctuation.

Both projections extend homomorphically over the term. For example:

$$
\Gamma\vdash \texttt{x}:X,\texttt{y}:Y
\quad\Longrightarrow\quad
\begin{aligned}
\pi_{\Sigma}(\texttt{x.f(y)}) &= \texttt{ID . ID ( ID )},\\
\pi_{\Gamma}(\texttt{x.f(y)}) &= \langle X\rangle \texttt{ . f ( }\langle Y\rangle \texttt{ )}.
\end{aligned}
$$

Thus $\sigma$ matches the surface fragment exactly when
$\pi_{\Sigma}(\sigma)\in\mathcal{L}(G_0)$. Type-directed identifier equivalence is simply equality under $\pi_{\Gamma}$:

$$
\sigma\cong_{\Gamma}\sigma'
\quad\stackrel{\mathrm{def}}{\Longleftrightarrow}\quad
\pi_{\Gamma}(\sigma)
=\pi_{\Gamma}(\sigma').
$$

Different names of the same type therefore match automatically, one or many occurrences at a time, while fixed API names remain significant.

For a library $\texttt{api}$, let $\Delta_{\texttt{api}}$ be its typing environment and $G_{\texttt{api}}$ its generated grammar. Write $\Delta_{\texttt{api}};\Gamma\vdash\sigma:\tau$ for typing against both the library and the ambient context, and $\langle\tau\rangle$ for the grammar nonterminal representing type $\tau$. The soundness-and-completeness target is:

$$
\forall \sigma,\tau.\quad
\pi_{\Sigma}(\sigma)\in\mathcal{L}(G_0)
\quad\Longrightarrow\quad
\left(
  \Delta_{\texttt{api}};\Gamma\vdash\sigma:\tau
  \quad\Longleftrightarrow\quad
  \langle\tau\rangle
    \Rightarrow^{*}_{G_{\texttt{api}}}
    \pi_{\Gamma}(\sigma)
\right).
$$

The right-to-left direction is **soundness**; the left-to-right direction is **completeness**. The image of $\pi_{\Gamma}$ may retain type nonterminals, so it is a sentential form rather than, in general, a terminal word in $\mathcal{L}(G_{\texttt{api}})$. Ordinary language membership requires continuing the derivation until every remaining type slot has been expanded.

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
