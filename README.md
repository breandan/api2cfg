# api2cfg

`api2cfg` generates a context-free grammar of type-safe expressions from a library API.

## Motivation

`api2cfg` aims to be sound and complete with respect to the surface grammar $G_0$:

```text
S    -> ID | ID ( ) | ID ( ARGS ) | S . S
ARGS -> S | S , S | S , S , S
```

Here, $\mathtt{ID}$ may be any identifier. The fragment contains names, calls with zero through three arguments, and arbitrary member chaining.

Let $e$ be a surface expression. The surface projection $\pi_{\Sigma}(e)$ replaces every identifier by $\mathtt{ID}$ while preserving punctuation. Thus $e$ belongs to the supported fragment exactly when

$$
\pi_{\Sigma}(e)\in\mathcal{L}(G_0).
$$

An identifier may have more than one static type. For each identifier $x$, the ambient environment $\Gamma(x)$ is a finite set of complete types. In particular, a callable type keeps its parameter and result types together as one signature. A variable and function both named $\texttt{abc}$ can therefore be represented by

$$
\Gamma(\texttt{abc})=\{\mathtt{Int},\,() \to \mathtt{Int}\}.
$$

Likewise, if $\texttt{x}:C$ is also known statically to satisfy contracts $I_1$ and $I_2$, then $\Gamma(\texttt{x})$ may contain all three types. For each legal reading of $e$, replace every context-bound occurrence $x:\tau$ by the type slot $\langle\tau\rangle$, while preserving fixed API names and punctuation. We write $\Pi_{\Gamma}(e)$ for the set of forms obtained from all such readings. Choices are made per occurrence, but only combinations permitted by the source language's whole-term name and overload resolution are included. Both projections preserve the expression's structure; they change only its identifiers. For example, if $\Gamma(\texttt{x})=\{X\}$ and $\Gamma(\texttt{y})=\{Y\}$, then

$$
\begin{aligned}
\pi_{\Sigma}(\texttt{x.f(y)}) &= \texttt{ID . ID ( ID )},\\
\Pi_{\Gamma}(\texttt{x.f(y)})
  &=\{\langle X\rangle \texttt{ . f ( }\langle Y\rangle \texttt{ )}\}.
\end{aligned}
$$

Different context names with a common admissible type therefore produce the same type slot, while fixed API names remain significant.

For a library typing environment $\Delta_{\texttt{api}}$, let
$G_{\texttt{api}}$ be its generated grammar. Write $\Delta_{\texttt{api}};\Gamma\vdash e:\tau$ for ordinary typing against the library and ambient environments, and write $\langle\tau\rangle$ for the grammar nonterminal representing type $\tau$. The correctness target is

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

The right-to-left direction is **soundness**; the left-to-right direction is **completeness**. The claim is relative to the source language's legal readings; the CFG is not itself a name or overload resolver. A projected form $\varphi$ may still contain type nonterminals, so it is a sentential form rather than necessarily a terminal word. When metadata is incomplete, a backend preserves soundness by omitting uncertain productions rather than inventing types.

## Backends

There are three backend families:

| Backend | API source | Example |
| --- | --- | --- |
| JVM reflection | Classes loaded on the runtime classpath | `./gradlew run --args='java.util'` |
| ClassGraph | JVM bytecode and generic-signature metadata | `./gradlew runClassGraph --args='java.util.function'` |
| Python | Runtime reflection or static `.pyi` files | `python3 main.py numpy`<br>`python3 main_pyi.py --alias=torch torch` |

The JVM reflection and Python reflection backends also support `--parameterized`. Every backend supports `--cnf`.

Generated grammars are written under `gen/` as `.cfg` files, or `.cnf` files when Chomsky normalization is requested.
