# H3.c documentation

Four files, one reading order. Implementation lives in [`h3c_repo/`](../h3c_repo/).

1. **[Concept primer](h3c_concepts_primer.md)** — first principles: tensors, transformers, attention, RoPE, latents, diffusion / Euler, DiT. Read this until “two networks, one Euler loop” is clear.
2. **[Architecture](h3c_architecture_and_theory.md) Part I** — how this repository runs MiniMax-H3, in pipeline order (walkthrough first, catalogs in appendices).
3. **[Formulas](h3c_formulas_and_concepts.md)** — exact equations (`#eq-…` anchors), tiny numeric sketches, mix-up warnings.
4. **[Architecture](h3c_architecture_and_theory.md#part-ii--mapping-toward-qwen38) Part II** — optional mapping toward a future Qwen3.8-27B Mac runtime. Skip it if you only want to understand h3.c.

| File | Role |
| --- | --- |
| `h3c_concepts_primer.md` | Teach the ideas |
| `h3c_architecture_and_theory.md` | Source-grounded wiring of this repo |
| `h3c_formulas_and_concepts.md` | Math dictionary |

Cursor and GitHub render math with dollar signs (one pair inline, two on their own lines for display).
