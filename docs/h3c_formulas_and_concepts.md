# h3.c Formulas and Technical Concepts

Companion to the [Concept primer](h3c_concepts_primer.md) (intuition) and [Architecture](h3c_architecture_and_theory.md) (what the code does). This file is the **equation dictionary**: every display equation has an anchor `#eq-…`. Architecture and the primer link here rather than leaving raw TeX inline.

**How to use an entry.** **Intuition** is the idea in words (the primer has the full chapter). **Definition** is the formula. **Tiny example** is a one-row sketch where it helps. **In h3.c** is the kernel or function. **Common mix-up** is the mistake this repo invites.

**Math preview.** Cursor / VS Code Markdown preview and GitHub render math when it is wrapped in dollar signs: one pair for inline, two on their own lines for display (blank line before and after). They do **not** compile LaTeX `\(...\)` or `\[...\]` delimiters.

Implementation lives under `h3c_repo/`.

---

## Contents

- [Notation](#notation)
- [Embeddings](#embeddings)
- [RMSNorm](#rmsnorm)
- [LayerNorm](#layernorm)
- [Head RMSNorm](#head-rmsnorm)
- [Group normalization](#group-normalization)
- [Residual connections and pre-norm](#residual-connections-and-pre-norm)
- [SiLU](#silu)
- [SwiGLU](#swiglu)
- [GeGLU](#geglu)
- [Snake](#snake)
- [GEMM](#gemm)
- [Qwen transformer layer](#qwen-transformer-layer)
- [Softmax](#softmax)
- [Scaled dot-product attention](#scaled-dot-product-attention)
- [Causal masking](#causal-masking)
- [Grouped-query attention](#grouped-query-attention)
- [Attention scale](#attention-scale)
- [Rotary position embeddings](#rotary-position-embeddings)
- [Multimodal RoPE](#multimodal-rope)
- [DiT 3D RoPE](#dit-3d-rope)
- [Partial rotary](#partial-rotary)
- [AdaLN-Zero](#adaln-zero)
- [DiT block](#dit-block)
- [Timestep embedding](#timestep-embedding)
- [Diffusion, flow matching, and Euler](#diffusion-flow-matching-and-euler)
- [RES sampler](#res-sampler)
- [Language-model head](#language-model-head)
- [KV cache](#kv-cache)
- [Symmetric int8 quantization](#symmetric-int8-quantization)
- [Grouped activation scales](#grouped-activation-scales)
- [bfloat16](#bfloat16)
- [Complexity notation](#complexity-notation)
- [VAE and latents](#vae-and-latents)
- [Patchify](#patchify)
- [Byte-pair encoding](#byte-pair-encoding)
- [Safetensors](#safetensors)
- [Unified memory](#unified-memory)
- [Roofline and memory bandwidth](#roofline-and-memory-bandwidth)
- [Morton order](#morton-order)
- [Token reduction](#token-reduction)
- [Core reuse and velocity reuse](#core-reuse-and-velocity-reuse)
- [Gated DeltaNet](#gated-deltanet)
- [YaRN](#yarn)
- [Token sampling](#token-sampling)
- [Multi-token prediction](#multi-token-prediction)
- [Equation index](#equation-index)

---

## Notation

| Symbol | Meaning in this document |
| --- | --- |
| $x$, $h$ | activation / hidden row, typically width $d$ |
| $d$, $d_{\mathrm{model}}$ | hidden width (Qwen $5120$, DiT $5376$) |
| $d_{\mathrm{head}}$ | attention head width ($128$ in H3 Qwen/DiT) |
| $T$ | sequence length in tokens (or frames, when stated) |
| $S$ | packed DiT sequence length (text + conditions + audio + video) |
| $n_q$, $n_{\mathrm{kv}}$ | query heads and KV heads |
| $\sigma$ | **overloaded:** (1) SiLU, or (2) diffusion noise level — context decides |
| $\varepsilon$ | RMS/LayerNorm stabilizer |
| $\theta$ | RoPE base frequency |
| $W$ | learned matrix; $w$ a learned per-channel vector |
| $\odot$ | elementwise (Hadamard) product |
| $\leftarrow$ | in-place residual update |

Vectors are row-major as in the C tensors: a hidden state of $T$ tokens is $[T, d]$. A linear layer $x W$ maps the last axis.

**Common mix-up.** $\sigma$ in [SiLU](#silu) is the activation $z/(1+e^{-z})$. $\sigma$ in [Euler](#diffusion-flow-matching-and-euler) is the noise level on the schedule. They are unrelated.

Primer: [§2](h3c_concepts_primer.md#2-tensors-and-linear-layers). Architecture: [§5](h3c_architecture_and_theory.md#5-the-qwen-text-tower).

---

## Embeddings

**Intuition.** Discrete token ids are not vectors. An embedding table is a phone book: id $t$ **selects a row**. There is no multiply. Primer: [§3](h3c_concepts_primer.md#3-tokens-and-embeddings).

**Definition.** $E \in \mathbb{R}^{V \times d}$. Token id $t$ gathers row $E[t]$.

<a id="eq-embedding"></a>

$$
h_0[t] = E[\mathrm{id}_t] \quad\text{or}\quad \mathrm{vision}[t]
$$

**Tiny example.** Vocab $V=151936$, $d=5120$. Encoding `"Hi"` as two ids `[1234, 567]` yields `hidden` of shape `[2, 5120]`: row 0 is `E[1234]`, row 1 is `E[567]`.

**Why it exists.** Transformers only add and multiply vectors. $V$ is the vocabulary (H3 Qwen: $151936$).

**In h3.c.** `h3_gpu_embedding_bf16` gathers `[T, 5120]` BF16. Optional vision spans **overwrite** those rows; layers $0$–$2$ then add deepstack residuals.

**Tied LM head (not in h3.c).** If the output projection reuses $E$, logits are $h E^\top$. See [language-model head](#language-model-head).

**Common mix-up.** Embedding is not a linear layer $xW$. A linear layer mixes **channels** of an already-present vector. An embedding **picks** a vector by integer index.

Architecture: [§5.1](h3c_architecture_and_theory.md#51-embeddings).

---

## RMSNorm

**Intuition.** Rows of activations can have wildly different magnitudes after a GEMM. RMSNorm rescales each row to unit root-mean-square, then applies a learned per-channel gain $w$. It does **not** subtract a mean (that is [LayerNorm](#layernorm)). Primer: [§4](h3c_concepts_primer.md#4-the-residual-stream).

**Definition.**

<a id="eq-rmsnorm"></a>

$$
\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \, w_i
$$

**Tiny example.** $x = (3, 0, 4, 0)$, $d=4$, $\varepsilon=0$, $w=\mathbf{1}$. RMS $= \sqrt{(9+0+16+0)/4} = \sqrt{6.25} = 2.5$. Output $= (1.2, 0, 1.6, 0)$.

**Why RMS, not LayerNorm.** Mean-centering is extra bandwidth for little gain in many LLMs (Zhang & Sennrich, 2019). The expensive part is the reduction $\sum x_j^2$ along the hidden axis.

**Stabilizer.** Qwen uses $\varepsilon = 10^{-6}$. DiT / refiner use $\varepsilon = 10^{-5}$. Too small and a near-zero row blows up; too large and the scale is biased.

**In h3.c.** `h3_rms_norm_bf16` / `h3_rms_norm_f32`: one threadgroup per row, shared-memory reduction.

**Common mix-up.** RMSNorm is not “batch norm.” There is no batch axis in the reduction — only the hidden width of **this row**.

Architecture: [§5.2](h3c_architecture_and_theory.md#52-rmsnorm-and-head-rmsnorm).

---

## LayerNorm

**Intuition.** Same job as RMSNorm, plus **centering**: subtract the mean so the row has mean zero before scaling. Slightly more arithmetic (two reductions: mean and variance).

**Definition.**

<a id="eq-layernorm"></a>

$$
\mathrm{LayerNorm}(x)_i = \frac{x_i - \mu}{\sqrt{\frac{1}{d}\sum_j (x_j-\mu)^2 + \varepsilon}} \, w_i + b_i, \quad \mu = \frac{1}{d}\sum_j x_j
$$

Vision in h3.c uses this (`h3_gpu_layer_norm_bf16`) with weight **and** bias. Qwen language layers and the DiT use RMSNorm instead.

**Common mix-up.** “Everything is LayerNorm” is false here. Only the vision tower (and VAE group-norm, which is related) center.

---

## Head RMSNorm

**Intuition.** After $Q$ and $K$ are projected, each **head** is a $128$-vector. Normalizing the concatenated $64\times 128$ blob would let one loud head dominate. Per-head RMS equalizes magnitude **inside** each head before RoPE and dots.

**Definition.** Qwen3-family models RMS-normalize **each attention head** of $Q$ and $K$ over $d_{\mathrm{head}}$ only.

<a id="eq-head-rmsnorm"></a>

$$
\mathrm{HeadRMS}(Q)_{t,h,:} = \mathrm{RMSNorm}_{d_{\mathrm{head}}}(Q_{t,h,:})
$$

**Tiny example.** One token, two heads of dim 2: $Q = [(10, 0),\ (0, 1)]$. Head RMS (ignore $\varepsilon$, $w=1$) maps head 0 to $(1, 0)$ and leaves head 1 at scale $\sim 1$. Dots are no longer dominated by head 0’s $10$.

**Why.** Per-head RMS keeps RoPE and dots from being dominated by a few heads’ magnitude. It is applied **after** the Q/K projections and **before** RoPE in H3’s `encode_layer`.

**In h3.c.** `h3_gpu_head_rms_norm_bf16`. DiT QKV kernels fold a related RMS into the RoPE kernel rather than a separate Qwen-style head RMS on the language tower.

**Common mix-up.** This is not the same as the layer’s input RMSNorm. There are two RMS stages in a Qwen layer: residual-stream RMS, then head RMS on Q/K.

Architecture: [§5.2](h3c_architecture_and_theory.md#52-rmsnorm-and-head-rmsnorm).

---

## Group normalization

**Intuition.** Used in the **video VAE**, not the transformers. Channels are split into $G$ groups; each group is normalized like LayerNorm over $(T,H,W)$ spatial-temporal extents — a compromise between per-channel instance norm and full LayerNorm.

**Definition.**

<a id="eq-groupnorm"></a>

$$
\mathrm{GN}(x)_{n,c} = \frac{x_{n,c} - \mu_{g(c)}}{\sqrt{\sigma^2_{g(c)} + \varepsilon}} \, w_c + b_c
$$

H3 fuses GN+SiLU in `h3_vae_encoder_group_norm_silu_f32`. Irrelevant to a Qwen3.8 text runtime.

Architecture: [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io).

---

## Residual connections and pre-norm

**Intuition.** A residual block predicts a **delta** and adds it back. The skip $x\to x$ is the identity path; $F$ is the learned update. **Pre-norm** applies the norm *before* $F$, which is what both H3 networks do. Primer: [§4](h3c_concepts_primer.md#4-the-residual-stream).

Ungated residual (Qwen, refiner, vision):

<a id="eq-ungated-residual"></a>

$$
x \leftarrow x + F(x)
$$

Gated residual (DiT AdaLN-Zero):

<a id="eq-gated-residual"></a>

$$
x \leftarrow x + F(x) \odot g(t, \mathrm{modality})
$$

$F$ is attention or MLP. $g$ is a learned per-channel gate that depends on timestep and modality (visual / text / audio).

**Tiny example.** $x=(1,1)$, $F(x)=(0.5,-0.2)$, ungated → $(1.5, 0.8)$. If $g=(0,1)$, gated → $(1, 0.8)$: channel 0 of $F$ is fully suppressed.

**Why pre-norm.** Gradients through the skip $x \to x$ stay well-scaled even when $F$ is large. Post-norm (norm after the add) is harder to train at this depth.

**Why gates on DiT.** AdaLN-Zero initializes $g \approx 0$ so each block starts as the identity; the network learns how much of $F$ to let through at each noise level. At inference the gates are ordinary tensors looked up from the schedule.

**Common mix-up.** Gating is **not** dropout and **not** a causal mask. It is a per-channel multiplier on the residual branch, looked up from $\sigma$ and modality.

Architecture: [§5.3](h3c_architecture_and_theory.md#53-qwen-layer-exact-order), [§6.3](h3c_architecture_and_theory.md#63-run_block).

---

## SiLU

**Intuition.** A smooth nonlinearity: roughly ReLU for large positive $z$, a soft negative dip near $0$. Cheap (one `exp`). Also called Swish.

**Definition.** Sigmoid Linear Unit:

<a id="eq-silu"></a>

$$
\sigma(z) = \frac{z}{1 + e^{-z}} = z \cdot \mathrm{sigmoid}(z)
$$

**Tiny example.** $\sigma(0)=0$, $\sigma(2)\approx 1.76$, $\sigma(-2)\approx -0.24$.

Smooth, non-monotonic near $0$. H3’s `h3_silu_mul_bf16` uses exactly $z / (1+e^{-z})$.

**Common mix-up.** Do not confuse this $\sigma$ with diffusion noise level $\sigma$ in [Euler](#diffusion-flow-matching-and-euler).

---

## SwiGLU

**Intuition.** A **gated** MLP: one projection is passed through SiLU and **multiplies** a second projection; a third projection writes back to $d_{\mathrm{model}}$. The gate can suppress channels per token. More expressivity than a plain ReLU MLP without widening $W_D$. Primer: [§4](h3c_concepts_primer.md#4-the-residual-stream) (MLP as per-token $F$).

**Definition.**

<a id="eq-swiglu"></a>

$$
\mathrm{SwiGLU}(x) = \sigma(x W_G) \odot (x W_U)
$$

Full block:

$$
\mathrm{MLP}(x) = \bigl(\sigma(x W_G) \odot (x W_U)\bigr) W_D
$$

**Tiny example.** One token, $d=2$, intermediate $2$. If $\sigma(xW_G)=(0.9, 0.1)$ and $xW_U=(4, 5)$, SwiGLU $=(3.6, 0.5)$ before $W_D$. Channel 1 of the up-projection is mostly shut.

**Shapes in this repo.**

| Network | $d_{\mathrm{model}}$ | intermediate (up/gate) |
| --- | --- | --- |
| H3 Qwen | $5120$ | $25600$ (separate `gate_proj` / `up_proj`) |
| H3 DiT | $5376$ | $14336$ (fused `fc1` of width $2 \times 14336$) |
| Qwen3.8-27B (public spec) | $5120$ | $17408$ |

**In h3.c.** Qwen: three linears + `h3_gpu_silu_mul_bf16`. DiT: fused `h3_gpu_swiglu_bf16` or MPSGraph / int8 TensorOps MLP.

**Common mix-up.** The “2×” on DiT `fc1` is **gate concatenated with up**, not a wider residual stream. After SwiGLU you are back to width $14336$ before $W_D$ maps to $5376$.

Architecture: [§5.4](h3c_architecture_and_theory.md#54-swiglu-mlp).

---

## GeGLU

Same idea as SwiGLU with GELU instead of SiLU. Present as `h3_geglu_f32` for audio pieces, not the Qwen/DiT core.

<a id="eq-geglu"></a>

$$
\mathrm{GeGLU}(x) = \mathrm{GELU}(x W_G) \odot (x W_U)
$$

**Intuition.** GELU is a smoother ReLU (Gaussian-error / tanh approximation in the shader). Same gate×up pattern.

Architecture: [Appendix B](h3c_architecture_and_theory.md#appendix-b--kernel-catalog-83).

---

## Snake

**Intuition.** Periodic activation used in neural audio (BigVGAN-style). The $\sin^2$ term injects a learnable oscillation so the vocoder can represent harmonics; $\alpha$ is a learned frequency.

**Definition.**

<a id="eq-snake"></a>

$$
\mathrm{Snake}(x) = x + \frac{1}{\alpha} \sin^2(\alpha x)
$$

`h3_snake1d_f32` and fused `h3_alias_free_snake_f32` implement this for the audio VAE / vocoder path.

Architecture: [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io).

---

## GEMM

**Intuition.** A linear layer on a token batch is a matrix multiply. Tokens do not mix: each row of $X$ is independently mapped by $W$. Mixing is attention’s job. Primer: [§2](h3c_concepts_primer.md#2-tensors-and-linear-layers).

**Definition.** With row-major activations $X \in \mathbb{R}^{T \times K}$ and weight $W \in \mathbb{R}^{N \times K}$ stored as “output rows”:

<a id="eq-gemm"></a>

$$
Y = X W^\top, \qquad Y \in \mathbb{R}^{T \times N}
$$

Accumulate in FP32 even when $X,W,Y$ are BF16 or int8. That is why H3’s portable kernels tile in `float` and only round at the store.

**Tiny example.** $T=1$, $K=2$, $N=2$, $X=(1, 0)$, $W = \begin{pmatrix} 2 & 3 \\ 4 & 5 \end{pmatrix}$ stored as output rows. $Y = X W^\top = (2, 4)$.

**Int8.** See [symmetric int8 quantization](#symmetric-int8-quantization): the integer product is rescaled by per-row (or grouped) factors.

Architecture: [§9.3](h3c_architecture_and_theory.md#93-mapping-a-neural-op-onto-hardware).

---

## Qwen transformer layer

**Intuition.** One pre-norm decoder block: mix tokens with causal GQA, then a SwiGLU MLP, each behind RMSNorm and an ungated residual. H3 runs this **once** over the full prompt. Primer: [§4](h3c_concepts_primer.md#4-the-residual-stream)–[§5](h3c_concepts_primer.md#5-attention).

Exact `encode_layer` order in `h3_text_encoder.c` (lines 409–465):

<a id="eq-qwen-layer"></a>

$$
\begin{aligned}
n &\leftarrow \mathrm{RMSNorm}(h, W_{\mathrm{in}}) \\
Q &\leftarrow n W_Q,\quad K \leftarrow n W_K,\quad V \leftarrow n W_V \\
Q &\leftarrow \mathrm{HeadRMS}(Q),\quad K \leftarrow \mathrm{HeadRMS}(K) \\
Q,K &\leftarrow \mathrm{RoPE}(Q,K) \\
a &\leftarrow \mathrm{GQA}_{\mathrm{causal}}(Q,K,V)\, W_O \\
h &\leftarrow h + a \\
n &\leftarrow \mathrm{RMSNorm}(h, W_{\mathrm{post}}) \\
h &\leftarrow h + W_D\bigl(\sigma(n W_G)\odot (n W_U)\bigr)
\end{aligned}
$$

This is a **pre-norm decoder block** with **ungated** residuals and **no KV cache**.

H3 constants: $d=5120$, $n_q=64$, $n_{\mathrm{kv}}=8$, $d_{\mathrm{head}}=128$, $50$ layers, RoPE $\theta = 5 \times 10^6$.

**Common mix-up.** This looks like an LLM layer because it is one. H3 still does not **decode** tokens with it. After layer 49 the hidden state is read back and fed to the DiT.

Architecture: [§5.3](h3c_architecture_and_theory.md#53-qwen-layer-exact-order).

---

## Softmax

**Intuition.** Turns a vector of scores into a probability vector: nonnegative, sums to 1. The largest score gets most of the mass; the rest is never exactly zero unless you mask.

**Definition.**

<a id="eq-softmax"></a>

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}
$$

**Tiny example.** $z=(1, 3, 0)$. Subtract max $3$: $(-2, 0, -3)$. Exp $\approx (0.135, 1, 0.050)$. Sum $\approx 1.185$. Weights $\approx (0.114, 0.844, 0.042)$.

The $\max$ subtract is only for numerical stability (prevents `exp` overflow); it does not change the result.

**Precision.** Attention softmax is computed in FP32 in `h3_gqa_causal_bf16` even though Q/K/V are BF16. Tiny errors in the exp-sum become wrong weights after $50$ layers.

**Common mix-up.** Softmax in attention is **over keys for one query**, not over the vocabulary. Vocabulary softmax would be an [LM head](#language-model-head), which H3 does not have.

Architecture: [§5.5](h3c_architecture_and_theory.md#55-causal-gqa).

---

## Scaled dot-product attention

**Intuition.** For each query, score every key, softmax, and read a weighted sum of values. That is the only token-mixing step in a standard block. Primer: [§5](h3c_concepts_primer.md#5-attention).

**Definition.** Classic Vaswani attention:

<a id="eq-sdpa"></a>

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_{\mathrm{head}}}}\right) V
$$

For one query row $q$ and keys $K \in \mathbb{R}^{L \times d_{\mathrm{head}}}$:

1. scores $s_\ell = q \cdot k_\ell / \sqrt{d_{\mathrm{head}}}$
2. weights $\alpha = \mathrm{softmax}(s)$
3. output $\sum_\ell \alpha_\ell v_\ell$

**Tiny example.** $d_{\mathrm{head}}=1$, $q=2$, keys $(1, 2)$, values $(10, 0)$, scale $1$. Scores $(2, 4)$. Softmax $\approx (0.12, 0.88)$. Output $\approx 1.2$.

**DiT.** Bidirectional: $L = S$ for every query (no causal mask), $Q=K=V$, $56$ heads, $d_{\mathrm{head}}=128$. Implemented by **MPSGraph** `scaledDotProductAttentionWithQueryTensor`, not a custom softmax kernel.

**Score memory.** Naively $\Theta(n_{\mathrm{heads}} S^2)$. Library Flash-style tiling can avoid materializing the full $S \times S$ matrix; you still pay $\Theta(S^2)$ compute per head.

**Common mix-up.** DiT SDPA is **not** GQA and **not** causal. Qwen GQA is the opposite on both counts.

Architecture: [§6.5](h3c_architecture_and_theory.md#65-bidirectional-sdpa).

---

## Causal masking

**Intuition.** A language-model query at position $t$ may attend only to keys $0..t$ (it must not see the future). Primer: [§5](h3c_concepts_primer.md#5-attention).

**Definition.**

<a id="eq-causal"></a>

$$
\mathrm{mask}_{t,\ell} =
\begin{cases}
0 & \ell \le t \\
-\infty & \ell > t
\end{cases}
\qquad \Rightarrow \qquad \mathrm{key\_count}(t) = t+1
$$

**Tiny example.** Sequence length 3. Query 0 dots with 1 key; query 1 with 2 keys; query 2 with 3 keys. H3’s GQA kernel does not add $-\infty$; it **shortens the loop** to `key_count = query_row + 1`.

DiT attention is **not** causal: every packed token (text, audio, video) can attend to every other.

**Common mix-up.** Causal masking is **not** AdaLN gating and **not** `--token-reduction`. Those change residual scale or sequence length, not the future-token triangle.

Architecture: [§5.5](h3c_architecture_and_theory.md#55-causal-gqa).

---

## Grouped-query attention

**Intuition.** Multi-head attention (MHA) has $n_q = n_{\mathrm{kv}}$. **GQA** shares each K/V head across a group of query heads, cutting KV storage (and, in an LLM, cache size) with a modest quality tradeoff. Primer: [§5](h3c_concepts_primer.md#5-attention).

**Definition.**

<a id="eq-gqa-index"></a>

$$
\mathrm{kv\_head}(h_q) = \Big\lfloor h_q \mathbin{/} (n_q / n_{\mathrm{kv}}) \Big\rfloor
$$

**Tiny example.** H3 Qwen: $n_q=64$, $n_{\mathrm{kv}}=8$, group size $8$. Query heads $0..7$ share KV head $0$; $8..15$ share KV head $1$; …; $56..63$ share KV head $7$. KV tensors are $[T, 8, 128]$ instead of $[T, 64, 128]$.

**In h3.c.** Custom kernel `h3_gqa_causal_bf16`. Optional `H3_MPS_GQA` uses MPSGraph instead. There is still **no cache**: every encode recomputes K and V for all $T$ tokens.

**Common mix-up.** DiT has $56$ heads and **does not** use GQA. Do not apply the $64/8$ map to `run_block`.

Architecture: [§5.5](h3c_architecture_and_theory.md#55-causal-gqa).

---

## Attention scale

**Intuition.** A dot product of two $d$-vectors has variance proportional to $d$ if coordinates are order-1. Softmax then saturates (one-hot weights). Dividing by $\sqrt{d}$ keeps scores in a healthy range.

**Definition.**

<a id="eq-attention-scale"></a>

$$
\mathrm{scale} = \frac{1}{\sqrt{d_{\mathrm{head}}}} = \frac{1}{\sqrt{128}}
$$

**Tiny example.** $\sqrt{128}=8\sqrt{2}\approx 11.31$, so scale $\approx 0.0884$. A raw dot of $20$ becomes $\approx 1.77$ before softmax.

**H3 GQA detail.** The scale is baked into **Q before the dots** to match MLX fused-SDPA order. MPSGraph DiT SDPA takes the scale as an argument instead.

**Common mix-up.** This is not temperature in [token sampling](#token-sampling). Temperature divides **logits over a vocab**. Attention scale divides **QK dots**.

Architecture: [§5.5](h3c_architecture_and_theory.md#55-causal-gqa).

---

## Rotary position embeddings

**Intuition.** Without position, attention is a bag of tokens. RoPE (Su et al.) encodes **relative** position by rotating pairs of dimensions of $Q$ and $K$. After rotation, $q_m^\top k_n$ depends on $m-n$. Primer: [§6](h3c_concepts_primer.md#6-rotary-position-embeddings).

**Definition.** Frequencies for pair $i = 0..d_{\mathrm{head}}/2 - 1$:

<a id="eq-rope-freq"></a>

$$
\omega_i = \theta^{-2i / d_{\mathrm{head}}}, \qquad \mathrm{inv\_freq}_i = \omega_i
$$

H3 Qwen: $\theta = 5 \times 10^6$. Qwen3.8 public spec: $\theta = 10^7$.

Rotation of a pair at position $p$:

<a id="eq-rope-rotate"></a>

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
=
\begin{pmatrix} \cos(p\omega_i) & -\sin(p\omega_i) \\ \sin(p\omega_i) & \cos(p\omega_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

Equivalent view: treat $(x_{2i}, x_{2i+1})$ as a complex number and multiply by $e^{i p \omega_i}$.

**Tiny example.** Pair $(1, 0)$, $p\omega = \pi/2$. Rotate → $(0, 1)$. Same pair at $p\omega = \pi$ → $(-1, 0)$. The **relative** angle between two positions is $(p-p')\omega$.

**In h3.c.** Qwen: `h3_gpu_rope_text_bf16` with **F32** cos/sin tables. DiT: fused into QKV+RoPE kernels; `prepare_rope` builds tables.

**Common mix-up.** RoPE is applied to **Q and K only**, not V. Adding position embeddings to $x$ *before* the layer is a different method (absolute / learned PE).

Architecture: [§5.6](h3c_architecture_and_theory.md#56-text-rope-and-mrope).

---

## Multimodal RoPE

**Intuition.** Images and video have **spatial** neighbors, not just a 1D token index. Qwen-VL **mRoPE** uses **three** position axes (temporal, height, width). Primer: [§6](h3c_concepts_primer.md#6-rotary-position-embeddings).

`position_ids` is stored axis-major `[3, tokens]`. For a feature index $i$ in the rotary pairs, H3’s text encoder comment says axes **cycle** when $i < 60$:

<a id="eq-mrope"></a>

$$
p^{(i)} = \mathrm{position\_ids}\bigl[i \bmod 3,\; t\bigr]
$$

Text-only prompts still use $p = $ token index on all axes that the kernel reads. Vision/video tokens get 2D/3D coordinates so spatial neighbors have close rotations.

**Vision RoPE** is a separate kernel (`h3_gpu_vision_qkv_rope_bf16`) with 2D/temporal positions and no Q/K RMS in that kernel.

**Common mix-up.** mRoPE is the **Qwen text/vision** presentation. DiT uses a different [3D RoPE](#dit-3d-rope) on packed patches (`prepare_rope`).

Architecture: [§5.6](h3c_architecture_and_theory.md#56-text-rope-and-mrope).

---

## DiT 3D RoPE

**Intuition.** Video patches live on a $(T, H, W)$ grid. Rotating with three coordinates lets attention treat nearby-in-time and nearby-in-space patches as nearby in RoPE angle.

**Definition.** `prepare_rope` uses `rope.inv_freq` of length $16$ and axes $(t,\, h\cdot s,\, w\cdot s)$.

<a id="eq-rope-3d"></a>

$$
p = (t,\; h \cdot s,\; w \cdot s)
$$

`ROPE_HALF = 48`: only the **first $48$ of $128$** head dims rotate; the last $80$ stay absolute features ([partial rotary](#partial-rotary)).

At native $256\times 256$, spatial coordinates are halved unless `--use-reference-rope`, to avoid a lattice artifact without adding tokens.

**Common mix-up.** `ROPE_HALF=48` is **not** $d_{\mathrm{head}}/2$ (that would be 64, which is Qwen’s `TEXT_ROPE_HALF`). DiT rotates a $48$-dim prefix.

Architecture: [§6.6](h3c_architecture_and_theory.md#66-dit-3d-rope).

---

## Partial rotary

**Intuition.** Only a prefix of $d_{\mathrm{head}}$ is rotated; the tail is not, so some channels stay pure content (not mixed with position).

**Definition.**

<a id="eq-partial-rotary"></a>

$$
\mathrm{RoPE}(x) = \bigl[\mathrm{rotate}(x_{0:r}),\; x_{r:d_{\mathrm{head}}}\bigr]
$$

| Model | $d_{\mathrm{head}}$ | rotated dims | factor |
| --- | --- | --- | --- |
| H3 DiT | $128$ | $48$ | $48/128 = 0.375$ |
| Qwen3.8 Gated Attention (public) | $256$ | $64$ | `partial_rotary_factor` $0.25$ |

The unrotated tail can carry content that should not be mixed with position.

Architecture: [§6.6](h3c_architecture_and_theory.md#66-dit-3d-rope).

---

## AdaLN-Zero

**Intuition.** The **same** DiT weights must denoise at every noise level. A small MLP of the timestep produces per-channel **shift, scale, and gate**, so each block sees a $\sigma$-dependent affine of the residual stream. Primer: [§10](h3c_concepts_primer.md#10-dit-adaln-zero-and-the-packed-sequence).

**Definition.** Adaptive LayerNorm (DiT, Peebles & Xie). Given RMS-normalized $x$ and modulation vectors:

<a id="eq-adaln"></a>

$$
\tilde{x} = \mathrm{RMSNorm}(x) \odot (1 + \mathrm{scale}) + \mathrm{shift}
$$

H3 stores **six** channels per modality (visual / text / audio):

| Slot | Role |
| --- | --- |
| 0 | attention shift |
| 1 | attention scale |
| 2 | attention gate |
| 3 | MLP shift |
| 4 | MLP scale |
| 5 | MLP gate |

**Tiny example.** After RMSNorm a channel is $0.5$, scale $=0.2$, shift $=1$. AdaLN value $= 0.5\cdot 1.2 + 1 = 1.6$. If the attention gate for that channel is $0$, the attention residual adds nothing there.

Row index into the table is `time_row * 3 + tag`. Precomputed by `h3_dit_schedule_precompute` so the timestep MLP is not rerun inside `run_block`.

“Zero” refers to **training init**: gates start near $0$, so the residual path dominates until the network learns otherwise.

**Common mix-up.** AdaLN is **not** “add the timestep embedding to $x$.” It is an affine of the **normalized** residual, plus a **gate on the skip branch**. Concatenating $t$ as extra tokens would be a different design.

Architecture: [§6.2](h3c_architecture_and_theory.md#62-adaln-zero-and-the-schedule).

---

## DiT block

**Intuition.** One AdaLN-gated transformer block: bidirectional attention over the **packed** sequence, then SwiGLU, each residual multiplied by a timestep/modality gate. Primer: [§10](h3c_concepts_primer.md#10-dit-adaln-zero-and-the-packed-sequence).

`run_block` (`h3_dit.c` lines 1876–2048):

<a id="eq-dit-block"></a>

$$
\begin{aligned}
\tilde{x} &= \mathrm{RMSNorm}(x)\,(1+\mathrm{scale}) + \mathrm{shift} \\
y &= W_O\,\mathrm{SDPA}(\mathrm{RoPE}(W_{QKV}\tilde{x})) \\
x &\leftarrow x + y \odot \mathrm{gate}_{\mathrm{attn}} \\
\tilde{x} &= \mathrm{AdaLN}_{\mathrm{MLP}}(x) \\
z &= W_2\,\mathrm{SwiGLU}(W_1\tilde{x}) \\
x &\leftarrow x + z \odot \mathrm{gate}_{\mathrm{MLP}}
\end{aligned}
$$

Attention is **bidirectional full SDPA**, $56$ heads, **no GQA**, **no causal mask**. Text, audio, and video tokens sit in one packed sequence (MMDiT-style). Architecture: [§3.4](h3c_architecture_and_theory.md#34-two-transformers), [§6.1](h3c_architecture_and_theory.md#61-packed-sequence).

Production fuses the attention gate with the following MLP AdaLN (`h3_gpu_gate_adaln_bf16`) and, across blocks, the MLP gate with the next attention AdaLN.

**Refiner.** Two DiT-like blocks on text only: full SDPA, **no RoPE**, **no AdaLN**, ungated pre-norm residuals. Run once at load, not every Euler step.

**Common mix-up.** `run_block` is not `encode_layer`. Different width (5376 vs 5120), different attention (full vs causal GQA), gated vs ungated residuals.

Architecture: [§6.3](h3c_architecture_and_theory.md#63-run_block).

---

## Timestep embedding

**Intuition.** Diffusion blocks need a vector that represents the current noise level. Sinusoidal features of $t$ (a scalar derived from $\sigma$) are the same idea as Transformer positional sinusoids, then a small MLP. The MLP output is **not** added to tokens; it is projected into the AdaLN table.

**Definition.** H3 uses $t = 1-\sigma$ (clean $\leftrightarrow$ $t \approx 1$, noisy $\leftrightarrow$ $t \approx 0$ depending on the $\sigma$ grid), sinusoidal features of size $256$, then an MLP.

<a id="eq-sinusoidal"></a>

$$
e_{2i}(t) = \sin\bigl(t \cdot \omega_i\bigr), \qquad e_{2i+1}(t) = \cos\bigl(t \cdot \omega_i\bigr)
$$

Then: Linear $256 \to 5376$ → SiLU → Linear $5376 \to 2688$ F32 → cast BF16 → SiLU → per-block `adaln_proj` into the six-slot table.

Visual conditions use timestep $0.999$; audio conditions use $1.0$ when not near-terminal (schedule row map).

**Common mix-up.** $t=1-\sigma$ is a **convention in this schedule**, not a universal diffusion definition. Always check `h3_dit_schedule.c` `prepare_rows` before porting formulas from a paper.

Architecture: [§6.2](h3c_architecture_and_theory.md#62-adaln-zero-and-the-schedule).

---

## Diffusion, flow matching, and Euler

**Intuition.** H3 is **not** an autoregressive LM. Start from a noisy latent, predict how it should **move** as noise decreases, take a small step, repeat, then VAE-decode. The network’s output is a **velocity field**, not next-token logits. Primer: [§9](h3c_concepts_primer.md#9-diffusion-flow-matching-and-euler) (read that chapter; this entry is the serving equation).

The sampler is an Euler integrator on a decreasing noise schedule $\sigma_0 > \sigma_1 > \cdots \ge 0$. Video shift $12$, audio shift $3$ (`h3_host.h`). Default $20$ steps.

**Definition.**

<a id="eq-euler"></a>

$$
x_{\sigma_{\mathrm{next}}} = x_{\sigma} + (\sigma - \sigma_{\mathrm{next}}) \, v_\theta(x_{\sigma}, \sigma)
$$

**Tiny example.** One latent entry $x=0.5$, $\sigma=1.0$, $\sigma_{\mathrm{next}}=0.8$, $v=-1.25$. Delta $=0.2$, update $x \leftarrow 0.5 + 0.2\cdot(-1.25)=0.25$. Do this for every entry of `[24,T,H,W]` and `[32,2,T]`.

That is exactly `h3_euler_velocity_step`: `sample += delta * velocity` with `delta = sigma - sigma_next`. The GPU kernel `h3_euler_bf16` is the same `fma` plus optional [velocity reuse](#core-reuse-and-velocity-reuse).

**Flow-matching view (THEORY, not trained in this repo).** If $x_\sigma$ interpolates between noise and data, $v_\theta$ approximates $\mathrm{d}x/\mathrm{d}\sigma$. Euler is the first-order ODE step. Training details live in MiniMax/H3 papers.

Three parameterizations you will see in papers (H3 serving uses velocity):

| Name | Predicts | H3 heads |
| --- | --- | --- |
| Noise $\varepsilon$ | the added noise | no |
| $x_0$ | the clean latent | no (RES path uses a denoised form; not default) |
| Velocity | $\mathrm{d}x/\mathrm{d}\sigma$ | **yes** |

**Not logits.** Final DiT heads emit video velocity $[24,T,H,W]$ F32 and audio velocity $[32,2,T]$ F32. There is no vocabulary softmax.

**Common mix-up.** “Sampler” in h3.c is this ODE step, not top-$p$ token sampling. Velocity is not a probability.

Architecture: [§2.6](h3c_architecture_and_theory.md#26-each-euler-step), [§3.1](h3c_architecture_and_theory.md#31-actual-execution-path), [§6.4](h3c_architecture_and_theory.md#64-final-heads-are-velocities).

---

## RES sampler

**Intuition.** A different integrator for a **denoised** parameterization (the network is treated as predicting $x_0$, not velocity). Implemented as `h3_res_step`; **not** the default serving path.

Let $\sigma > \sigma_{\mathrm{next}} \ge 0$. First-order (no previous denoised, or last step):

<a id="eq-res-euler"></a>

$$
\frac{\mathrm{d}x}{\mathrm{d}\sigma} \approx \frac{x - x_0}{\sigma}, \qquad x \leftarrow x + \frac{x - x_0}{\sigma}\,(\sigma_{\mathrm{next}}-\sigma)
$$

Higher-order branch uses $t = -\log\sigma$, $h = t_{\mathrm{next}} - t$, $\varphi_1(h) = (e^{h}-1)/h$ (code: `expm1(value)/value` on $-h$), and mixes current and previous denoised with coefficients $b_1,b_2$. This is a **2nd-order exponential integrator**.

**Common mix-up.** RES is not `--reuse`. `--reuse` extrapolates **velocities** between Euler steps. RES is a different ODE scheme.

Architecture: [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io) (host geometry / `h3_host.c`).

---

## Language-model head

**Intuition.** A causal LM maps the last hidden row to vocabulary logits, then samples a token. **H3 has no LM head.** Primer: [§7](h3c_concepts_primer.md#7-this-is-not-an-autoregressive-language-model), [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii).

**Definition.**

<a id="eq-lm-head"></a>

$$
\mathrm{logits} = h W_E^\top \in \mathbb{R}^{V}
$$

Often $W_E = E$ (tied embeddings). Qwen3.8 needs this plus [token sampling](#token-sampling). $V = 248320$ in the public 3.8 spec versus $151936$ in H3’s Qwen tower.

**Tiny example.** $h$ is `[5120]`, $W_E$ is `[248320, 5120]`, logits are `[248320]`. Argmax is the greedy next token. H3 never builds this tensor.

Architecture: [§6.4](h3c_architecture_and_theory.md#64-final-heads-are-velocities), [Part II §16](h3c_architecture_and_theory.md#16-if-hello-were-qwen38-decode).

---

## KV cache

**Intuition.** Autoregressive decoding of token $t+1$ needs $K_{0:t}$, $V_{0:t}$. Recomputing projections from all past hidden states is $\Theta(t)$ GEMM per new token per layer. Storing K/V makes the extra work **attention against the cache only**. **H3 does not have this.** Primer: [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii).

**Definition.** For GQA, cache bytes (K and V) are about

<a id="eq-kv-cache-size"></a>

$$
2 \times n_{\mathrm{layers}} \times n_{\mathrm{kv}} \times d_{\mathrm{head}} \times T \times \mathrm{sizeof}(\mathrm{dtype})
$$

Text encode is a full-sequence forward; DiT is full bidirectional SDPA every step. Caches that *do* exist (session embeddings, AdaLN tables, SSD slots, …) are listed in [architecture §5.7](h3c_architecture_and_theory.md#57-caches-that-exist).

**Tiny example.** One H3 Qwen layer, $T=2$, BF16: K is `[2, 8, 128]` = $4096$ bytes, V the same, $8$ KiB. After encode, both are **freed**. An LLM would keep them and append a third row.

**Qwen3.8 proposal.** (1) GQA KV for $16$ Gated Attention layers ($24$ Q / $4$ KV, $d_{\mathrm{head}}=256$). (2) A **fixed-size** [DeltaNet](#gated-deltanet) recurrent state for $48$ linear layers — not a growing KV tensor.

Worked numbers (BF16, GQA only, ignoring DeltaNet): $16$ layers $\times$ $4$ KV heads $\times$ $256$ $\times$ $T$ $\times$ $2$ (K and V) $\times$ $2$ bytes $= 65536\, T$ bytes. At $T=262144$ that is $16$ GiB **just** for softmax-attention KV, before activations and weights.

**Common mix-up.** AdaLN tables, SSD slots, and `--core-reuse` residuals are caches, but they are **not** a transformer KV cache.

Architecture: [Part II §17](h3c_architecture_and_theory.md#17-why-a-kv-cache-exists-in-llms).

---

## Symmetric int8 quantization

**Intuition.** Store weights (and, at runtime, activations) as `int8` so GEMMs move half the bytes of BF16. One scale reconstructs the real range. **No zero-point** (symmetric around $0$). Primer: [§11](h3c_concepts_primer.md#11-inference-engineering).

**Definition.** H3 weight quant uses one F32 scale per **output row**, values in $[-127, 127]$.

<a id="eq-absmax"></a>

$$
s = \frac{\max_j |W_{r,j}|}{127}, \qquad q_j = \mathrm{round}\bigl(\mathrm{clip}(W_{r,j}/s,\,-127,\,127)\bigr)
$$

Dequant of a matmul accumulation (kernel comments):

<a id="eq-int8-dequant"></a>

$$
y_{r,c} = \mathrm{i32\_acc}_{r,c} \cdot s^{\mathrm{in}}_r \cdot s^{W}_c
$$

**Tiny example.** Row max $|W|=1.27$, $s=0.01$. Value $0.63$ → $q=\mathrm{round}(63)=63$. Reconstruct $63\times 0.01=0.63$. Integer MMA runs on $q_{\mathrm{in}} \times q_W$; the two scales reconstruct BF16/F32.

$s^{W}_c$ is the weight scale for output channel $c$. $s^{\mathrm{in}}_r$ is the **activation** scale for row $r$ (or a group — next section).

**Why 127 not 128.** `int8` can represent $-128$, but using $[-127,127]$ keeps the grid symmetric so $0$ is exact and absmax scaling is unbiased around zero.

**Common mix-up.** This is **runtime** quant of DiT matrices at load, not an FP8 checkpoint format. SSD streaming **disables** int8.

Architecture: [§10](h3c_architecture_and_theory.md#10-quantization).

---

## Grouped activation scales

**Intuition.** One scale per whole activation row is crude when the row is wide (DiT FC2 $K=14336$). A few huge channels would force a coarse scale on the rest. Groups of $1024$ give each slice its own absmax.

**Definition.**

<a id="eq-grouped-scale"></a>

$$
s^{\mathrm{in}}_{r,g} = \frac{\max_{j \in g} |x_{r,j}|}{127}, \qquad |g| = 1024
$$

`--use-int8-row-fc2` falls back to one scale per row and a full-$K$ TensorOps product (faster, less conservative).

QKV / attention-out can fuse absmax into AdaLN (`h3_gpu_gate_adaln_quantize_int8`) so the extra pass does not reread DRAM.

**Tiny example.** $K=2048$, two groups. Group 0 max $1.27$ → $s=0.01$. Group 1 max $12.7$ → $s=0.1$. A $0.5$ in group 0 quantizes relative to $0.01$; the same $0.5$ in group 1 relative to $0.1$ (fewer distinct levels).

Architecture: [§10](h3c_architecture_and_theory.md#10-quantization).

---

## bfloat16

**Intuition.** Same **range** as FP32 (8-bit exponent), much **coarser** precision (7-bit mantissa). Good for storing activations/weights without overflowing; bad as an accumulator for long sums. Primer: [§2](h3c_concepts_primer.md#2-tensors-and-linear-layers), [§11](h3c_concepts_primer.md#11-inference-engineering).

**Definition.** BF16: $1$ sign bit, **$8$ exponent bits**, **$7$ mantissa bits**.

<a id="eq-bf16"></a>

$$
\mathrm{value} = (-1)^s \cdot 2^{e-127} \cdot (1 + m/128)
$$

(for normal numbers; $e=0$ is subnormal/zero as in IEEE-ish BF16).

**Tiny example.** Unit in the last place around $1.0$ is $2^{-7}\approx 0.0078$ ($\sim 0.8\%$). $1.0+0.004$ may round back to $1.0$ in BF16; FP32 would keep it.

**Consequence.** Range matches FP32, so GEMM accumulators rarely overflow; precision is coarse. H3 therefore:

- accumulates in FP32
- rounds at operation boundaries to match the released compute dtype
- stores tensors as `uint16` / Metal `ushort` bit patterns
- has **no FP16 path**

That is also why late DiT oracles use loose rel-L2: BF16 + attention is not associative.

**Common mix-up.** BF16 is not FP16. FP16 has 5 exponent bits and overflows much more easily; H3 never uses it.

Architecture: [§8.2](h3c_architecture_and_theory.md#82-dtypes-and-layout).

---

## Complexity notation

$\Theta(f)$ means “grows like $f$” (upper and lower bound up to constants).

Full self-attention scores for sequence $T$, $n_q$ heads, head width $d$:

<a id="eq-complexity-attn"></a>

$$
\Theta(T^2 \, n_q \, d_{\mathrm{head}}) \quad \text{compute (naive)}, \qquad \Theta(n_q T^2) \quad \text{score storage (naive)}
$$

H3’s GQA kernel is $\Theta(T^2)$ **per query head** with no cache (full encode). LLM **decode** with a cache is $\Theta(T)$ per new token per head (dots against stored keys), plus $\Theta(1)$ to append the new K/V.

DiT pays $\Theta(S^2)$ **every Euler evaluation**. `--token-reduction` cuts $S$ in middle blocks to shrink that quadratic.

**Tiny example.** Double video tokens in $S$ (keep heads fixed) and SDPA compute $\times 4$. That is why token reduction and internal render resolution exist.

Architecture: [§5.5](h3c_architecture_and_theory.md#55-causal-gqa), [§6.5](h3c_architecture_and_theory.md#65-bidirectional-sdpa).

---

## VAE and latents

**Intuition.** Denoising RGB video in pixel space is attention over millions of locations. A VAE maps pixels/PCM $\leftrightarrow$ a smaller **latent** that the DiT denoises. Primer: [§8](h3c_concepts_primer.md#8-latents-vaes-and-patchify).

H3 video: spatial ratio $16$, causal temporal compression $\lceil T/4 \rceil$, latent channels $24$, patchify $2\times 2$ so tokens are $96$ wide before `video_patch_proj`. Audio: $32$ kHz stereo → latent `[32, 2, T]` with hop $800$.

You do not need VAE math for a Qwen3.8 **text** runtime. For media: encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$; at inference H3 uses a deterministic conv encoder/decoder, not a sampled posterior.

**Common mix-up.** The vision **ViT** encodes reference images into the **Qwen** prompt. The video **VAE** encodes/decodes the **DiT latent**. They are different networks.

Architecture: [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io).

---

## Patchify

**Intuition.** A transformer wants a sequence of vectors. Cut the video latent grid into non-overlapping $2\times 2$ patches and flatten each patch to a token, then linearly project to width $5376$.

**Definition.**

<a id="eq-patchify"></a>

$$
\text{video tokens} = T \cdot \frac{H}{2} \cdot \frac{W}{2}, \qquad \text{patch} \in \mathbb{R}^{2 \cdot 2 \cdot 24} = \mathbb{R}^{96}
$$

**Tiny example.** Latent $T=7$, $H=W=32$ → $7\cdot 16\cdot 16 = 1792$ video tokens, each $96$ wide, then `video_patch_proj` → $5376$. Inverse **unpatchify** after the velocity head.

`--token-reduction` averages pairs of horizontal video tokens in middle DiT blocks (`h3_token_pool_bf16`) and expands the residual back (`h3_token_expand_*`), cutting $S$.

Architecture: [§8.3](h3c_architecture_and_theory.md#83-important-tensors), [§6.1](h3c_architecture_and_theory.md#61-packed-sequence).

---

## Byte-pair encoding

**Intuition.** Map text to integer ids without a neural net. Start from bytes; repeatedly merge the most frequent adjacent pair; encode greedily left-to-right with those merges. Primer: [§3](h3c_concepts_primer.md#3-tokens-and-embeddings).

BPE builds a vocabulary by repeatedly merging the most frequent adjacent pair of symbols, starting from bytes (here: byte-level BPE after ICU + NFC).

Encoding is greedy: left-to-right longest merges from `tokenizer.json`. H3 requires BPE, NFC, **no unk token**, pad id `151643`.

This is not a neural formula; it is a deterministic string $\to$ id map. Wrong merges produce a fluent but **wrong** embedding sequence that no later kernel can undo.

Architecture: [§2.1](h3c_architecture_and_theory.md#21-text-to-token-ids), [Appendix A](h3c_architecture_and_theory.md#appendix-a--file-to-module-inventory).

---

## Safetensors

**Intuition.** Hugging Face on-disk tensor format: a JSON header of names/shapes/offsets, then raw payload bytes. No pickle.

On-disk layout:

```text
uint64 little-endian header_size
JSON { name: { dtype, shape, data_offsets: [begin, end] }, __metadata__? }
payload at file offset 8 + header_size + begin
```

No math beyond “bytes at an offset.” H3 parses with `h3_st_read_header` (max $256$ MiB JSON) and `pread`s payloads in $1$ GiB chunks. mmap/zero-copy is optional later in `h3_gpu_tensor_load_file`, not in the parser.

Architecture: [§8.1](h3c_architecture_and_theory.md#81-checkpoint-layout).

---

## Unified memory

**Intuition.** On Apple Silicon, CPU and GPU share one DRAM. There is little PCIe-style “upload”; the cost is **DRAM bandwidth** and cache coherence. Primer: [§11](h3c_concepts_primer.md#11-inference-engineering).

`MTLResourceStorageModeShared` makes `buffer.contents` CPU-visible.

Implication: large GEMMs are usually [bandwidth-bound](#roofline-and-memory-bandwidth). Int8, fusion, and activation aliasing exist to **move fewer bytes**, not to invent FLOPs.

Architecture: [§9.3](h3c_architecture_and_theory.md#93-mapping-a-neural-op-onto-hardware), [§11](h3c_architecture_and_theory.md#11-memory-management).

---

## Roofline and memory bandwidth

**Intuition.** Arithmetic intensity = FLOPs per byte moved. If intensity is low, you are **bandwidth-bound**: faster ALUs do not help until you move fewer bytes (int8, fusion) or cache better (Morton).

**Definition.** GEMM $Y = X W^\top$ with $T$ large, $K,N$ the inner/output:

<a id="eq-roofline"></a>

$$
I \approx \frac{2 T K N}{2(TK + KN + TN)} \;\text{FLOPs/byte (F32)}
$$

**Tiny example.** FC1 $T=2048$, $K=5376$, $N=28672$. Bytes of $W$ dominate $KN$. Int8 halves those bytes per MAC; TensorOps MMA raises the FLOP ceiling so the roof is even more clearly bandwidth.

Architecture: [§11](h3c_architecture_and_theory.md#11-memory-management).

---

## Morton order

**Intuition.** A Morton (Z-order) curve interleaves $x$ and $y$ bits so adjacent tiles in 2D stay nearby in 1D DRAM. Walking the launch grid that way cuts cache thrash.

H3’s M5 TensorOps GEMMs launch in Morton tile order (`*_morton`, `*_morton4`) on $128\times 64$ / $128\times 128$ tiles. Comments credit Draw Things / ccv.

No closed-form ML formula; it is a memory-layout permutation of the launch grid.

Architecture: [§9.3](h3c_architecture_and_theory.md#93-mapping-a-neural-op-onto-hardware), [Appendix B](h3c_architecture_and_theory.md#appendix-b--kernel-catalog-83).

---

## Token reduction

**Intuition.** Quadratic attention cost tracks $S^2$. Average neighboring **horizontal** video tokens in middle blocks, run attention/MLP on the short sequence, expand the residual back to the full grid. Roughly $S/2$ video tokens in those blocks.

Default block range **4:30** (`h3_dit.c`). Quality tradeoff is documented in the README (aggressive settings can ghost limbs).

**Common mix-up.** This is not GQA and not causal masking. It is an **algorithmic** cut of spatial tokens, optional via `--token-reduction`.

Architecture: [§6.1](h3c_architecture_and_theory.md#61-packed-sequence).

---

## Core reuse and velocity reuse

**Intuition.** Nearby $\sigma$ produce similar DiT internals. Two optional hacks skip work. Neither is a learned sampler. Primer: [§9](h3c_concepts_primer.md#9-diffusion-flow-matching-and-euler).

**`--core-reuse`.** The expensive $50$-block trunk is assumed to change slowly. Cache

$$
\Delta = h_{\mathrm{after}} - h_{\mathrm{before}}
$$

and skip the trunk, applying a cheaper timestep-dependent head.

**`--reuse` (velocity extrapolation).** Skip some DiT forwards. The GPU Euler kernel blends the last two BF16 velocities:

<a id="eq-velocity-reuse"></a>

$$
v = v_{\mathrm{last}} + r\,(v_{\mathrm{last}} - v_{\mathrm{previous}})
$$

**Tiny example.** $v_{\mathrm{last}}=3$, $v_{\mathrm{previous}}=1$, $r=1$ → $v=3+(3-1)=5$ (linear extrapolate). $r=0$ → $v=3$ (hold last). Then $x \leftarrow x + \Delta\sigma \cdot v$.

This is linear extrapolation in step index. Exclusive with `--core-reuse` (README).

**Common mix-up.** `--reuse` is not RES and not AdaLN. It only fakes a DiT **output** (velocity) for skipped steps.

Architecture: [§12.1](h3c_architecture_and_theory.md#121-optimization-inventory).

---

## Gated DeltaNet

**Not implemented in h3.c.** Primer: [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii).

**Intuition.** Softmax attention stores a growing KV cache. Linear attention replaces $QK^\top$ softmax with a **recurrent state** $S_t$ of **fixed size** (independent of $T$). The **delta rule** (Schlag et al.; Gated DeltaNet, Yang et al.) updates a matrix-valued memory with a gated write.

Public Qwen3.8-27B uses $48$ Gated DeltaNet layers + $16$ Gated Attention layers.

**Definition.**

<a id="eq-deltanet"></a>

$$
S_t = S_{t-1}\bigl(I - \beta_t k_t k_t^\top\bigr) + \beta_t v_t k_t^\top
$$

$k_t,v_t$ are per-token keys/values; $\beta_t$ is a learned write strength. Extra **gates** and a short conv (kernel $4$ in the 3.8 spec) modify this rule; the exact Qwen3.8 kernel is defined by the official code, not by this repository.

**Why it matters for a Mac runtime.** State is $O(1)$ in $T$ per layer (unlike [KV cache](#kv-cache)), which is the only plausible way to approach $262$K context on unified memory. It is also the highest-risk new Metal work: recurrence, layout, and numerical parity against Transformers/vLLM.

Architecture: [Part II §18](h3c_architecture_and_theory.md#18-component-mapping), [§19](h3c_architecture_and_theory.md#19-gaps-and-new-challenges).

---

## YaRN

**Intuition.** Yet another RoPE extensioN: interpolate / scale RoPE frequencies so a model trained at context $L_{\mathrm{train}}$ can run at $L_{\mathrm{test}} \gg L_{\mathrm{train}}$ without fully retraining. Primer: [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii).

Schematically, wavelengths longer than some cutoff are stretched by $s = L_{\mathrm{test}} / L_{\mathrm{train}}$, with a temperature correction on the attention scale. Qwen3.8 public spec: native $262144$, YaRN to $1$M.

H3’s Qwen tower uses $\theta = 5\times 10^6$ at encode lengths $\ll 262$K. YaRN is a **Qwen3.8 serving** problem, not an H3 DiT problem.

Architecture: [Part II](h3c_architecture_and_theory.md#part-ii--mapping-toward-qwen38).

---

## Token sampling

**Intuition.** Turn vocab logits into a next-token distribution. **H3 does not sample tokens.** Primer: [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii).

**Definition.** Given logits $z \in \mathbb{R}^{V}$, temperature $T>0$:

<a id="eq-temperature"></a>

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

**Tiny example.** $z=(2, 0)$, $T=1$ → $p \approx (0.88, 0.12)$. $T\to 0$ is greedy argmax (here always class 0). **Top-$k$:** keep the $k$ largest $z_i$, renormalize. **Top-$p$ (nucleus):** smallest set whose cumulative $p$ is $\ge p_{\mathrm{cut}}$, renormalize.

Qwen3.8 thinking-mode defaults cited in the architecture report: $T=1.0$, $\mathrm{top\_p}=0.95$, $\mathrm{top\_k}=20$.

**Common mix-up.** Temperature $T$ here is **not** sequence length and **not** diffusion timestep $t$. Attention’s $1/\sqrt{d}$ is not temperature.

Architecture: [Part II §16](h3c_architecture_and_theory.md#16-if-hello-were-qwen38-decode).

---

## Multi-token prediction

**Intuition.** MTP trains extra heads to predict tokens $t+1, t+2, \ldots$ from the same hidden state (or a small extra trunk). At decode this can become **speculative** drafts that a main head verifies.

Public Qwen3.8 includes MTP. H3 has nothing analogous: one velocity field per Euler evaluation, not a token draft.

Architecture: [Part II](h3c_architecture_and_theory.md#part-ii--mapping-toward-qwen38).

---

## Equation index

| Id | Name | Architecture | Primer |
| --- | --- | --- | --- |
| [eq-embedding](#eq-embedding) | Embedding gather | [§5.1](h3c_architecture_and_theory.md#51-embeddings) | [§3](h3c_concepts_primer.md#3-tokens-and-embeddings) |
| [eq-rmsnorm](#eq-rmsnorm) | RMSNorm | [§5.2](h3c_architecture_and_theory.md#52-rmsnorm-and-head-rmsnorm) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-layernorm](#eq-layernorm) | LayerNorm | [§5.2](h3c_architecture_and_theory.md#52-rmsnorm-and-head-rmsnorm) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-head-rmsnorm](#eq-head-rmsnorm) | Per-head RMS | [§5.2](h3c_architecture_and_theory.md#52-rmsnorm-and-head-rmsnorm) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-groupnorm](#eq-groupnorm) | GroupNorm (VAE) | [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io) | [§8](h3c_concepts_primer.md#8-latents-vaes-and-patchify) |
| [eq-ungated-residual](#eq-ungated-residual) | Ungated residual | [§5.3](h3c_architecture_and_theory.md#53-qwen-layer-exact-order) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-gated-residual](#eq-gated-residual) | AdaLN gate residual | [§6.3](h3c_architecture_and_theory.md#63-run_block) | [§10](h3c_concepts_primer.md#10-dit-adaln-zero-and-the-packed-sequence) |
| [eq-silu](#eq-silu) | SiLU | [§5.3](h3c_architecture_and_theory.md#53-qwen-layer-exact-order) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-swiglu](#eq-swiglu) | SwiGLU | [§5.4](h3c_architecture_and_theory.md#54-swiglu-mlp) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-geglu](#eq-geglu) | GeGLU | [Appendix B](h3c_architecture_and_theory.md#appendix-b--kernel-catalog-83) | — |
| [eq-snake](#eq-snake) | Snake | [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io) | [§8](h3c_concepts_primer.md#8-latents-vaes-and-patchify) |
| [eq-gemm](#eq-gemm) | Linear / GEMM | [§9.3](h3c_architecture_and_theory.md#93-mapping-a-neural-op-onto-hardware) | [§2](h3c_concepts_primer.md#2-tensors-and-linear-layers) |
| [eq-qwen-layer](#eq-qwen-layer) | Qwen `encode_layer` | [§5.3](h3c_architecture_and_theory.md#53-qwen-layer-exact-order) | [§4](h3c_concepts_primer.md#4-the-residual-stream) |
| [eq-softmax](#eq-softmax) | Softmax | [§5.5](h3c_architecture_and_theory.md#55-causal-gqa) | [§5](h3c_concepts_primer.md#5-attention) |
| [eq-sdpa](#eq-sdpa) | Scaled dot-product attention | [§6.5](h3c_architecture_and_theory.md#65-bidirectional-sdpa) | [§5](h3c_concepts_primer.md#5-attention) |
| [eq-causal](#eq-causal) | Causal mask | [§5.5](h3c_architecture_and_theory.md#55-causal-gqa) | [§5](h3c_concepts_primer.md#5-attention) |
| [eq-gqa-index](#eq-gqa-index) | GQA head map | [§5.5](h3c_architecture_and_theory.md#55-causal-gqa) | [§5](h3c_concepts_primer.md#5-attention) |
| [eq-attention-scale](#eq-attention-scale) | $1/\sqrt{d}$ | [§5.5](h3c_architecture_and_theory.md#55-causal-gqa) | [§5](h3c_concepts_primer.md#5-attention) |
| [eq-rope-freq](#eq-rope-freq) | RoPE $\omega_i$ | [§5.6](h3c_architecture_and_theory.md#56-text-rope-and-mrope) | [§6](h3c_concepts_primer.md#6-rotary-position-embeddings) |
| [eq-rope-rotate](#eq-rope-rotate) | 2D rotation | [§5.6](h3c_architecture_and_theory.md#56-text-rope-and-mrope) | [§6](h3c_concepts_primer.md#6-rotary-position-embeddings) |
| [eq-mrope](#eq-mrope) | mRoPE axis cycle | [§5.6](h3c_architecture_and_theory.md#56-text-rope-and-mrope) | [§6](h3c_concepts_primer.md#6-rotary-position-embeddings) |
| [eq-rope-3d](#eq-rope-3d) | DiT 3D positions | [§6.6](h3c_architecture_and_theory.md#66-dit-3d-rope) | [§6](h3c_concepts_primer.md#6-rotary-position-embeddings) |
| [eq-partial-rotary](#eq-partial-rotary) | Partial RoPE | [§6.6](h3c_architecture_and_theory.md#66-dit-3d-rope) | [§6](h3c_concepts_primer.md#6-rotary-position-embeddings) |
| [eq-adaln](#eq-adaln) | AdaLN affine | [§6.2](h3c_architecture_and_theory.md#62-adaln-zero-and-the-schedule) | [§10](h3c_concepts_primer.md#10-dit-adaln-zero-and-the-packed-sequence) |
| [eq-dit-block](#eq-dit-block) | DiT `run_block` | [§6.3](h3c_architecture_and_theory.md#63-run_block) | [§10](h3c_concepts_primer.md#10-dit-adaln-zero-and-the-packed-sequence) |
| [eq-sinusoidal](#eq-sinusoidal) | Timestep sinusoid | [§6.2](h3c_architecture_and_theory.md#62-adaln-zero-and-the-schedule) | [§10](h3c_concepts_primer.md#10-dit-adaln-zero-and-the-packed-sequence) |
| [eq-euler](#eq-euler) | Velocity Euler | [§6.4](h3c_architecture_and_theory.md#64-final-heads-are-velocities) | [§9](h3c_concepts_primer.md#9-diffusion-flow-matching-and-euler) |
| [eq-res-euler](#eq-res-euler) | RES first order | [§7](h3c_architecture_and_theory.md#7-vision-vaes-and-media-io) | [§9](h3c_concepts_primer.md#9-diffusion-flow-matching-and-euler) |
| [eq-lm-head](#eq-lm-head) | LM logits | [§6.4](h3c_architecture_and_theory.md#64-final-heads-are-velocities) | [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii) |
| [eq-kv-cache-size](#eq-kv-cache-size) | KV cache bytes | [§17](h3c_architecture_and_theory.md#17-why-a-kv-cache-exists-in-llms) | [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii) |
| [eq-absmax](#eq-absmax) | Symmetric absmax | [§10](h3c_architecture_and_theory.md#10-quantization) | [§11](h3c_concepts_primer.md#11-inference-engineering) |
| [eq-int8-dequant](#eq-int8-dequant) | Int8 GEMM dequant | [§10](h3c_architecture_and_theory.md#10-quantization) | [§11](h3c_concepts_primer.md#11-inference-engineering) |
| [eq-grouped-scale](#eq-grouped-scale) | Grouped act. scales | [§10](h3c_architecture_and_theory.md#10-quantization) | [§11](h3c_concepts_primer.md#11-inference-engineering) |
| [eq-bf16](#eq-bf16) | bfloat16 value | [§8.2](h3c_architecture_and_theory.md#82-dtypes-and-layout) | [§11](h3c_concepts_primer.md#11-inference-engineering) |
| [eq-complexity-attn](#eq-complexity-attn) | Attention $\Theta$ | [§5.5](h3c_architecture_and_theory.md#55-causal-gqa) | [§5](h3c_concepts_primer.md#5-attention) |
| [eq-patchify](#eq-patchify) | Video patch tokens | [§8.3](h3c_architecture_and_theory.md#83-important-tensors) | [§8](h3c_concepts_primer.md#8-latents-vaes-and-patchify) |
| [eq-roofline](#eq-roofline) | GEMM intensity | [§11](h3c_architecture_and_theory.md#11-memory-management) | [§11](h3c_concepts_primer.md#11-inference-engineering) |
| [eq-velocity-reuse](#eq-velocity-reuse) | Velocity extrapolate | [§12.1](h3c_architecture_and_theory.md#121-optimization-inventory) | [§9](h3c_concepts_primer.md#9-diffusion-flow-matching-and-euler) |
| [eq-deltanet](#eq-deltanet) | Delta rule (Qwen3.8) | [§18](h3c_architecture_and_theory.md#18-component-mapping) | [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii) |
| [eq-temperature](#eq-temperature) | Temperature softmax | [§16](h3c_architecture_and_theory.md#16-if-hello-were-qwen38-decode) | [§12](h3c_concepts_primer.md#12-vocabulary-that-appears-only-in-architecture-part-ii) |

---

*This file is THEORY for the math and FACT for “in h3.c” wiring notes. It does not modify source. Pedagogy: [Concept primer](h3c_concepts_primer.md).*
