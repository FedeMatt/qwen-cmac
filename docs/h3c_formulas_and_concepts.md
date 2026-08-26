# h3.c Formulas and Technical Concepts

Companion to [Architecture and Theoretical Analysis](h3c_architecture_and_theory.md). That document is source-grounded (what the code does). This document defines the **ML, linear-algebra, and numerical** objects the architecture report refers to.

Every display equation has an anchor of the form `#eq-…`. The architecture report links here rather than leaving raw TeX inline.

**Math preview.** Cursor / VS Code Markdown preview and GitHub render math when it is wrapped in dollar signs: one pair for inline, two on their own lines for display (blank line before and after). They do **not** compile LaTeX parenthesis-backslash or bracket-backslash delimiters. This file uses only dollar delimiters so the preview compiles.

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
| $\sigma$ | (1) SiLU, or (2) diffusion noise level — context decides |
| $\varepsilon$ | RMS/LayerNorm stabilizer |
| $\theta$ | RoPE base frequency |
| $W$ | learned matrix; $w$ a learned per-channel vector |
| $\odot$ | elementwise (Hadamard) product |
| $\leftarrow$ | in-place residual update |

Vectors are row-major as in the C tensors: a hidden state of $T$ tokens is $[T, d]$. A linear layer $x W$ maps the last axis.

Back to the architecture report: [§6 Transformer Theory](h3c_architecture_and_theory.md#6-transformer-theory).

---

## Embeddings

An **embedding table** $E \in \mathbb{R}^{V \times d}$ is a lookup: token id $t$ selects row $E[t]$. There is no multiply; it is a gather.

<a id="eq-embedding"></a>

$$
h_0[t] = E[\mathrm{id}_t] \quad\text{or}\quad \mathrm{vision}[t]
$$

**Why it exists.** Discrete tokens have to become vectors before a transformer can run. $V$ is the vocabulary (H3 Qwen: $151936$).

**In h3.c.** `h3_gpu_embedding_bf16` gathers `[T, 5120]` BF16. Optional vision spans **overwrite** those rows; layers $0$–$2$ then add deepstack residuals.

**Tied LM head (not in h3.c).** If the output projection reuses $E$, logits are $h E^\top$. See [language-model head](#language-model-head).

Architecture: [§6.1](h3c_architecture_and_theory.md#61-embeddings-qwen).

---

## RMSNorm

Root-mean-square normalization scales a vector by its RMS, then by a learned gain $w \in \mathbb{R}^d$. It does **not** subtract a mean (that is [LayerNorm](#layernorm)).

<a id="eq-rmsnorm"></a>

$$
\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \, w_i
$$

**Why RMS, not LayerNorm.** Mean-centering is extra bandwidth for little gain in many LLMs (Zhang & Sennrich, 2019). The expensive part is the reduction $\sum x_j^2$ along the hidden axis.

**Stabilizer.** Qwen uses $\varepsilon = 10^{-6}$. DiT / refiner use $\varepsilon = 10^{-5}$. Too small and a near-zero row blows up; too large and the scale is biased.

**In h3.c.** `h3_rms_norm_bf16` / `h3_rms_norm_f32`: one threadgroup per row, shared-memory reduction.

Architecture: [§6.2](h3c_architecture_and_theory.md#62-rmsnorm).

---

## LayerNorm

LayerNorm also centers:

<a id="eq-layernorm"></a>

$$
\mathrm{LayerNorm}(x)_i = \frac{x_i - \mu}{\sqrt{\frac{1}{d}\sum_j (x_j-\mu)^2 + \varepsilon}} \, w_i + b_i, \quad \mu = \frac{1}{d}\sum_j x_j
$$

Vision in h3.c uses this (`h3_gpu_layer_norm_bf16`) with weight **and** bias. Qwen language layers and the DiT use RMSNorm instead.

---

## Head RMSNorm

Qwen3-family models RMS-normalize **each attention head** of $Q$ and $K$ over $d_{\mathrm{head}}$ only, not the concatenated width $n_q \, d_{\mathrm{head}}$.

<a id="eq-head-rmsnorm"></a>

$$
\mathrm{HeadRMS}(Q)_{t,h,:} = \mathrm{RMSNorm}_{d_{\mathrm{head}}}(Q_{t,h,:})
$$

**Why.** Per-head RMS keeps RoPE and dots from being dominated by a few heads’ magnitude. It is applied **after** the Q/K projections and **before** RoPE in H3’s `encode_layer`.

**In h3.c.** `h3_gpu_head_rms_norm_bf16`. DiT QKV kernels fold a related RMS into the RoPE kernel rather than a separate Qwen-style head RMS on the language tower.

---

## Group normalization

Used in the video VAE, not the transformers. Channels are split into $G$ groups; each group is normalized like LayerNorm over $(T,H,W)$ spatial-temporal extents.

<a id="eq-groupnorm"></a>

$$
\mathrm{GN}(x)_{n,c} = \frac{x_{n,c} - \mu_{g(c)}}{\sqrt{\sigma^2_{g(c)} + \varepsilon}} \, w_c + b_c
$$

H3 fuses GN+SiLU in `h3_vae_encoder_group_norm_silu_f32`. Irrelevant to a Qwen3.8 text runtime.

---

## Residual connections and pre-norm

A residual block predicts a **delta** and adds it back. **Pre-norm** applies the norm *before* the sublayer, which is what both H3 networks do.

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

**Why pre-norm.** Gradients through the skip $x \to x$ stay well-scaled even when $F$ is large. Post-norm (norm after the add) is harder to train at this depth.

**Why gates on DiT.** AdaLN-Zero initializes $g \approx 0$ so each block starts as the identity; the network learns how much of $F$ to let through at each noise level. At inference the gates are ordinary tensors looked up from the schedule.

Architecture: [§6.9](h3c_architecture_and_theory.md#69-residual-connections).

---

## SiLU

Sigmoid Linear Unit, also called Swish:

<a id="eq-silu"></a>

$$
\sigma(z) = \frac{z}{1 + e^{-z}} = z \cdot \mathrm{sigmoid}(z)
$$

Smooth, non-monotonic near $0$, cheap (one `exp`). H3’s `h3_silu_mul_bf16` uses exactly $z / (1+e^{-z})$.

Do not confuse this $\sigma$ with diffusion noise level $\sigma$ in [Euler](#diffusion-flow-matching-and-euler).

---

## SwiGLU

A **gated** MLP: one projection is passed through SiLU and multiplies a second projection; a third projection writes back to $d_{\mathrm{model}}$.

<a id="eq-swiglu"></a>

$$
\mathrm{SwiGLU}(x) = \sigma(x W_G) \odot (x W_U)
$$

Full block:

$$
\mathrm{MLP}(x) = \bigl(\sigma(x W_G) \odot (x W_U)\bigr) W_D
$$

**Why gated.** The gate can suppress channels per token. You get more expressivity than a plain ReLU MLP without widening $W_D$.

**Shapes in this repo.**

| Network | $d_{\mathrm{model}}$ | intermediate (up/gate) |
| --- | --- | --- |
| H3 Qwen | $5120$ | $25600$ (separate `gate_proj` / `up_proj`) |
| H3 DiT | $5376$ | $14336$ (fused `fc1` of width $2 \times 14336$) |
| Qwen3.8-27B (public spec) | $5120$ | $17408$ |

**In h3.c.** Qwen: three linears + `h3_gpu_silu_mul_bf16`. DiT: fused `h3_gpu_swiglu_bf16` or MPSGraph / int8 TensorOps MLP.

Architecture: [§6.4](h3c_architecture_and_theory.md#64-swiglu-mlp).

---

## GeGLU

Same idea as SwiGLU with GELU instead of SiLU. Present as `h3_geglu_f32` for audio pieces, not the Qwen/DiT core.

<a id="eq-geglu"></a>

$$
\mathrm{GeGLU}(x) = \mathrm{GELU}(x W_G) \odot (x W_U)
$$

---

## Snake

Periodic activation used in neural audio (BigVGAN-style):

<a id="eq-snake"></a>

$$
\mathrm{Snake}(x) = x + \frac{1}{\alpha} \sin^2(\alpha x)
$$

$\alpha$ is a learned frequency. `h3_snake1d_f32` and fused `h3_alias_free_snake_f32` implement this for the audio VAE / vocoder path.

---

## GEMM

A linear layer on a token batch is a matrix multiply. With row-major activations $X \in \mathbb{R}^{T \times K}$ and weight $W \in \mathbb{R}^{N \times K}$ stored as “output rows”:

<a id="eq-gemm"></a>

$$
Y = X W^\top, \qquad Y \in \mathbb{R}^{T \times N}
$$

Accumulate in FP32 even when $X,W,Y$ are BF16 or int8. That is why H3’s portable kernels tile in `float` and only round at the store.

**Int8.** See [symmetric int8 quantization](#symmetric-int8-quantization): the integer product is rescaled by per-row (or grouped) factors.

**In h3.c.** Portable: `h3_linear_f32_tiled` / `h3_linear_bf16` (16×16). M5: TensorOps `matmul2d` at 128-wide tiles, optional Morton walk.

---

## Qwen transformer layer

Exact `encode_layer` order in `h3_text_encoder.c`:

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

This is a **pre-norm decoder block** with **ungated** residuals and **no KV cache**. H3 runs it once over the full prompt.

H3 constants: $d=5120$, $n_q=64$, $n_{\mathrm{kv}}=8$, $d_{\mathrm{head}}=128$, $50$ layers, RoPE $\theta=5 \times 10^6$.

Architecture: [§6.3](h3c_architecture_and_theory.md#63-qwen-layer-exact-order).

---

## Softmax

Turns logits into a probability vector:

<a id="eq-softmax"></a>

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}
$$

The $\max$ subtract is only for numerical stability (prevents `exp` overflow); it does not change the result.

**Precision.** Attention softmax is computed in FP32 in `h3_gqa_causal_bf16` even though Q/K/V are BF16. Tiny errors in the exp-sum become wrong weights after $50$ layers.

---

## Scaled dot-product attention

Classic Vaswani attention:

<a id="eq-sdpa"></a>

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_{\mathrm{head}}}}\right) V
$$

For one query row $q$ and keys $K \in \mathbb{R}^{L \times d_{\mathrm{head}}}$:

1. scores $s_\ell = q \cdot k_\ell / \sqrt{d_{\mathrm{head}}}$
2. weights $\alpha = \mathrm{softmax}(s)$
3. output $\sum_\ell \alpha_\ell v_\ell$

**DiT.** Bidirectional: $L = S$ for every query (no causal mask), $Q=K=V$, $56$ heads, $d_{\mathrm{head}}=128$. Implemented by **MPSGraph** `scaledDotProductAttentionWithQueryTensor`, not a custom softmax kernel.

**Score memory.** Naively $\Theta(n_{\mathrm{heads}} S^2)$. Library Flash-style tiling can avoid materializing the full $S \times S$ matrix; you still pay $\Theta(S^2)$ compute per head.

Architecture: [§7.2](h3c_architecture_and_theory.md#72-dit-bidirectional-sdpa).

---

## Causal masking

A language-model query at position $t$ may attend only to keys $0..t$ (it must not see the future).

<a id="eq-causal"></a>

$$
\mathrm{mask}_{t,\ell} =
\begin{cases}
0 & \ell \le t \\
-\infty & \ell > t
\end{cases}
\qquad \Rightarrow \qquad \mathrm{key\_count}(t) = t+1
$$

H3’s GQA kernel does not add $-\infty$; it **shortens the loop** to `key_count = query_row + 1`.

DiT attention is **not** causal: every packed token (text, audio, video) can attend to every other.

---

## Grouped-query attention

Multi-head attention (MHA) has $n_q = n_{\mathrm{kv}}$. **GQA** shares each K/V head across a group of query heads.

<a id="eq-gqa-index"></a>

$$
\mathrm{kv\_head}(h_q) = \Big\lfloor h_q \mathbin{/} (n_q / n_{\mathrm{kv}}) \Big\rfloor
$$

H3 Qwen: $n_q=64$, $n_{\mathrm{kv}}=8$, group size $8$. KV tensors are $[T, 8, 128]$ instead of $[T, 64, 128]$.

**Why.** KV size (and later, [KV cache](#kv-cache) size) drops by $n_q / n_{\mathrm{kv}} = 8$. Quality is close to MHA for large models.

**In h3.c.** Custom kernel `h3_gqa_causal_bf16`. Optional `H3_MPS_GQA` uses MPSGraph instead. There is still **no cache**: every encode recomputes K and V for all $T$ tokens.

Architecture: [§7.1](h3c_architecture_and_theory.md#71-qwen-causal-gqa).

---

## Attention scale

<a id="eq-attention-scale"></a>

$$
\mathrm{scale} = \frac{1}{\sqrt{d_{\mathrm{head}}}} = \frac{1}{\sqrt{128}}
$$

Without this, $\mathrm{Var}(q \cdot k) \propto d_{\mathrm{head}}$ and softmax saturates (one-hot weights, dead gradients).

**H3 GQA detail.** The scale is baked into **Q before the dots** to match MLX fused-SDPA order. MPSGraph DiT SDPA takes the scale as an argument instead.

---

## Rotary position embeddings

RoPE (Su et al.) encodes **relative** position by rotating pairs of dimensions of $Q$ and $K$. After rotation, $q_m^\top k_n$ depends on $m-n$, not on absolute indices alone.

Frequencies for pair $i = 0..d_{\mathrm{head}}/2 - 1$:

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

**In h3.c.** Qwen: `h3_gpu_rope_text_bf16` with **F32** cos/sin tables. DiT: fused into QKV+RoPE kernels; `prepare_rope` builds tables.

Architecture: [§7.3](h3c_architecture_and_theory.md#73-rope).

---

## Multimodal RoPE

Qwen-VL **mRoPE** uses **three** position axes (temporal, height, width) instead of a single token index. `position_ids` is stored axis-major `[3, tokens]`.

For a feature index $i$ in the rotary pairs, H3’s text encoder comment says axes **cycle** when $i < 60$:

<a id="eq-mrope"></a>

$$
p^{(i)} = \mathrm{position\_ids}\bigl[i \bmod 3,\; t\bigr]
$$

Text-only prompts still use $p = $ token index on all axes that the kernel reads. Vision/video tokens get 2D/3D coordinates so spatial neighbors have close rotations.

**Vision RoPE** is a separate kernel (`h3_gpu_vision_qkv_rope_bf16`) with 2D/temporal positions and no Q/K RMS in that kernel.

---

## DiT 3D RoPE

Video patches live on a $(T, H, W)$ grid. DiT rotates with three coordinates. `prepare_rope` uses `rope.inv_freq` of length $16$ and axes $(t,\, h\cdot s,\, w\cdot s)$.

<a id="eq-rope-3d"></a>

$$
p = (t,\; h \cdot s,\; w \cdot s)
$$

`ROPE_HALF = 48`: only the **first $48$ of $128$** head dims rotate; the last $80$ stay absolute features ([partial rotary](#partial-rotary)).

At native $256\times 256$, spatial coordinates are halved unless `--use-reference-rope`, to avoid a lattice artifact without adding tokens.

---

## Partial rotary

Only a prefix of $d_{\mathrm{head}}$ is rotated; the tail is not.

<a id="eq-partial-rotary"></a>

$$
\mathrm{RoPE}(x) = \bigl[\mathrm{rotate}(x_{0:r}),\; x_{r:d_{\mathrm{head}}}\bigr]
$$

| Model | $d_{\mathrm{head}}$ | rotated dims | factor |
| --- | --- | --- | --- |
| H3 DiT | $128$ | $48$ | $48/128 = 0.375$ |
| Qwen3.8 Gated Attention (public) | $256$ | $64$ | `partial_rotary_factor` $0.25$ |

The unrotated tail can carry content that should not be mixed with position.

---

## AdaLN-Zero

Adaptive LayerNorm (DiT, Peebles & Xie): the **same** weights denoise at every noise level because a small MLP of the timestep produces per-channel **shift, scale, and gate**.

Given RMS-normalized $x$ and modulation vectors:

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

Row index into the table is `time_row * 3 + tag`. Precomputed by `h3_dit_schedule_precompute` so the timestep MLP is not rerun inside `run_block`.

“Zero” refers to **training init**: gates start near $0$, so the residual path dominates until the network learns otherwise.

Architecture: [§6.5](h3c_architecture_and_theory.md#65-dit-adaln-zero-block).

---

## DiT block

`run_block` (`h3_dit.c`):

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

Attention is **bidirectional full SDPA**, $56$ heads, **no GQA**, **no causal mask**. Text, audio, and video tokens sit in one packed sequence ([MMDiT](h3c_architecture_and_theory.md#34-two-transformers)-style).

Production fuses the attention gate with the following MLP AdaLN (`h3_gpu_gate_adaln_bf16`) and, across blocks, the MLP gate with the next attention AdaLN.

**Refiner.** Two DiT-like blocks on text only: full SDPA, **no RoPE**, **no AdaLN**, ungated pre-norm residuals. Run once at load, not every Euler step.

---

## Timestep embedding

Diffusion blocks need a vector that represents the current noise level. H3 uses $t = 1 - \sigma$ (clean $\leftrightarrow$ $t \approx 1$, noisy $\leftrightarrow$ $t \approx 0$ depending on the $\sigma$ grid), sinusoidal features of size $256$, then an MLP.

<a id="eq-sinusoidal"></a>

$$
e_{2i}(t) = \sin\bigl(t \cdot \omega_i\bigr), \qquad e_{2i+1}(t) = \cos\bigl(t \cdot \omega_i\bigr)
$$

Then: Linear $256 \to 5376$ → SiLU → Linear $5376 \to 2688$ F32 → cast BF16 → SiLU → per-block `adaln_proj` into the six-slot table.

Visual conditions use timestep $0.999$; audio conditions use $1.0$ when not near-terminal (schedule row map).

Architecture: [§6.7](h3c_architecture_and_theory.md#67-timestep-embedding).

---

## Diffusion, flow matching, and Euler

H3 is **not** an autoregressive LM. The DiT predicts a **velocity** field on video/audio latents. The sampler is an Euler integrator on a decreasing noise schedule $\sigma_0 > \sigma_1 > \cdots \ge 0$.

<a id="eq-euler"></a>

$$
x_{\sigma_{\mathrm{next}}} = x_{\sigma} + (\sigma - \sigma_{\mathrm{next}}) \, v_\theta(x_{\sigma}, \sigma)
$$

That is exactly `h3_euler_velocity_step`: `sample += delta * velocity` with `delta = sigma - sigma_next`. The GPU kernel `h3_euler_bf16` is the same `fma` plus optional [velocity reuse](#core-reuse-and-velocity-reuse).

**Flow-matching view.** If $x_\sigma$ interpolates between noise and data, $v_\theta$ approximates $\mathrm{d}x/\mathrm{d}\sigma$. Euler is the first-order ODE step. (Training details live in MiniMax/H3 papers, not in this repo.)

**Not logits.** Final DiT heads emit video velocity $[24,T,H,W]$ F32 and audio velocity $[32,2,T]$ F32. There is no vocabulary softmax.

Architecture: [§3.1](h3c_architecture_and_theory.md#31-actual-execution-path), [§6.8](h3c_architecture_and_theory.md#68-final-head-and-logits).

---

## RES sampler

H3 also implements a rectified exponential solver `h3_res_step` for a **denoised** parameterization (distinct from the production velocity Euler).

Let $\sigma > \sigma_{\mathrm{next}} \ge 0$. First-order (no previous denoised, or last step):

<a id="eq-res-euler"></a>

$$
\frac{\mathrm{d}x}{\mathrm{d}\sigma} \approx \frac{x - x_0}{\sigma}, \qquad x \leftarrow x + \frac{x - x_0}{\sigma}\,(\sigma_{\mathrm{next}}-\sigma)
$$

Higher-order branch uses $t = -\log\sigma$, $h = t_{\mathrm{next}} - t$, $\varphi_1(h) = (e^{h}-1)/h$ (code: `expm1(value)/value` on $-h$), and mixes current and previous denoised with coefficients $b_1,b_2$. This is a **2nd-order exponential integrator**, not the default serving path.

---

## Language-model head

A causal LM maps the last hidden row to vocabulary logits:

<a id="eq-lm-head"></a>

$$
\mathrm{logits} = h W_E^\top \in \mathbb{R}^{V}
$$

Often $W_E = E$ (tied embeddings). H3 **has no LM head**. Qwen3.8 needs this plus [token sampling](#token-sampling). $V = 248320$ in the public 3.8 spec versus $151936$ in H3’s Qwen tower.

Architecture: [§6.8](h3c_architecture_and_theory.md#68-final-head-and-logits).

---

## KV cache

Autoregressive decoding of token $t+1$ needs $K_{0:t}$, $V_{0:t}$. Recomputing projections from all past hidden states is $\Theta(t)$ GEMM per new token per layer. Storing K/V makes the extra work **attention against the cache only**.

For GQA, cache bytes (K and V) are about

<a id="eq-kv-cache-size"></a>

$$
2 \times n_{\mathrm{layers}} \times n_{\mathrm{kv}} \times d_{\mathrm{head}} \times T \times \mathrm{sizeof}(\mathrm{dtype})
$$

**H3 does not have this.** Text encode is a full-sequence forward; DiT is full bidirectional SDPA every step. Caches that *do* exist (session embeddings, AdaLN tables, SSD slots, …) are listed in [architecture §8.1](h3c_architecture_and_theory.md#81-what-h3c-stores-fact).

**Qwen3.8 proposal.** (1) GQA KV for $16$ Gated Attention layers ($24$ Q / $4$ KV, $d_{\mathrm{head}}=256$). (2) A **fixed-size** [DeltaNet](#gated-deltanet) recurrent state for $48$ linear layers — not a growing KV tensor.

Worked numbers (BF16, GQA only, ignoring DeltaNet): $16$ layers $\times$ $4$ KV heads $\times$ $256$ $\times$ $T$ $\times$ $2$ (K and V) $\times$ $2$ bytes $= 65536\, T$ bytes. At $T=262144$ that is $16$ GiB **just** for softmax-attention KV, before activations and weights.

Architecture: [§8](h3c_architecture_and_theory.md#8-kv-cache).

---

## Symmetric int8 quantization

Map a real vector to `int8` with **no zero-point** (symmetric around $0$). H3 weight quant uses one F32 scale per **output row**, values in $[-127, 127]$.

<a id="eq-absmax"></a>

$$
s = \frac{\max_j |W_{r,j}|}{127}, \qquad q_j = \mathrm{round}\bigl(\mathrm{clip}(W_{r,j}/s,\,-127,\,127)\bigr)
$$

Dequant of a matmul accumulation (kernel comments):

<a id="eq-int8-dequant"></a>

$$
y_{r,c} = \mathrm{i32\_acc}_{r,c} \cdot s^{\mathrm{in}}_r \cdot s^{W}_c
$$

$s^{W}_c$ is the weight scale for output channel $c$. $s^{\mathrm{in}}_r$ is the **activation** scale for row $r$ (or a group — next section). Integer MMA runs on $q_{\mathrm{in}} \times q_W$; the two scales reconstruct BF16/F32.

**Why 127 not 128.** `int8` can represent $-128$, but using $[-127,127]$ keeps the grid symmetric so $0$ is exact and absmax scaling is unbiased around zero.

Architecture: [§9.1](h3c_architecture_and_theory.md#91-weight-quantization-fact).

---

## Grouped activation scales

One scale per whole activation row is crude when the row is wide (DiT FC2 $K=14336$). H3’s default FC2 path uses groups of $1024$:

<a id="eq-grouped-scale"></a>

$$
s^{\mathrm{in}}_{r,g} = \frac{\max_{j \in g} |x_{r,j}|}{127}, \qquad |g| = 1024
$$

`--use-int8-row-fc2` falls back to one scale per row and a full-$K$ TensorOps product (faster, less conservative).

QKV / attention-out can fuse absmax into AdaLN (`h3_gpu_gate_adaln_quantize_int8`) so the extra pass does not reread DRAM.

Architecture: [§9.2](h3c_architecture_and_theory.md#92-activation-quantization-fact).

---

## bfloat16

BF16: $1$ sign bit, **$8$ exponent bits** (same range as FP32), **$7$ mantissa bits**.

<a id="eq-bf16"></a>

$$
\mathrm{value} = (-1)^s \cdot 2^{e-127} \cdot (1 + m/128)
$$

(for normal numbers; $e=0$ is subnormal/zero as in IEEE-ish BF16).

**Consequence.** Range matches FP32, so GEMM accumulators rarely overflow; precision is coarse (~$2^{-7} \approx 0.8\%$). H3 therefore:

- accumulates in FP32
- rounds at operation boundaries to match the released compute dtype
- stores tensors as `uint16` / Metal `ushort` bit patterns
- has **no FP16 path**

That is also why late DiT oracles use loose rel-L2: BF16 + attention is not associative.

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

---

## VAE and latents

A variational autoencoder maps RGB/PCM $\leftrightarrow$ a smaller **latent** that the DiT denoises.

H3 video: spatial ratio $16$, causal temporal compression $\lceil T/4 \rceil$, latent channels $24$, patchify $2\times 2$ so tokens are $96$ wide before `video_patch_proj`. Audio: $32$ kHz stereo → latent `[32, 2, T]` with hop $800$.

You do not need VAE math for a Qwen3.8 **text** runtime. For media: encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$; at inference H3 uses a deterministic conv encoder/decoder, not a sampled posterior.

Architecture: [§4.7](h3c_architecture_and_theory.md#47-vision-vaes-media).

---

## Patchify

Split a latent grid into non-overlapping patches and flatten each patch to a token.

<a id="eq-patchify"></a>

$$
\text{video tokens} = T \cdot \frac{H}{2} \cdot \frac{W}{2}, \qquad \text{patch} \in \mathbb{R}^{2 \cdot 2 \cdot 24} = \mathbb{R}^{96}
$$

Then a linear `video_patch_proj` maps $96 \to 5376$. Inverse **unpatchify** after the velocity head.

`--token-reduction` averages pairs of horizontal video tokens in middle DiT blocks (`h3_token_pool_bf16`) and expands the residual back (`h3_token_expand_*`), cutting $S$.

---

## Byte-pair encoding

BPE builds a vocabulary by repeatedly merging the most frequent adjacent pair of symbols, starting from bytes (here: byte-level BPE after ICU + NFC).

Encoding is greedy: left-to-right longest merges from `tokenizer.json`. H3 requires BPE, NFC, **no unk token**, pad id `151643`.

This is not a neural formula; it is a deterministic string $\to$ id map. Wrong merges produce a fluent but **wrong** embedding sequence that no later kernel can undo.

Architecture: [§4.5](h3c_architecture_and_theory.md#45-tokenizer-and-text).

---

## Safetensors

On-disk layout (Hugging Face):

```text
uint64 little-endian header_size
JSON { name: { dtype, shape, data_offsets: [begin, end] }, __metadata__? }
payload at file offset 8 + header_size + begin
```

No math beyond “bytes at an offset.” H3 parses with `h3_st_read_header` (max $256$ MiB JSON) and `pread`s payloads in $1$ GiB chunks. mmap/zero-copy is optional later in `h3_gpu_tensor_load_file`, not in the parser.

Architecture: [§5.1](h3c_architecture_and_theory.md#51-checkpoint-layout).

---

## Unified memory

On Apple Silicon, CPU and GPU share one DRAM. `MTLResourceStorageModeShared` makes `buffer.contents` CPU-visible. There is little PCIe-style “upload”; the cost is **DRAM bandwidth** and cache coherence.

Implication: large GEMMs are usually [bandwidth-bound](#roofline-and-memory-bandwidth). Int8, fusion, and activation aliasing exist to **move fewer bytes**, not to invent FLOPs.

Architecture: [§10.3](h3c_architecture_and_theory.md#103-mapping-neural-op-hardware), [§12](h3c_architecture_and_theory.md#12-memory-management).

---

## Roofline and memory bandwidth

Arithmetic intensity of a GEMM $Y = X W^\top$ with $T$ large, $K,N$ the inner/output:

<a id="eq-roofline"></a>

$$
I \approx \frac{2 T K N}{2(TK + KN + TN)} \;\text{FLOPs/byte (F32)}
$$

For H3’s huge $K$ (e.g. FC1 $5376 \to 28672$) intensity is still low enough that **bytes of $W$** dominate. Int8 halves those bytes per MAC; TensorOps MMA raises the FLOP ceiling so the roof is even more clearly bandwidth.

---

## Morton order

A Morton (Z-order) curve interleaves $x$ and $y$ bits so adjacent tiles in 2D stay nearby in 1D DRAM. H3’s M5 TensorOps GEMMs launch in Morton tile order (`*_morton`, `*_morton4`) to cut cache thrash on $128\times 64$ / $128\times 128$ tiles. Comments credit Draw Things / ccv.

No closed-form ML formula; it is a memory-layout permutation of the launch grid.

---

## Token reduction

Algorithmic cut of DiT sequence length $S$: average neighboring **horizontal** video tokens in middle blocks, run attention/MLP on the short sequence, expand the residual back to the full grid.

Quadratic attention then sees roughly $S/2$ video tokens in those blocks. Quality tradeoff is documented in the README (aggressive settings can ghost limbs).

---

## Core reuse and velocity reuse

**`--core-reuse`.** The expensive $50$-block trunk is assumed to change slowly across nearby $\sigma$. Cache

$$
\Delta = h_{\mathrm{after}} - h_{\mathrm{before}}
$$

and skip the trunk, applying a cheaper timestep-dependent head.

**`--reuse` (velocity extrapolation).** Skip some DiT forwards. The GPU Euler kernel blends the last two BF16 velocities:

<a id="eq-velocity-reuse"></a>

$$
v = v_{\mathrm{last}} + r\,(v_{\mathrm{last}} - v_{\mathrm{previous}})
$$

then $x \leftarrow x + \Delta\sigma \cdot v$. $r=0$ is “use last velocity only.” This is linear extrapolation in step index, not a learned sampler.

Architecture: [§13.3](h3c_architecture_and_theory.md#133-optimization-inventory).

---

## Gated DeltaNet

**Not implemented in h3.c.** Public Qwen3.8-27B uses $48$ Gated DeltaNet layers + $16$ Gated Attention layers.

Linear attention replaces $QK^\top$ softmax with a **recurrent state** $S_t$ of fixed size (independent of $T$). The **delta rule** (Schlag et al.; Gated DeltaNet, Yang et al.) updates a matrix-valued memory with a gated write:

<a id="eq-deltanet"></a>

$$
S_t = S_{t-1}\bigl(I - \beta_t k_t k_t^\top\bigr) + \beta_t v_t k_t^\top
$$

$k_t,v_t$ are per-token keys/values; $\beta_t$ is a learned write strength. Extra **gates** and a short conv (kernel $4$ in the 3.8 spec) modify this rule; the exact Qwen3.8 kernel is defined by the official code, not by this repository.

**Why it matters for a Mac runtime.** State is $O(1)$ in $T$ per layer (unlike [KV cache](#kv-cache)), which is the only plausible way to approach $262$K context on unified memory. It is also the highest-risk new Metal work: recurrence, layout, and numerical parity against Transformers/vLLM.

Architecture: [§17](h3c_architecture_and_theory.md#17-mapping-to-qwen38), [§19.4](h3c_architecture_and_theory.md#194-new-research-required).

---

## YaRN

Yet another RoPE extensioN: interpolate / scale RoPE frequencies so a model trained at context $L_{\mathrm{train}}$ can run at $L_{\mathrm{test}} \gg L_{\mathrm{train}}$ without fully retraining.

Schematically, wavelengths longer than some cutoff are stretched by $s = L_{\mathrm{test}} / L_{\mathrm{train}}$, with a temperature correction on the attention scale. Qwen3.8 public spec: native $262144$, YaRN to $1$M.

H3’s Qwen tower uses $\theta = 5\times 10^6$ at encode lengths $\ll 262$K. YaRN is a **Qwen3.8 serving** problem, not an H3 DiT problem.

---

## Token sampling

Given logits $z \in \mathbb{R}^{V}$, temperature $T>0$:

<a id="eq-temperature"></a>

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

$T \to 0$ is greedy argmax. **Top-$k$:** keep the $k$ largest $z_i$, renormalize. **Top-$p$ (nucleus):** smallest set whose cumulative $p$ is $\ge p_{\mathrm{cut}}$, renormalize.

H3 does not sample tokens. Qwen3.8 thinking-mode defaults cited in the architecture report: $T=1.0$, $\mathrm{top\_p}=0.95$, $\mathrm{top\_k}=20$.

---

## Multi-token prediction

MTP trains extra heads to predict tokens $t+1, t+2, \ldots$ from the same hidden state (or a small extra trunk). At decode this can become **speculative** drafts that a main head verifies.

Public Qwen3.8 includes MTP. H3 has nothing analogous: one velocity field per Euler evaluation, not a token draft.

---

## Equation index

| Id | Name | Architecture |
| --- | --- | --- |
| [eq-embedding](#eq-embedding) | Embedding gather | [§6.1](h3c_architecture_and_theory.md#61-embeddings-qwen) |
| [eq-rmsnorm](#eq-rmsnorm) | RMSNorm | [§6.2](h3c_architecture_and_theory.md#62-rmsnorm) |
| [eq-layernorm](#eq-layernorm) | LayerNorm | [§6.2](h3c_architecture_and_theory.md#62-rmsnorm) |
| [eq-head-rmsnorm](#eq-head-rmsnorm) | Per-head RMS | [§6.2](h3c_architecture_and_theory.md#62-rmsnorm) |
| [eq-groupnorm](#eq-groupnorm) | GroupNorm (VAE) | [§4.7](h3c_architecture_and_theory.md#47-vision-vaes-media) |
| [eq-ungated-residual](#eq-ungated-residual) | Ungated residual | [§6.9](h3c_architecture_and_theory.md#69-residual-connections) |
| [eq-gated-residual](#eq-gated-residual) | AdaLN gate residual | [§6.9](h3c_architecture_and_theory.md#69-residual-connections) |
| [eq-silu](#eq-silu) | SiLU | [§6.3](h3c_architecture_and_theory.md#63-qwen-layer-exact-order) |
| [eq-swiglu](#eq-swiglu) | SwiGLU | [§6.4](h3c_architecture_and_theory.md#64-swiglu-mlp) |
| [eq-geglu](#eq-geglu) | GeGLU | [§11.1](h3c_architecture_and_theory.md#111-fp32-portable) |
| [eq-snake](#eq-snake) | Snake | [§4.7](h3c_architecture_and_theory.md#47-vision-vaes-media) |
| [eq-gemm](#eq-gemm) | Linear / GEMM | [§10.3](h3c_architecture_and_theory.md#103-mapping-neural-op-hardware) |
| [eq-qwen-layer](#eq-qwen-layer) | Qwen `encode_layer` | [§6.3](h3c_architecture_and_theory.md#63-qwen-layer-exact-order) |
| [eq-softmax](#eq-softmax) | Softmax | [§7](h3c_architecture_and_theory.md#7-attention) |
| [eq-sdpa](#eq-sdpa) | Scaled dot-product attention | [§7.2](h3c_architecture_and_theory.md#72-dit-bidirectional-sdpa) |
| [eq-causal](#eq-causal) | Causal mask | [§7.1](h3c_architecture_and_theory.md#71-qwen-causal-gqa) |
| [eq-gqa-index](#eq-gqa-index) | GQA head map | [§7.1](h3c_architecture_and_theory.md#71-qwen-causal-gqa) |
| [eq-attention-scale](#eq-attention-scale) | $1/\sqrt{d}$ | [§7.1](h3c_architecture_and_theory.md#71-qwen-causal-gqa) |
| [eq-rope-freq](#eq-rope-freq) | RoPE $\omega_i$ | [§7.3](h3c_architecture_and_theory.md#73-rope) |
| [eq-rope-rotate](#eq-rope-rotate) | 2D rotation | [§7.3](h3c_architecture_and_theory.md#73-rope) |
| [eq-mrope](#eq-mrope) | mRoPE axis cycle | [§7.3](h3c_architecture_and_theory.md#73-rope) |
| [eq-rope-3d](#eq-rope-3d) | DiT 3D positions | [§7.3](h3c_architecture_and_theory.md#73-rope) |
| [eq-partial-rotary](#eq-partial-rotary) | Partial RoPE | [§7.3](h3c_architecture_and_theory.md#73-rope) |
| [eq-adaln](#eq-adaln) | AdaLN affine | [§6.5](h3c_architecture_and_theory.md#65-dit-adaln-zero-block) |
| [eq-dit-block](#eq-dit-block) | DiT `run_block` | [§6.5](h3c_architecture_and_theory.md#65-dit-adaln-zero-block) |
| [eq-sinusoidal](#eq-sinusoidal) | Timestep sinusoid | [§6.7](h3c_architecture_and_theory.md#67-timestep-embedding) |
| [eq-euler](#eq-euler) | Velocity Euler | [§6.8](h3c_architecture_and_theory.md#68-final-head-and-logits) |
| [eq-res-euler](#eq-res-euler) | RES first order | [§4.4](h3c_architecture_and_theory.md#44-host-geometry-cpu) |
| [eq-lm-head](#eq-lm-head) | LM logits | [§6.8](h3c_architecture_and_theory.md#68-final-head-and-logits) |
| [eq-kv-cache-size](#eq-kv-cache-size) | KV cache bytes | [§8.2](h3c_architecture_and_theory.md#82-why-a-kv-cache-exists-in-llms-theory) |
| [eq-absmax](#eq-absmax) | Symmetric absmax | [§9.1](h3c_architecture_and_theory.md#91-weight-quantization-fact) |
| [eq-int8-dequant](#eq-int8-dequant) | Int8 GEMM dequant | [§9.1](h3c_architecture_and_theory.md#91-weight-quantization-fact) |
| [eq-grouped-scale](#eq-grouped-scale) | Grouped act. scales | [§9.2](h3c_architecture_and_theory.md#92-activation-quantization-fact) |
| [eq-bf16](#eq-bf16) | bfloat16 value | [§5.2](h3c_architecture_and_theory.md#52-dtypes-and-layout) |
| [eq-complexity-attn](#eq-complexity-attn) | Attention $\Theta$ | [§7.1](h3c_architecture_and_theory.md#71-qwen-causal-gqa) |
| [eq-patchify](#eq-patchify) | Video patch tokens | [§5.3](h3c_architecture_and_theory.md#53-important-tensors) |
| [eq-roofline](#eq-roofline) | GEMM intensity | [§12.4](h3c_architecture_and_theory.md#124-bandwidth-vs-compute) |
| [eq-velocity-reuse](#eq-velocity-reuse) | Velocity extrapolate | [§13.3](h3c_architecture_and_theory.md#133-optimization-inventory) |
| [eq-deltanet](#eq-deltanet) | Delta rule (Qwen3.8) | [§17](h3c_architecture_and_theory.md#17-mapping-to-qwen38) |
| [eq-temperature](#eq-temperature) | Temperature softmax | [§15.2](h3c_architecture_and_theory.md#152-proposal-if-hello-were-qwen38-decode) |

---

*This file is THEORY for the math and FACT for “in h3.c” wiring notes. It does not modify source.*
