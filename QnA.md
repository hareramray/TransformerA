# Transformer — Extensive Technical Q&A

A deep, self-test style reference tied to the code in `model.py` and
`train_demo.py`. Questions move from intuition → math → implementation →
training → edge cases → extensions. Use it to check your understanding or as
interview prep.

**Shape legend (used throughout):**
`B` batch · `L` sequence length (`Lq` query, `Lk` key) · `D` model dim (`d_model`) ·
`H` heads · `Dh = D/H` per-head dim · `V` vocab size.

---

## Part 1 — Big picture & motivation

### Q1. What problem does the Transformer solve that RNNs and CNNs did not?
RNNs process tokens sequentially, so (a) they can't parallelize across the time
axis during training, and (b) long-range dependencies must survive many
sequential steps, causing vanishing/exploding gradients. CNNs parallelize but
need many stacked layers to grow a receptive field large enough to connect
distant tokens. The Transformer replaces recurrence with **self-attention**,
which connects *any* two positions in a single layer (path length O(1)) and is
fully parallel across positions. The trade-off is O(L²) compute/memory in
sequence length.

### Q2. Why is it called "attention is all you need"?
The paper showed you can drop recurrence and convolutions entirely and rely on
attention (plus position-wise MLPs, residuals, normalization, and positional
encodings). Attention is the only mechanism that mixes information *across*
positions.

### Q3. Encoder-decoder vs encoder-only vs decoder-only — when do you use each?
- **Encoder-decoder** (this repo): sequence-to-sequence tasks where input and
  output are different sequences — translation, summarization. The encoder
  builds a representation of the source; the decoder generates the target while
  attending to it.
- **Encoder-only** (BERT-style): understanding tasks — classification, NER,
  embeddings. Bidirectional, no causal mask.
- **Decoder-only** (GPT-style): autoregressive generation / language modeling.
  Causal mask, no separate encoder, no cross-attention.

### Q4. Draw the data flow of this model.
```
src ─► Embedding·√D ─► +PosEnc ─► EncoderLayer×N ──┐ (keys & values)
                                                   │
tgt ─► Embedding·√D ─► +PosEnc ─► DecoderLayer×N ◄─┘
                                       │
                                  Linear(generator) ─► logits (B,L,V)
```

---

## Part 2 — Scaled dot-product attention

### Q5. Write the attention formula and name every term.
`Attention(Q,K,V) = softmax(QKᵀ / √Dh) · V`
- `Q` queries `(…,Lq,Dh)`: "what each position is looking for".
- `K` keys `(…,Lk,Dh)`: "what each position offers as an index".
- `V` values `(…,Lk,Dh)`: "the actual content returned".
- `QKᵀ` `(…,Lq,Lk)`: raw similarity of every query to every key.
- `softmax(…)` over the **key axis**: turns similarities into weights summing to 1.
- `· V`: weighted average of values → output `(…,Lq,Dh)`.

### Q6. Why divide by √Dh? What breaks without it?
If `q` and `k` have components with mean 0 and variance 1, their dot product over
`Dh` dimensions has variance `Dh`. As `Dh` grows the logits get large in
magnitude, pushing softmax into a saturated region where one weight ≈ 1 and the
rest ≈ 0. There the softmax gradient is ~0, so learning stalls. Dividing by `√Dh`
rescales the variance back to ~1, keeping softmax in a responsive regime.

### Q7. Over which axis is softmax taken, and why does that axis matter?
`dim=-1`, the **key** axis (`Lk`). Each query produces a distribution over all
keys, so the weights for a single query sum to 1. Softmax over the wrong axis
(queries) would make each key's weights sum to 1 — meaningless.

### Q8. In the code, where does masking happen and how?
In `ScaledDotProductAttention.forward`:
```python
scores = scores.masked_fill(mask == 0, float("-inf"))
attn = F.softmax(scores, dim=-1)
```
Forbidden positions are set to `-inf` **before** softmax, so `exp(-inf)=0` gives
them exactly zero weight. Masking after softmax would break the "sums to 1"
property.

### Q9. Why `-inf` and not a large negative number like `-1e9`?
`-1e9` is the common practical choice and works. `float("-inf")` is exact
(`exp` → 0) and avoids leaking a tiny weight. The risk with `-inf`: if an entire
row is masked, softmax produces `NaN` (0/0). In this model every query can attend
to at least itself, so no fully-masked rows occur. With variable padding you may
prefer `-1e9` to stay numerically safe.

### Q10. What is the time and memory complexity of attention?
The `scores` tensor is `(B,H,Lq,Lk)`, so both compute and memory are **O(L²)** in
sequence length (for self-attention `Lq=Lk=L`). This is the Transformer's main
scaling bottleneck and the reason for FlashAttention, sliding-window attention,
linear-attention variants, etc.

---

## Part 3 — Multi-head attention

### Q11. Why use multiple heads instead of one big attention?
A single softmax can only put attention in essentially one place per query. With
`H` heads, each operating in a `Dh`-dim subspace, the model can attend to several
different positions/relationships at once (e.g. one head tracks the verb, another
the subject). Total cost is the same as one `D`-dim attention because `H·Dh = D`.

### Q12. Walk through `_split_heads` and `_merge_heads`.
- `_split_heads`: `(B,L,D) → (B,L,H,Dh) → transpose → (B,H,L,Dh)`. Moving the head
  axis next to batch lets `matmul` treat `(B,H)` as independent batch dims, so all
  heads attend in parallel.
- `_merge_heads`: the inverse — `(B,H,L,Dh) → (B,L,H,Dh) → view → (B,L,D)`. The
  `.contiguous()` is required because `transpose` returns a non-contiguous view
  and `.view` needs contiguous memory.

### Q13. What are `w_q, w_k, w_v, w_o` and why are there four?
Four learned linear projections. `w_q/w_k/w_v` project the inputs into the
query/key/value spaces (per-head, since the `D`-dim output is split into heads).
`w_o` mixes the concatenated head outputs back into model space — without it the
heads would never exchange information.

### Q14. In `forward(query, key, value, ...)`, when are those three the same tensor and when different?
- **Self-attention** (encoder, decoder's first sublayer): all three are the same
  sequence `x`.
- **Cross-attention** (decoder's second sublayer): `query` is the decoder state,
  while `key` and `value` are the **encoder output**. That's how the decoder reads
  the source.

### Q15. The mask is described as already having a "head dimension." Explain.
`make_pad_mask` returns `(B,1,1,Lk)` and `make_tgt_mask` returns `(B,1,L,L)`. The
axis-1 singleton broadcasts across all `H` heads when added to `scores`
`(B,H,Lq,Lk)`. The original bug was an extra `unsqueeze(1)` inside
`MultiHeadAttention`, which made the mask 5-D and broke the merge — removing it
fixed it. **Lesson:** decide *one* place that owns the head axis and keep mask
construction consistent with it.

---

## Part 4 — Feed-forward, embeddings, positional encoding

### Q16. What does the position-wise feed-forward network do, and why "position-wise"?
`FFN(x) = Linear2(ReLU(Linear1(x)))`, applied **independently and identically to
every position** (the same weights, no mixing across positions). Attention mixes
across positions; the FFN then transforms each position's vector on its own,
giving the model non-linear per-token processing capacity. `d_ff` is typically
`4·d_model`.

### Q17. Why scale embeddings by √d_model before adding positional encodings?
Embedding weights are initialized small (variance ~1/D). Multiplying by `√D`
raises the embedding magnitude to roughly the same scale as the positional
sinusoids (which range in [-1,1] but accumulate across `D` dims), so neither
signal drowns out the other when summed. It also matches the convention of
weight-tying between embeddings and the output projection.

### Q18. Why do Transformers need positional encodings at all?
Self-attention is **permutation-invariant**: shuffle the input tokens and the set
of outputs just shuffles correspondingly — the mechanism has no inherent notion
of order. Positional encodings inject "where am I" so the model can use word
order.

### Q19. Explain the sinusoidal positional encoding formula.
For position `pos` and dimension `i`:
`PE(pos,2i)=sin(pos/10000^(2i/D))`, `PE(pos,2i+1)=cos(pos/10000^(2i/D))`.
Each dimension is a sinusoid; wavelengths form a geometric progression from `2π`
to `~10000·2π`. Properties: (a) every position gets a unique vector; (b) for any
fixed offset `k`, `PE(pos+k)` is a *linear function* of `PE(pos)`, letting the
model learn to attend by **relative** position; (c) it extrapolates to lengths
longer than seen in training.

### Q20. In code, what is `div_term` and why compute it in log-space?
```python
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
```
It's `1 / 10000^(2i/D)`, the per-dimension frequency. Computing it as
`exp(2i · -log(10000)/D)` is numerically stabler than raising 10000 to a
fractional power directly, and vectorizes cleanly over the even indices.

### Q21. Why `register_buffer` for `pe` instead of `nn.Parameter` or a plain attribute?
`register_buffer` makes `pe` part of the module's state — it moves with
`.to(device)`, is saved/loaded in `state_dict` — but is **not** a trainable
parameter (no gradient, optimizer ignores it). A plain attribute wouldn't move to
GPU automatically; a `Parameter` would wrongly try to learn the fixed encoding.

### Q22. Fixed sinusoidal vs learned positional embeddings — trade-offs?
Sinusoidal: zero parameters, extrapolates to unseen lengths, encodes relative
position naturally. Learned (`nn.Embedding(max_len, D)`): can fit data-specific
patterns, often marginally better in-distribution, but capped at `max_len` and
doesn't extrapolate. Modern models often use neither and adopt rotary (RoPE) or
ALiBi for better length generalization.

---

## Part 5 — Residuals, normalization, and block structure

### Q23. What is the sublayer pattern in `EncoderLayer`/`DecoderLayer`?
`x = LayerNorm(x + Sublayer(x))` — a residual (skip) connection around each
sublayer, followed by layer normalization. Encoder layer = [self-attn, FFN];
decoder layer = [masked self-attn, cross-attn, FFN].

### Q24. Why residual connections?
They give gradients a direct path backward (`∂/∂x` of `x+f(x)` includes the
identity), which prevents vanishing gradients in deep stacks and lets each
sublayer learn a *residual* (a refinement) rather than a full transformation —
much easier to optimize.

### Q25. What does LayerNorm normalize, and how does it differ from BatchNorm?
LayerNorm normalizes across the **feature dimension** (`D`) for each token
independently — mean/variance computed per position, per example. BatchNorm
normalizes across the **batch** for each feature. LayerNorm is preferred for
sequences because it's independent of batch size and sequence length, works the
same in training and inference, and doesn't break with padding/variable lengths.

### Q26. This code uses **post-norm**. What is post-norm vs pre-norm and which is more stable?
- **Post-norm** (original paper, this repo): `LayerNorm(x + Sublayer(x))` — norm
  *after* the residual add.
- **Pre-norm** (most modern models): `x + Sublayer(LayerNorm(x))` — norm *inside*,
  on the sublayer input; the residual stream itself is never normalized.
Pre-norm keeps a clean identity path from input to output, giving more stable
gradients in deep models — often trainable without learning-rate warmup. Post-norm
can reach slightly better final quality when tuned but typically needs warmup and
careful init. (See the README's "next steps" to convert this repo to pre-norm.)

### Q27. Why does the decoder layer have *three* sublayers while the encoder has two?
The decoder adds a **cross-attention** sublayer between its masked self-attention
and its FFN. Self-attention lets the target attend to itself (causally);
cross-attention lets it attend to the encoded source. The encoder has nothing
external to attend to, so it only needs self-attention + FFN.

---

## Part 6 — Masking

### Q28. Name the two kinds of masks and what each prevents.
- **Padding mask**: zeros out attention to `<pad>` tokens so variable-length
  sequences batched together don't attend to meaningless filler.
- **Causal (look-ahead) mask**: lower-triangular; stops position `i` from
  attending to any `j > i`. Essential in the decoder so that predicting token `i`
  cannot peek at the answer (tokens `≥ i`).

### Q29. Why must the decoder's self-attention be causal but the encoder's is not?
The encoder sees the whole source at once (understanding task → bidirectional is
fine and helpful). The decoder is **autoregressive**: at inference it generates
left-to-right and only has past tokens. Training must simulate that constraint,
or the model would learn to cheat by reading future target tokens — and collapse
at inference time when they're absent.

### Q30. Walk through `make_tgt_mask`.
```python
pad    = make_pad_mask(tgt, pad_idx)          # (B,1,1,L) — block pad columns
causal = make_causal_mask(L)                  # (1,L,L)   — block future
return pad & causal.unsqueeze(1)              # (B,1,L,L) — both must allow
```
The logical AND means a position is attendable only if it's **both** a real
(non-pad) token **and** not in the future. Result broadcasts over heads via the
axis-1 singleton.

### Q31. Why is `make_causal_mask` built with `torch.tril`?
`torch.tril(ones(L,L))` is a lower-triangular matrix: entry `(i,j)=1` iff `j ≤ i`.
Row `i` (the query at position `i`) therefore permits keys `0..i` and forbids
`i+1..L-1` — exactly "attend to past and present only".

### Q32. The mask carries the head dimension as a singleton at axis 1. Why is that the right design?
Building the mask once with shape `(B,1,…,Lk)` and letting it broadcast over `H`
avoids materializing an `H`-times-larger mask tensor and keeps a single source of
truth for mask shape. The earlier crash came from adding a *second* head axis
inside attention — the fix was to let the prebuilt singleton do the broadcasting.

---

## Part 7 — The full model & initialization

### Q33. What does the `generator` (final `nn.Linear`) do?
It projects each decoder output vector `(…,D)` to logits over the target
vocabulary `(…,V)`. A softmax over those logits (done inside the loss during
training, or for sampling at inference) gives next-token probabilities.

### Q34. Why Xavier (Glorot) initialization for parameters with `dim > 1`?
Xavier sets weight variance to keep the signal variance roughly constant across
layers (`Var(W) ≈ 2/(fan_in+fan_out)`), which prevents activations/gradients from
exploding or vanishing through the deep stack at the start of training. The
`dim > 1` guard skips biases and 1-D tensors (e.g. LayerNorm params), which should
keep their default init (zeros/ones).

### Q35. Trace the shapes through `Transformer.forward` for `src=(2,9)`, `tgt=(2,7)`, `D=64, V=50`.
- `encode`: `(2,9) → embed → (2,9,64) → +pos → encoder → (2,9,64)`.
- `decode`: `(2,7) → embed → (2,7,64) → +pos → decoder(attends to enc (2,9,64)) → (2,7,64)`.
- `generator`: `(2,7,64) → (2,7,50)`. (Matches the `model.py` self-test.)

### Q36. Could `src_vocab_size` and `tgt_vocab_size` differ? When would they?
Yes — they're independent `nn.Embedding` tables. They differ when source and
target are different languages/symbol sets (e.g. English→German translation).
For same-vocabulary tasks (like the reverse demo) they're equal, and you could
even tie the embedding weights to save parameters.

---

## Part 8 — Training (the reverse-sequence demo)

### Q37. What is the toy task and why is it a good sanity check?
Reverse a sequence of integers (`[1 5 3 8] → [8 3 5 1]`). It's
trivial logically but forces correct use of attention and positional information:
each output position must attend to a *specific* input position determined by
order. It trains in seconds on CPU and the loss should approach zero — a clean
signal that the architecture and masking are wired correctly.

### Q38. Explain the right-shift: why `tgt_in = [BOS, r0, r1, …]` and `tgt_out = [r0, r1, …]`?
**Teacher forcing.** The decoder predicts token `t` from tokens `< t`. Feeding
`tgt_in` shifted right by one (prepend `BOS`, drop the last) means at each
position the decoder's input is the *previous* ground-truth token, and it must
predict the *current* one (`tgt_out`). This aligns inputs and labels so the whole
sequence is trained in parallel in a single forward pass.

### Q39. What is teacher forcing and what's its downside?
During training we feed the **ground-truth** previous token rather than the
model's own prediction. This stabilizes and parallelizes training. Downside:
**exposure bias** — at inference the model consumes its *own* (possibly wrong)
predictions, a distribution it never saw in training, so early mistakes can
compound. Mitigations: scheduled sampling, sequence-level objectives.

### Q40. Why `CrossEntropyLoss(ignore_index=PAD)`?
Cross-entropy is the standard classification loss over the vocabulary at each
position. `ignore_index=PAD` excludes padding positions from the loss (and
gradient) so the model isn't trained to "predict padding" and the loss isn't
diluted by filler. (The toy demo has no padding, but it's correct practice.)

### Q41. Why does the loss reshape to `(-1, V)` and the target to `(-1)`?
`CrossEntropyLoss` expects logits `(N, C)` and integer targets `(N,)`. We flatten
the batch and sequence axes together: `logits (B,L,V) → (B·L, V)` and
`targets (B,L) → (B·L,)`, treating every position as one classification example.

### Q42. Walk through one training step in `train_demo.py`.
1. `make_batch` → `src, tgt_in, tgt_out`.
2. Build `src_mask` (pad) and `tgt_mask` (pad ∧ causal).
3. Forward: `logits = model(src, tgt_in, src_mask, tgt_mask)`.
4. `loss = CE(logits.reshape(-1,V), tgt_out.reshape(-1))`.
5. `optimizer.zero_grad(); loss.backward(); optimizer.step()`.
The `zero_grad` matters because PyTorch **accumulates** gradients by default;
forgetting it sums gradients across steps.

### Q43. Why is decoding done step-by-step in `greedy_decode` instead of one forward pass like training?
At inference there is no ground-truth target to teacher-force. Generation is
inherently sequential: you produce token `t`, append it, and feed the longer
sequence back to produce `t+1`. Each step encodes the source once (cached as
`enc_out`) and runs the decoder on the tokens generated so far, taking the
`argmax` of the **last** position's logits.

### Q44. What is greedy decoding and what are the alternatives?
Greedy = take the single highest-probability token at each step. Simple and
deterministic but myopic (a locally-best token can lead to a globally-worse
sequence). Alternatives: **beam search** (keep top-k partial sequences),
**sampling** with temperature, **top-k** / **top-p (nucleus)** sampling for
diverse generation.

### Q45. The demo re-encodes the source every decode step. Is that necessary? How would you optimize?
The *encoder* output doesn't change, and indeed the code computes `enc_out` once
before the loop — good. The remaining inefficiency is the **decoder**: it
recomputes attention over all previously generated tokens each step. Production
implementations add a **KV cache**, storing each layer's past keys/values so each
new step only computes attention for the single new query — turning generation
from O(L²) per sequence into O(L) incremental work per step.

---

## Part 9 — Hyperparameters & scaling

### Q46. What are the paper's "base" hyperparameters and what does the demo use?
Base: `d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1` (~65M
params for translation). The demo shrinks to `d_model=64, heads=4, layers=2,
d_ff=128` (~171K params) so it trains in ~30s on CPU.

### Q47. The constraint `d_model % num_heads == 0` — why?
Each head gets `Dh = d_model / num_heads` dimensions and the heads must tile
`d_model` exactly when concatenated. A non-divisible split would leave dangling
dimensions. The code `assert`s this in `MultiHeadAttention.__init__`.

### Q48. How do parameters scale with `d_model`, `num_layers`, `d_ff`?
Per layer, attention is ~`4·d_model²` (the four projections) and the FFN is
~`2·d_model·d_ff`. Total scales roughly **linearly in `num_layers`** and
**quadratically in `d_model`** (since `d_ff ∝ d_model`). Embeddings add
`V·d_model`, which dominates at small `d_model` / large vocab.

### Q49. What role does dropout play and where is it applied here?
Dropout randomly zeros activations during training for regularization (reducing
overfitting). In this code it's applied: after the output projection in
`MultiHeadAttention`, inside the FFN (after ReLU), and after adding positional
encodings. It's automatically disabled by `model.eval()` during decoding.

### Q50. Why call `model.train()` vs `model.eval()`?
They flip the mode of dropout and (if present) batchnorm. `train()` enables
dropout (stochastic); `eval()` disables it for deterministic inference.
`greedy_decode` calls `model.eval()` so generation isn't randomly perturbed.

---

## Part 10 — Edge cases, debugging, extensions

### Q51. You see `NaN` loss after a few steps. What are the usual suspects?
Learning rate too high (exploding activations); a fully-masked attention row
producing `NaN` from softmax; missing `zero_grad` causing gradient blow-up;
numerical overflow without the `√Dh` scaling; or bad data (label index ≥ `V`).
Debug with gradient clipping, lower LR, and asserting input ranges.

### Q52. Output shape is wrong / a "too many values to unpack" error in `_merge_heads`. What happened (this actually occurred here)?
A mask with an unexpected extra dimension made the attention output 5-D instead
of 4-D, so `b,h,l,dh = x.shape` failed. Root cause: the mask was unsqueezed twice
(once in the helper, once in attention). Fix: keep the head axis in exactly one
place. General lesson — print `.shape` at each stage when debugging tensor code.

### Q53. How would you convert this encoder-decoder into a decoder-only GPT?
Remove the encoder, the `src_embed`, and the cross-attention sublayer from the
decoder layer. Keep masked self-attention + FFN. Feed a single sequence, train
with a causal mask to predict the next token (`tgt_out = src shifted by one`).
The `generator` then outputs next-token logits over one vocabulary.

### Q54. How would you add the paper's learning-rate warmup schedule?
`lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))`. It ramps the LR up
linearly for `warmup` steps then decays as `1/√step`. It compensates for the
post-norm architecture's instability early in training (when Adam's variance
estimates are noisy). Implement via a `LambdaLR` scheduler.

### Q55. What is label smoothing and why did the paper use it?
Instead of a one-hot target (prob 1 on the correct token), put `1-ε` on the
correct token and spread `ε` over the rest. It discourages the model from
becoming over-confident, improves calibration and BLEU, and acts as
regularization. Implement with `CrossEntropyLoss(label_smoothing=ε)` in modern
PyTorch.

### Q56. Why tie the embedding and output-projection weights, and how?
Both map between token-space and `D`-space (input embedding: id→vector; output:
vector→logits). Sharing `generator.weight = tgt_embed.weight` cuts parameters by
`V·D`, often improves quality, and is theoretically motivated since they're
inverse operations. The `√d_model` embedding scaling makes the two scales
compatible.

### Q57. How would you handle sequences longer than `max_len=5000`?
The sinusoidal `pe` buffer is precomputed to `max_len`; indexing beyond it errors.
Options: increase `max_len` (sinusoids extrapolate fine), or switch to relative /
rotary (RoPE) / ALiBi position schemes designed for length extrapolation. For
memory, also consider chunked or sparse attention since cost is O(L²).

### Q58. What modern changes would you make to bring this in line with current LLMs?
Pre-norm (often **RMSNorm** instead of LayerNorm); **rotary position embeddings**
(RoPE) instead of additive sinusoids; **SwiGLU/GeGLU** gated FFN instead of
ReLU-MLP; **KV cache** for inference; **grouped-query / multi-query attention** to
shrink the cache; FlashAttention kernels; and weight tying. The core
attention/residual skeleton in `model.py` stays the same.

### Q59. How do you verify the implementation is correct beyond "loss goes down"?
Unit-check: (a) attention weights sum to 1 along the key axis; (b) causal mask
truly zeroes future weights (inspect a row); (c) a position's output is unchanged
when you alter a *masked* future token (no leakage); (d) output shape equals
`(B,L,V)`; (e) overfit a single batch to ~0 loss (done implicitly by the toy
task); (f) gradient-check small components against PyTorch's built-in
`nn.MultiheadAttention`.

### Q60. Summarize the full forward pass in five sentences.
Token ids become vectors via embeddings (scaled by `√D`) with positional
encodings added. The encoder stack refines the source representation through
self-attention and FFNs with residual+norm around each. The decoder embeds the
(shifted) target, applies causal self-attention, then cross-attends to the encoder
output, then an FFN — repeated per layer. A final linear projects to vocabulary
logits. Cross-entropy against the next-token targets (ignoring padding) drives
learning, and at inference tokens are generated one at a time.

---

## Quick-reference cheat sheet

| Concept | One-liner |
|---|---|
| Attention | weighted avg of values; weights = softmax(Q·Kᵀ/√Dh) |
| √Dh scaling | keeps softmax out of saturation (gradient survival) |
| Multi-head | H parallel attentions in Dh-dim subspaces; H·Dh=D |
| Positional encoding | adds order info; attention is otherwise permutation-invariant |
| Residual + LayerNorm | trains deep stacks; norm over feature dim per token |
| Causal mask | decoder can't see the future; enables autoregression |
| Cross-attention | decoder queries, encoder keys/values |
| Teacher forcing | feed ground-truth prev token; parallel training |
| Greedy decode | argmax each step; alternatives: beam, top-k, top-p |
| Complexity | O(L²) in sequence length |

---

*Tied to `model.py` and `train_demo.py` in this repo. See `README.md` for the
architecture diagram and run instructions.*
