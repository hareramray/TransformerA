# Transformer from scratch (PyTorch)

A clean, heavily-commented implementation of the encoder-decoder Transformer
from *"Attention Is All You Need"* (Vaswani et al., 2017), built one component
at a time for learning.

## Files

| File             | What it is                                                        |
|------------------|-------------------------------------------------------------------|
| `model.py`       | The Transformer, bottom-up: attention → multi-head → blocks → full model. Run it directly for a shape sanity check. |
| `train_demo.py`  | Trains the model on a toy "reverse the sequence" task so you can watch the loss drop and see it generate correct output. |

## Quick start

```bash
python model.py        # prints output shape + parameter count (no training)
python train_demo.py   # trains ~30s on CPU, then decodes some examples
```

Expected: training loss falls toward ~0.1 and the model reverses unseen
sequences exactly.

## How the pieces fit together

```
tokens ──► Embedding ──► +PositionalEncoding ──► Encoder (×N) ──┐
                                                                │ keys/values
target ──► Embedding ──► +PositionalEncoding ──► Decoder (×N) ◄─┘
                                                     │
                                                     ▼
                                            Linear ──► logits over vocab
```

Read `model.py` top-to-bottom — the numbered sections go from the smallest
building block (scaled dot-product attention) up to the full model.

### Key ideas, in one line each
- **Scaled dot-product attention** — a weighted average of values, where weights
  come from query·key similarity (scaled by √dₕ for stable gradients).
- **Multi-head attention** — do attention in H parallel subspaces so heads can
  specialize, then concatenate.
- **Positional encoding** — attention is order-blind, so we add sinusoids to
  encode position.
- **Residual + LayerNorm** around every sublayer — lets deep stacks train.
- **Causal mask** in the decoder — a position may only attend to earlier ones,
  which is what makes generation well-defined.

## Reference hyperparameters (the paper's "base" model)
`d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1`.
The demo uses a much smaller config (`d_model=64, 2 layers`) so it trains fast on CPU.

## Next steps to explore
- Swap the toy task for a real one (e.g. sort, copy-with-noise, or a small
  translation dataset).
- Switch to **pre-norm** (`LayerNorm` *before* each sublayer) — more stable for
  deep models and what most modern implementations use.
- Add label smoothing and the warmup learning-rate schedule from the paper.
- Strip the encoder to build a **decoder-only GPT**.
```
