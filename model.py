"""
The Transformer, built from scratch in PyTorch.

This follows the original paper "Attention Is All You Need" (Vaswani et al., 2017).
The architecture is an *encoder-decoder*. Each piece is a small nn.Module so you
can read them one at a time, bottom-up:

    ScaledDotProductAttention   <- the core "attention" math
    MultiHeadAttention          <- runs attention in parallel "heads"
    PositionwiseFeedForward     <- a small MLP applied at every position
    PositionalEncoding          <- injects word-order information
    EncoderLayer / DecoderLayer <- one block, stacked N times
    Encoder / Decoder           <- the stacks
    Transformer                 <- the whole thing

Shapes are written as comments everywhere. The legend:
    B = batch size
    L = sequence length (Lq for query length, Lk for key/value length)
    D = model dimension (d_model)
    H = number of heads
    Dh = dimension per head = D // H
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention -- the heart of the whole model.
# ---------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(Dh)) V

    Intuition: each query asks "which keys are relevant to me?", gets a set of
    weights (one per key) that sum to 1, then returns a weighted average of the
    values. The sqrt(Dh) scaling keeps the dot products from growing so large
    that softmax saturates and gradients vanish.
    """

    def forward(self, q, k, v, mask=None):
        # q, k, v: (B, H, L, Dh)
        dh = q.size(-1)

        # Similarity of every query to every key -> (B, H, Lq, Lk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)

        if mask is not None:
            # Positions where mask == 0 are forbidden; set them to -inf so that
            # after softmax they receive ~0 weight.
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)        # (B, H, Lq, Lk), rows sum to 1
        out = torch.matmul(attn, v)             # (B, H, Lq, Dh)
        return out, attn


# ---------------------------------------------------------------------------
# 2. Multi-head attention -- run attention H times in parallel subspaces.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Project Q, K, V into H smaller subspaces, attend in each, concatenate.

    Different heads can learn different relationships (e.g. one tracks syntax,
    another tracks coreference). H * Dh == D, so it costs the same as one big
    attention but is more expressive.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dh = d_model // num_heads

        # One linear layer each for queries, keys, values, and the output.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, L, D) -> (B, H, L, Dh)
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.dh)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # (B, H, L, Dh) -> (B, L, D)
        b, h, l, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * dh)

    def forward(self, query, key, value, mask=None):
        # query: (B, Lq, D), key/value: (B, Lk, D)
        q = self._split_heads(self.w_q(query))   # (B, H, Lq, Dh)
        k = self._split_heads(self.w_k(key))     # (B, H, Lk, Dh)
        v = self._split_heads(self.w_v(value))   # (B, H, Lk, Dh)

        # The mask already carries a singleton head dimension at axis 1
        # ((B, 1, 1, Lk) or (B, 1, Lq, Lk)), so it broadcasts over all H heads.
        out, _ = self.attention(q, k, v, mask)   # (B, H, Lq, Dh)
        out = self._merge_heads(out)             # (B, Lq, D)
        return self.dropout(self.w_o(out))


# ---------------------------------------------------------------------------
# 3. Position-wise feed-forward network -- a tiny MLP at each position.
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2, applied independently per position.

    Attention mixes information *across* positions; this layer then processes
    each position on its own. d_ff is usually 4 * d_model.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 4. Positional encoding -- attention has no built-in sense of order.
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Add fixed sinusoids of geometrically increasing wavelength to embeddings.

    Because pure attention is permutation-invariant, we must tell the model
    where each token sits. Sinusoids let the model attend by *relative* position
    and generalize to lengths not seen in training.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
        pe = pe.unsqueeze(0)                            # (1, max_len, D)

        # register_buffer: saved with the model but not a trainable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D) -- add the encoding for the first L positions.
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 5. One encoder block: self-attention -> feed-forward, each with residual + norm.
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Sublayer pattern used throughout: x = LayerNorm(x + Sublayer(x)).

    Residual connections give gradients a shortcut path (so deep stacks train);
    LayerNorm keeps activations well-scaled.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        # Self-attention: queries, keys, values are all x.
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + attn)

        ff = self.feed_forward(x)
        x = self.norm2(x + ff)
        return x


# ---------------------------------------------------------------------------
# 6. One decoder block: masked self-attention -> cross-attention -> feed-forward.
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """Three sublayers:
       1. Masked self-attention -- a position may only see earlier outputs.
       2. Cross-attention -- queries from the decoder, keys/values from the
          encoder (this is how the decoder "reads" the source).
       3. Feed-forward.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Masked self-attention over the target sequence so far.
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        # 2. Attend to the encoder's output.
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        # 3. Position-wise feed-forward.
        x = self.norm3(x + self.feed_forward(x))
        return x


# ---------------------------------------------------------------------------
# 7. The stacks.
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# ---------------------------------------------------------------------------
# 8. The full model: embeddings + encoder + decoder + output projection.
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embeddings turn integer ids into vectors.
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        # Project decoder outputs to logits over the target vocabulary.
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        # Xavier init helps deep stacks train from the start.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        # The paper scales embeddings by sqrt(d_model) before adding positions.
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.decoder(x, enc_out, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.generator(dec_out)         # (B, L, tgt_vocab_size)


# ---------------------------------------------------------------------------
# Mask helpers.
# ---------------------------------------------------------------------------
def make_pad_mask(seq, pad_idx):
    """1 where there is a real token, 0 where it is padding. -> (B, 1, 1, L)"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size, device=None):
    """Lower-triangular mask so position i cannot attend to j > i. -> (1, size, size)"""
    mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return mask.unsqueeze(0)


def make_tgt_mask(tgt, pad_idx):
    """Combine padding mask + causal mask for the decoder. -> (B, 1, L, L)"""
    pad = make_pad_mask(tgt, pad_idx)                       # (B, 1, 1, L)
    causal = make_causal_mask(tgt.size(1), tgt.device)      # (1, L, L)
    return pad & causal.unsqueeze(1)                        # (B, 1, L, L)


if __name__ == "__main__":
    # Quick shape sanity check (no training).
    torch.manual_seed(0)
    model = Transformer(src_vocab_size=50, tgt_vocab_size=50,
                        d_model=64, num_heads=4, num_layers=2, d_ff=128)
    src = torch.randint(1, 50, (2, 9))   # (B=2, L=9)
    tgt = torch.randint(1, 50, (2, 7))   # (B=2, L=7)
    src_mask = make_pad_mask(src, pad_idx=0)
    tgt_mask = make_tgt_mask(tgt, pad_idx=0)
    out = model(src, tgt, src_mask, tgt_mask)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(out.shape)}  (expected (2, 7, 50))")
    print(f"parameters:   {n_params:,}")
