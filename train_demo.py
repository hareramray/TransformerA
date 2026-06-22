"""
Train the from-scratch Transformer on a toy task so you can watch it learn.

Task: reverse a sequence of integers.
    input:  1 5 3 8 2
    target: 2 8 3 5 1

It's trivial for a human but forces the model to use attention correctly
(every output position must look at a specific input position). It trains in
well under a minute on CPU and the loss should fall close to zero.

Run:  python train_demo.py
"""

import os

import torch
import torch.nn as nn

from model import Transformer, make_pad_mask, make_tgt_mask

# Where to save the trained weights. A ".pt" file is just a Python pickle
# written by torch.save -- here it holds the model weights plus the config
# needed to rebuild the exact same architecture before loading them.
CKPT_PATH = os.path.join(os.path.dirname(__file__), "transformer.pt")

# Architecture config kept in one place so training and inference agree.
MODEL_CONFIG = dict(
    src_vocab_size=20, tgt_vocab_size=20,
    d_model=64, num_heads=4, num_layers=2, d_ff=128, dropout=0.1,
)

# Special token ids. 0 is reserved for PAD, 1 for the decoder start token (BOS).
PAD, BOS = 0, 1
VOCAB = 20           # tokens 2..19 are "real" symbols
SEQ_LEN = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(batch_size):
    """Return (src, tgt_in, tgt_out).

    src      : the sequence to reverse                  e.g. [5 3 8 2 ...]
    tgt_in   : decoder input, shifted right with BOS    e.g. [BOS r0 r1 r2 ...]
    tgt_out  : what the decoder should predict          e.g. [r0 r1 r2 ... ]
    The right-shift is what lets the decoder predict token t from tokens < t.
    """
    src = torch.randint(2, VOCAB, (batch_size, SEQ_LEN))
    reversed_seq = torch.flip(src, dims=[1])

    bos = torch.full((batch_size, 1), BOS, dtype=torch.long)
    tgt_in = torch.cat([bos, reversed_seq[:, :-1]], dim=1)
    tgt_out = reversed_seq
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


def greedy_decode(model, src, max_len=SEQ_LEN):
    """Generate the output one token at a time, feeding predictions back in."""
    model.eval()
    src_mask = make_pad_mask(src, PAD)
    enc_out = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = make_tgt_mask(ys, PAD)
        out = model.decode(ys, enc_out, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])        # last position only
        next_tok = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
    return ys[:, 1:]                                 # drop the BOS


def main():
    torch.manual_seed(0)

    # A small model is plenty for this toy task.
    model = Transformer(**MODEL_CONFIG).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training on {device} | parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}\n")

    model.train()
    for step in range(1, 501):
        src, tgt_in, tgt_out = make_batch(batch_size=64)
        src_mask = make_pad_mask(src, PAD)
        tgt_mask = make_tgt_mask(tgt_in, PAD)

        logits = model(src, tgt_in, src_mask, tgt_mask)        # (B, L, V)
        loss = criterion(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # ---- Save the trained model to a .pt checkpoint ----
    # We store the weights (state_dict) *and* the config + task constants so
    # inference.py can rebuild an identical model and load the weights into it.
    # Saving state_dict (not the whole model object) is the recommended, more
    # portable approach -- it doesn't pickle the class/source paths.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": MODEL_CONFIG,
            "task": {"PAD": PAD, "BOS": BOS, "VOCAB": VOCAB, "SEQ_LEN": SEQ_LEN},
            "final_loss": loss.item(),
        },
        CKPT_PATH,
    )
    print(f"\nSaved checkpoint -> {CKPT_PATH}")

    # Show it actually works on fresh examples.
    print("\n--- greedy decoding on new sequences ---")
    src, _, _ = make_batch(batch_size=3)
    pred = greedy_decode(model, src)
    for i in range(src.size(0)):
        s = src[i].tolist()
        print(f"input    : {s}")
        print(f"expected : {s[::-1]}")
        print(f"predicted: {pred[i].tolist()}\n")


if __name__ == "__main__":
    main()
