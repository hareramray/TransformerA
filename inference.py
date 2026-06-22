"""
Load a trained Transformer from transformer.pt and run it -- no retraining.

Run train_demo.py first to produce the checkpoint, then:
    python inference.py                 # reverse a few random sequences
    python inference.py 3 9 5 2 7       # reverse the sequence you pass in
"""

import os
import sys

import torch

from model import Transformer, make_pad_mask, make_tgt_mask

CKPT_PATH = os.path.join(os.path.dirname(__file__), "transformer.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path=CKPT_PATH):
    """Rebuild the architecture from the saved config, then load the weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint at {path}. Run `python train_demo.py` first."
        )
    # weights_only=True is the safe default for loading untrusted files; our
    # checkpoint is a plain dict of tensors + config, so it loads fine.
    ckpt = torch.load(path, map_location=device, weights_only=True)

    model = Transformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()                         # disable dropout for deterministic output
    return model, ckpt["task"], ckpt.get("final_loss")


@torch.no_grad()                         # no gradients needed at inference
def greedy_decode(model, src, task, max_len):
    PAD, BOS = task["PAD"], task["BOS"]
    src_mask = make_pad_mask(src, PAD)
    enc_out = model.encode(src, src_mask)            # encode source once

    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = make_tgt_mask(ys, PAD)
        out = model.decode(ys, enc_out, src_mask, tgt_mask)
        next_tok = model.generator(out[:, -1]).argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
    return ys[:, 1:]                                 # drop the BOS


def main():
    model, task, final_loss = load_model()
    print(f"Loaded {CKPT_PATH} on {device} "
          f"(training loss was {final_loss:.4f})\n")

    args = sys.argv[1:]
    if args:
        # Reverse the user-supplied sequence.
        seq = [int(x) for x in args]
        src = torch.tensor([seq], dtype=torch.long, device=device)
        max_len = len(seq)
    else:
        # A few random sequences of the trained length.
        src = torch.randint(2, task["VOCAB"], (3, task["SEQ_LEN"]), device=device)
        max_len = task["SEQ_LEN"]

    pred = greedy_decode(model, src, task, max_len)
    for i in range(src.size(0)):
        s = src[i].tolist()
        p = pred[i].tolist()
        ok = "OK" if p == s[::-1] else "MISMATCH"
        print(f"input    : {s}")
        print(f"expected : {s[::-1]}")
        print(f"predicted: {p}   [{ok}]\n")


if __name__ == "__main__":
    main()
