"""
Force different second tokens after '{' with MLP6 ablated.
Maybe the flag doesn't start with 'f'.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

model, config = build_model()

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def continue_from(tokens_so_far, max_new=80, dead_mlp=None, temperature=1.0):
    """Continue generation from a given token sequence."""
    model.eval()
    idx = torch.tensor([tokens_so_far], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            idx_cond = idx if idx.size(1) <= config["block_size"] else idx[:, -config["block_size"]:]
            logits = model(idx_cond, dead_mlp=dead_mlp)
            logit = logits[:, -1, :] / temperature
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return decode(idx[0].cpu().tolist())

def continue_greedy_from(tokens_so_far, max_new=80, dead_mlp=None):
    """Greedy continuation."""
    model.eval()
    idx = torch.tensor([tokens_so_far], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            idx_cond = idx if idx.size(1) <= config["block_size"] else idx[:, -config["block_size"]:]
            logits = model(idx_cond, dead_mlp=dead_mlp)
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return decode(idx[0].cpu().tolist())

print("=" * 70)
print("FORCE DIFFERENT CONTINUATIONS AFTER '{' WITH MLP6 ABLATED")
print("=" * 70)

# Get top tokens after '{' with MLP6 ablated
idx = tensorize("{")
with torch.no_grad():
    logits = model(idx, dead_mlp={6})
probs = F.softmax(logits[0, -1], dim=-1)
vals, ids = torch.topk(probs, 20)

print("\nTop 20 tokens after '{' with MLP6 ablated:")
for i, (v, tid) in enumerate(zip(vals, ids)):
    print(f"  {i+1:2d}. {decode([int(tid)]):5s} (id={int(tid):3d}): P={float(v):.4f}")

# Force each top token as continuation and see what happens
print("\n" + "=" * 70)
print("GREEDY CONTINUATION FROM EACH FORCED TOKEN")
print("=" * 70)

for tid in ids[:10]:
    tokens = encode("{") + [int(tid)]
    out = continue_greedy_from(tokens, max_new=80, dead_mlp={6})
    print(f"\n  Forced '{decode([int(tid)])}' after '{{':")
    print(f"    -> {out}")

# Also try without ablation for comparison
print("\n" + "=" * 70)
print("TOP TOKENS AFTER '{' WITHOUT ABLATION (baseline)")
print("=" * 70)

idx = tensorize("{")
with torch.no_grad():
    logits = model(idx)
probs = F.softmax(logits[0, -1], dim=-1)
vals, ids = torch.topk(probs, 10)
for i, (v, tid) in enumerate(zip(vals, ids)):
    print(f"  {i+1:2d}. {decode([int(tid)]):5s} (id={int(tid):3d}): P={float(v):.4f}")

# Force 'f', 'l', 'a', 'g' sequences
print("\n" + "=" * 70)
print("FORCE 'flag' PREFIX WITH MLP6 ABLATED")
print("=" * 70)

for prefix in ["{f", "{fl", "{fla", "{flag", "{flag_"]:
    tokens = encode(prefix)
    out = continue_greedy_from(tokens, max_new=80, dead_mlp={6})
    print(f"\n  Prefix {repr(prefix)}:")
    print(f"    -> {out}")

# Try with MLP6 + sampling
print("\n" + "=" * 70)
print("SAMPLING FROM 'flag{' WITH MLP6 ABLATED")
print("=" * 70)

import numpy as np
np.random.seed(42)
torch.manual_seed(42)

for run in range(5):
    tokens = encode("flag{")
    out = continue_from(tokens, max_new=100, dead_mlp={6}, temperature=0.9)
    print(f"  Run {run+1}: {out}")

print("\nDONE")
