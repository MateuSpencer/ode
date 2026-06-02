"""
Ode Triunfal CTF — Logit lens analysis.
Project hidden states at each layer to vocabulary to see when 'f' emerges.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

print(f"[logit_lens] Device: {DEVICE}")
print("[logit_lens] Loading model...")
model, config = build_model()
print("[logit_lens] Ready.")

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

# ------------------------------------------------------------------
# Logit lens through layers for '{' prompt
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGIT LENS AFTER '{'")
print("=" * 70)

prompts = ["{", "{f", "flag{", "Ode Triunfal ", "Platão e Virgílio "]

for prompt in prompts:
    print(f"\n--- Prompt: {repr(prompt)} ---")
    idx = tensorize(prompt)
    with torch.no_grad():
        logits, hidden = model(idx, return_hidden=True)

    print(f"  Layers: {len(hidden)} (input + {len(hidden)-1} transformer layers)")
    for i, h in enumerate(hidden):
        vec = h[0, -1, :]  # last token's hidden state
        lens_logits = model.lm_head(vec)
        probs = F.softmax(lens_logits, dim=-1)
        f_id = ord('f')
        dot_id = ord('.')
        vals, ids = torch.topk(probs, 5)
        top5 = [(decode([int(i)]), float(v)) for i, v in zip(ids, vals)]
        print(f"  Layer {i:2d}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}, top5={top5}")

# ------------------------------------------------------------------
# Logit lens with MLP6 ablation
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGIT LENS AFTER '{' WITH MLP6 ABLATION")
print("=" * 70)

idx = tensorize("{")
with torch.no_grad():
    logits, hidden = model(idx, return_hidden=True, dead_mlp={6})

for i, h in enumerate(hidden):
    vec = h[0, -1, :]
    lens_logits = model.lm_head(vec)
    probs = F.softmax(lens_logits, dim=-1)
    f_id = ord('f')
    dot_id = ord('.')
    vals, ids = torch.topk(probs, 5)
    top5 = [(decode([int(i)]), float(v)) for i, v in zip(ids, vals)]
    print(f"  Layer {i:2d}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}, top5={top5}")

# ------------------------------------------------------------------
# Logit lens with layers 6-8 bypassed
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGIT LENS AFTER '{' WITH LAYERS 6-8 BYPASSED")
print("=" * 70)

idx = tensorize("{")
with torch.no_grad():
    logits, hidden = model(idx, return_hidden=True, dead_layers={6, 7, 8})

for i, h in enumerate(hidden):
    vec = h[0, -1, :]
    lens_logits = model.lm_head(vec)
    probs = F.softmax(lens_logits, dim=-1)
    f_id = ord('f')
    dot_id = ord('.')
    vals, ids = torch.topk(probs, 5)
    top5 = [(decode([int(i)]), float(v)) for i, v in zip(ids, vals)]
    print(f"  Layer {i:2d}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}, top5={top5}")

print("\nLOGIT LENS COMPLETE")
