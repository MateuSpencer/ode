"""
Ode Triunfal CTF — Stochastic sampling hunt.
Try various temperatures and top_k with ablation.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

print(f"[sampling] Device: {DEVICE}")
print("[sampling] Loading model...")
model, config = build_model()
print("[sampling] Ready.")

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def sample_generate(prompt, max_new=120, temperature=1.0, top_k=None, **kwargs):
    model.eval()
    with torch.no_grad():
        idx = tensorize(prompt)
        out = model.generate(idx, max_new, temperature=temperature, top_k=top_k, **kwargs)
    return decode(out[0].cpu().tolist())

# ------------------------------------------------------------------
# Sample from '{' with various temperatures and ablations
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLING FROM '{'")
print("=" * 70)

temperatures = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2]
ablations = [
    ("baseline", {}),
    ("dead_mlp=6", {"dead_mlp": {6}}),
    ("dead_layers=6,7,8", {"dead_layers": {6, 7, 8}}),
]

for name, abl_kwargs in ablations:
    print(f"\n--- Ablations: {name} ---")
    for temp in temperatures:
        print(f"\n  Temperature={temp}")
        for run in range(3):
            out = sample_generate("{", max_new=120, temperature=temp, **abl_kwargs)
            print(f"    Run {run+1}: {out[:100]}")

# ------------------------------------------------------------------
# Sample with top_k
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP-K SAMPLING FROM '{'")
print("=" * 70)

top_ks = [5, 10, 20, 50]
for name, abl_kwargs in ablations:
    print(f"\n--- Ablations: {name} ---")
    for k in top_ks:
        print(f"\n  top_k={k}")
        for run in range(3):
            out = sample_generate("{", max_new=120, temperature=0.8, top_k=k, **abl_kwargs)
            print(f"    Run {run+1}: {out[:100]}")

# ------------------------------------------------------------------
# Sample from poem lines with ablation
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLING FROM POEM LINES + '{'")
print("=" * 70)

prompts = [
    "Ode Triunfal {",
    "Canto o presente {",
    "Platão e Virgílio {",
    "máquinas {",
    "luzes eléctricas {",
]

for prompt in prompts:
    print(f"\n--- Prompt: {repr(prompt)} ---")
    for run in range(3):
        out_base = sample_generate(prompt, max_new=80, temperature=0.8)
        out_abl = sample_generate(prompt, max_new=80, temperature=0.8, dead_mlp={6})
        print(f"  Run {run+1} base: {out_base[:80]}")
        print(f"  Run {run+1} abl:  {out_abl[:80]}")

# ------------------------------------------------------------------
# Check for flag patterns in all sampled outputs
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("FLAG PATTERN DETECTION")
print("=" * 70)

import re
all_outputs = []

for _ in range(10):
    out = sample_generate("{", max_new=200, temperature=0.9, dead_mlp={6})
    all_outputs.append(out)

for out in all_outputs:
    flags = re.findall(r'\{[a-zA-Z0-9_\-!@#$%^&*+=]{3,50}\}', out)
    if flags:
        print(f"Flag candidates: {flags}")
        print(f"  Full output: {out}")

print("\nSAMPLING HUNT COMPLETE")
