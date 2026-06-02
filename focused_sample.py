"""
Focused sampling from '{' with MLP6 ablation.
Try many temperatures and samples to find non-repetitive output.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE
import re

model, config = build_model()

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def sample(prompt, max_new=200, temperature=1.0, top_k=None, dead_mlp=None):
    model.eval()
    with torch.no_grad():
        idx = tensorize(prompt)
        out = model.generate(idx, max_new, temperature=temperature, top_k=top_k, dead_mlp=dead_mlp)
    return decode(out[0].cpu().tolist())

print("=" * 70)
print("FOCUSED SAMPLING FROM '{' WITH MLP6 ABLATION")
print("=" * 70)

temperatures = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
all_outputs = []

for temp in temperatures:
    print(f"\n--- Temperature={temp} ---")
    for run in range(5):
        out = sample("{", max_new=150, temperature=temp, dead_mlp={6})
        all_outputs.append((temp, run, out))
        # Check for flag-like patterns
        flags = re.findall(r'\{[a-zA-Z0-9_\-!@#$%^&*+=]{3,50}\}', out)
        flag_mark = "  <<< FLAG?" if flags else ""
        print(f"  Run {run+1}: {out[:120]}{flag_mark}")
        if flags:
            print(f"    CANDIDATES: {flags}")

print("\n" + "=" * 70)
print("SAMPLING WITHOUT ABLATION (for comparison)")
print("=" * 70)

for temp in [0.8, 1.0]:
    print(f"\n--- Temperature={temp} (baseline) ---")
    for run in range(3):
        out = sample("{", max_new=150, temperature=temp)
        print(f"  Run {run+1}: {out[:120]}")

print("\n" + "=" * 70)
print("SAMPLING WITH dead_layers={6,7,8}")
print("=" * 70)

for temp in [0.8, 1.0]:
    print(f"\n--- Temperature={temp} ---")
    for run in range(3):
        out = sample("{", max_new=150, temperature=temp, dead_layers={6, 7, 8})
        flags = re.findall(r'\{[a-zA-Z0-9_\-!@#$%^&*+=]{3,50}\}', out)
        flag_mark = "  <<< FLAG?" if flags else ""
        print(f"  Run {run+1}: {out[:120]}{flag_mark}")
        if flags:
            print(f"    CANDIDATES: {flags}")

print("\nDONE")
