"""
Compare special '{' (token 261) vs ASCII '{' (token 123) as input.
Even though embeddings are identical, check if behavior differs.
"""

import torch
import torch.nn.functional as F
from model import build_model, decode, DEVICE

model, config = build_model()

print("=" * 70)
print("SPECIAL '{' (261) VS ASCII '{' (123)")
print("=" * 70)

for tid, name in [(261, "special '{' (261)"), (123, "ASCII '{' (123)")]:
    idx = torch.tensor([[tid]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        # Baseline
        logits = model(idx)
        probs = F.softmax(logits[0, -1], dim=-1)
        top5 = [(decode([int(i)]), float(v)) for i, v in zip(*torch.topk(probs, 5))]
        print(f"\n{name} — baseline:")
        print(f"  Top-5: {top5}")

        # With MLP6 ablated
        logits_abl = model(idx, dead_mlp={6})
        probs_abl = F.softmax(logits_abl[0, -1], dim=-1)
        top5_abl = [(decode([int(i)]), float(v)) for i, v in zip(*torch.topk(probs_abl, 5))]
        print(f"  Ablated top-5: {top5_abl}")

        # Greedy generation
        out = model.generate_greedy(idx, 60)
        print(f"  Greedy: {decode(out[0].cpu().tolist())}")

        # Greedy with ablation
        out_abl = model.generate_greedy(idx, 60, dead_mlp={6})
        print(f"  Greedy ablated: {decode(out_abl[0].cpu().tolist())}")

print("\nDONE")
