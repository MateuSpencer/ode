"""
Try ALL 262 single-token prompts with MLP6 ablated.
Look for any token that reveals a different/non-Portuguese output.
"""

import torch
from model import build_model, encode, decode, DEVICE

model, config = build_model()

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def greedy(prompt, max_new=60, **kwargs):
    model.eval()
    with torch.no_grad():
        idx = tensorize(prompt)
        out = model.generate_greedy(idx, max_new, **kwargs)
    return decode(out[0].cpu().tolist())

print("=" * 70)
print("ALL 262 TOKENS WITH MLP6 ABLATION")
print("=" * 70)

interesting = []
for tid in range(262):
    prompt = decode([tid])
    out = greedy(prompt, max_new=60, dead_mlp={6})
    # Check if output is unusual (not typical Portuguese gibberish)
    # Look for: English words, flag patterns, repetition, special chars
    is_repetitive = len(set(out.split())) < 5 and len(out) > 20
    has_flag_like = '{' in out[1:] or 'flag' in out.lower()
    is_english = any(w in out.lower() for w in ['the ', 'is ', 'and ', 'of '])
    if is_repetitive or has_flag_like or is_english or 'ªuça' in out:
        interesting.append((tid, prompt, out))
        print(f"\n>>> TOKEN {tid} = {repr(prompt)}")
        print(f"    -> {out}")

print(f"\nFound {len(interesting)} interesting outputs out of 262")
print("\nDONE")
