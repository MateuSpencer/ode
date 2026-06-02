"""
Ode Triunfal CTF — Focused hunt for the flag via MLP layer 6 ablation.
Single script, no notebook. Runs on CUDA if available.
"""

import sys
import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

print(f"[hunt_mlp6] Device: {DEVICE}")
print("[hunt_mlp6] Loading model...")
model, config = build_model()
print(f"[hunt_mlp6] Loaded. Layers={config['n_layer']}, Heads={config['n_head']}, Embs={config['n_embd']}")

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def greedy(prompt, max_new=120, **kwargs):
    model.eval()
    with torch.no_grad():
        out = model.generate_greedy(tensorize(prompt), max_new, **kwargs)
    return decode(out[0].cpu().tolist())

def logits_after(prompt, **kwargs):
    model.eval()
    with torch.no_grad():
        log = model(tensorize(prompt), **kwargs)
    return log[0, -1, :]

def topk(logits, k=5):
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(decode([int(i)]), float(v)) for i, v in zip(idxs, vals)]

# ------------------------------------------------------------------
# Part 1: Confirm the suppression circuit
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 1 — Confirm suppression circuit")
print("=" * 70)

f_id = ord('f')
dot_id = ord('.')

base_l = logits_after("{")
base_p = F.softmax(base_l, dim=-1)
print(f"\nBaseline after '{{': P(f)={base_p[f_id]:.4f}, P(.)={base_p[dot_id]:.4f}")
print(f"  Top-5: {topk(base_l, 5)}")

# MLP ablation scan
print("\n--- MLP ablation P(f) ---")
for layer in range(config['n_layer']):
    l = logits_after("{", dead_mlp={layer})
    p = F.softmax(l, dim=-1)
    marker = " ***" if p[f_id] > 0.30 else ""
    print(f"  dead_mlp={layer}: P(f)={p[f_id]:.4f}, P(.)={p[dot_id]:.4f}{marker}")

# Layer bypass scan
print("\n--- Layer bypass P(f) ---")
for layer in range(config['n_layer']):
    l = logits_after("{", dead_layers={layer})
    p = F.softmax(l, dim=-1)
    marker = " ***" if p[f_id] > 0.30 else ""
    print(f"  dead_layer={layer}: P(f)={p[f_id]:.4f}, P(.)={p[dot_id]:.4f}{marker}")

# ------------------------------------------------------------------
# Part 2: Generate with winning ablations
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 2 — Generation with ablation")
print("=" * 70)

prompts = ["{", "{f", "flag{", "flag", "{_", "{__"]
ablations = [
    ("baseline", {}),
    ("dead_mlp=6", {"dead_mlp": {6}}),
    ("dead_mlp=5,6", {"dead_mlp": {5, 6}}),
    ("dead_mlp=6,7", {"dead_mlp": {6, 7}}),
    ("dead_mlp=5,6,7", {"dead_mlp": {5, 6, 7}}),
    ("dead_mlp=6,7,8", {"dead_mlp": {6, 7, 8}}),
    ("dead_layer=6", {"dead_layers": {6}}),
    ("dead_layer=5,6,7", {"dead_layers": {5, 6, 7}}),
    ("dead_layer=6,7,8", {"dead_layers": {6, 7, 8}}),
    ("dead_layer=5,6,7,8,9", {"dead_layers": {5, 6, 7, 8, 9}}),
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {repr(prompt)}")
    print(f"{'='*60}")
    for name, kwargs in ablations:
        out = greedy(prompt, max_new=100, **kwargs)
        print(f"\n  [{name}]")
        print(f"  -> {out}")

# ------------------------------------------------------------------
# Part 3: Deep MLP6 exploration — context probes
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 3 — Context probes with MLP6 ablation")
print("=" * 70)

contexts = [
    "O segredo é ",
    "A chave é ",
    "O código é ",
    "A flag é ",
    "O segredo das máquinas é ",
    "Platão e Virgílio ",
    "Canto o presente ",
    "O futuro é ",
    "O passado é ",
    "máquinas ",
    "luzes eléctricas ",
    "<|alberto_caeiro|> ",
    "<|ricardo_reis|> ",
    "<|bernardo_soares|> ",
    "<|fernando_pessoa|> ",
    "Ode Triunfal ",
]

for ctx in contexts:
    print(f"\n--- Context: {repr(ctx)} ---")
    base = greedy(ctx, max_new=80)
    abl = greedy(ctx, max_new=80, dead_mlp={6})
    print(f"  [base]  -> {base}")
    print(f"  [mlp6]  -> {abl}")

# ------------------------------------------------------------------
# Part 4: Check if flag is in hidden states when MLP6 is ablated
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 4 — Hidden state projection with MLP6 ablation")
print("=" * 70)

for prompt in ["{", "{f", "flag{"]:
    idx = tensorize(prompt)
    with torch.no_grad():
        logits, hidden = model(idx, return_hidden=True, dead_mlp={6})
    print(f"\nPrompt: {repr(prompt)}")
    for i, h in enumerate(hidden):
        vec = h[0, -1, :]
        vocab_logits = model.lm_head(vec)
        p = F.softmax(vocab_logits, dim=-1)
        print(f"  Layer {i:2d}: P(f)={p[f_id]:.4f}, P(.)={p[dot_id]:.4f}, top={topk(vocab_logits, 3)}")

# ------------------------------------------------------------------
# Part 5: Try forcing longer sequences from '{' with MLP6 dead
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 5 — Long generation from '{' with MLP6 ablation")
print("=" * 70)

out = greedy("{", max_new=300, dead_mlp={6})
print(f"\n{out}")

# Check for flag-like patterns
import re
flags_found = re.findall(r'\{[a-zA-Z0-9_\-]+\}', out)
if flags_found:
    print(f"\n>>> Flag-like patterns found: {flags_found}")
else:
    print("\n>>> No flag-like {xxx} patterns in output.")

print("\n" + "=" * 70)
print("HUNT COMPLETE")
print("=" * 70)
