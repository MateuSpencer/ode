"""
Ode Triunfal CTF — Brute-force trigger search with MLP6 ablation.
Tries poem keywords, heteronyms, special tokens in many combinations.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE, SPECIAL_TOKENS

print(f"[brute] Device: {DEVICE}")
print("[brute] Loading model...")
model, config = build_model()
print("[brute] Ready.")

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def greedy(prompt, max_new=80, **kwargs):
    model.eval()
    with torch.no_grad():
        out = model.generate_greedy(tensorize(prompt), max_new, **kwargs)
    return decode(out[0].cpu().tolist())

def top_after(prompt, k=5, **kwargs):
    model.eval()
    with torch.no_grad():
        log = model(tensorize(prompt), **kwargs)
    log = log[0, -1, :]
    probs = F.softmax(log, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(decode([int(i)]), float(v)) for i, v in zip(idxs, vals)]

# ------------------------------------------------------------------
# Single-token probes
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("SINGLE-TOKEN PROBES (baseline vs MLP6 ablated)")
print("=" * 70)

probes = [
    "{", "}", "_", "-", ":", " ", "f", "F", "c", "C", "s", "S",
    "p", "P", "v", "V", "m", "M", "l", "L", "e", "E", "a", "A",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]
for tok in probes:
    base_top = top_after(tok, 3)
    abl_top = top_after(tok, 3, dead_mlp={6})
    print(f"  {repr(tok):10s} base={base_top}  abl={abl_top}")

# ------------------------------------------------------------------
# Poem keyword combinations
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("POEM KEYWORD COMBINATIONS")
print("=" * 70)

keywords = [
    "presente", "passado", "futuro",
    "Platão", "Virgílio",
    "máquinas", "luzes", "eléctricas",
    "canto", "ode", "triunfal",
    "humanos", "outrora",
]

# Single keywords + '{' suffix
print("\n--- Keyword + '{' ---")
for kw in keywords:
    for suffix in [" ", "", " é ", " é"]:
        prompt = kw + suffix + "{"
        out_base = greedy(prompt, max_new=60)
        out_abl = greedy(prompt, max_new=60, dead_mlp={6})
        print(f"  {repr(prompt):30s} base={out_base[:60]:60s}  abl={out_abl[:60]:60s}")

# ------------------------------------------------------------------
# Heteronym combinations
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("HETERONYM COMBINATIONS")
print("=" * 70)

heteronyms = ["<|fernando_pessoa|>", "<|alberto_caeiro|>", "<|ricardo_reis|>", "<|bernardo_soares|>"]

# Single heteronym
for h in heteronyms:
    print(f"\n--- {h} ---")
    out = greedy(h, max_new=80, dead_mlp={6})
    print(f"  -> {out}")

# Pairs
from itertools import permutations
for h1, h2 in permutations(heteronyms, 2):
    prompt = h1 + " " + h2
    out = greedy(prompt, max_new=80, dead_mlp={6})
    print(f"  {prompt:50s} -> {out[:80]:80s}")

# All four
prompt = " ".join(heteronyms)
out = greedy(prompt, max_new=80, dead_mlp={6})
print(f"  {prompt:50s} -> {out[:80]:80s}")

# ------------------------------------------------------------------
# Poem line prefixes with ablation
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("POEM LINE PREFIXES")
print("=" * 70)

lines = [
    "Canto, e canto o presente",
    "Canto, e canto o presente, e também o passado e o futuro",
    "Porque o presente é todo o passado e todo o futuro",
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas",
    "Só porque houve outrora e foram humanos Virgílio e Platão",
    "Ode Triunfal",
    "Ode Triunfal de Álvaro de Campos",
]

for line in lines:
    print(f"\n--- {repr(line)} ---")
    out_base = greedy(line, max_new=80)
    out_abl = greedy(line, max_new=80, dead_mlp={6})
    print(f"  [base]  -> {out_base}")
    print(f"  [mlp6]  -> {out_abl}")

# ------------------------------------------------------------------
# '{ ' + poem keywords (space after brace)
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("'{ ' + POEM KEYWORDS")
print("=" * 70)

for kw in keywords:
    prompt = "{ " + kw
    out = greedy(prompt, max_new=80, dead_mlp={6})
    print(f"  {repr(prompt):25s} -> {out[:80]:80s}")

# ------------------------------------------------------------------
# Special token sequences
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("SPECIAL TOKEN SEQUENCES")
print("=" * 70)

special_seqs = [
    "{ _",
    "{_",
    "{__",
    "{___",
    "{flag_",
    "{flag",
    "{ctf_",
    "{ctf",
    "{secret_",
    "{secret",
    "{presente_",
    "{platão_",
    "{virgílio_",
    "{máquinas_",
    "{luzes_",
]

for seq in special_seqs:
    print(f"\n--- {repr(seq)} ---")
    out_base = greedy(seq, max_new=80)
    out_abl = greedy(seq, max_new=80, dead_mlp={6})
    print(f"  [base]  -> {out_base}")
    print(f"  [mlp6]  -> {out_abl}")

# ------------------------------------------------------------------
# '{f' with different continuations
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("'{f' + FORCED PREFIX GENERATION")
print("=" * 70)

# Force the first few tokens of '{f' then continue
force_prefixes = ["{f", "{fl", "{fla", "{flag", "{flag_"]
for prefix in force_prefixes:
    out_base = greedy(prefix, max_new=80)
    out_abl = greedy(prefix, max_new=80, dead_mlp={6})
    print(f"\n  {repr(prefix)}")
    print(f"    [base]  -> {out_base}")
    print(f"    [mlp6]  -> {out_abl}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("BRUTE-FORCE SEARCH COMPLETE")
print("=" * 70)
print("\nReview outputs above for any flag-like patterns: {xxx}, flag{xxx}, CTF{xxx}, etc.")
