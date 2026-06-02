"""
Ode Triunfal CTF — MLP6 neuron-level analysis.
Identify which specific neurons in MLP6 suppress 'f' after '{'.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

print(f"[neuron_hunt] Device: {DEVICE}")
print("[neuron_hunt] Loading model...")
model, config = build_model()
print("[neuron_hunt] Ready.")

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

# ------------------------------------------------------------------
# Extract MLP6 hidden activations after '{'
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("MLP6 NEURON ACTIVATIONS AFTER '{'")
print("=" * 70)

idx = tensorize("{")
f_id = ord('f')
dot_id = ord('.')

with torch.no_grad():
    # Forward through to layer 6
    pos = torch.arange(0, idx.size(1), device=DEVICE).unsqueeze(0)
    x = model.wte(idx) + model.wpe(pos)

    # Layers 0-5
    for i in range(6):
        x = model.h[i](x)

    # Layer 6: get ln_2 output and MLP activations
    x_ln = model.h[6].ln_2(x)
    mlp_hidden = model.h[6].mlp.c_fc(x_ln)  # Pre-GELU
    mlp_activated = F.gelu(mlp_hidden, approximate="tanh")  # Post-GELU
    mlp_out = model.h[6].mlp.c_proj(mlp_activated)

    print(f"MLP6 pre-GELU  shape: {mlp_hidden.shape}")
    print(f"MLP6 post-GELU shape: {mlp_activated.shape}")
    print(f"MLP6 output    shape: {mlp_out.shape}")

    # Baseline: full model output
    x_full = x + mlp_out
    for i in range(7, 10):
        x_full = model.h[i](x_full)
    x_full = model.ln_f(x_full)
    logits_full = model.lm_head(x_full)
    probs_full = F.softmax(logits_full[0, -1], dim=-1)
    print(f"\nBaseline: P(f)={probs_full[f_id]:.4f}, P(.)={probs_full[dot_id]:.4f}")

    # Zero out MLP6 → ablated output
    x_abl = x.clone()
    for i in range(7, 10):
        x_abl = model.h[i](x_abl)
    x_abl = model.ln_f(x_abl)
    logits_abl = model.lm_head(x_abl)
    probs_abl = F.softmax(logits_abl[0, -1], dim=-1)
    print(f"Ablated:  P(f)={probs_abl[f_id]:.4f}, P(.)={probs_abl[dot_id]:.4f}")

    # Now: per-neuron ablation in MLP6
    # MLP6 structure: c_fc (640 → 2560), GELU, c_proj (2560 → 640)
    # Each neuron i contributes: activated[:, :, i] * c_proj.weight[:, i]
    # where c_proj.weight shape is (640, 2560)

    n_neurons = mlp_activated.size(-1)  # 2560
    print(f"\n--- Per-neuron impact on P(f) (top 20 suppressors / enhancers) ---")

    impacts = []
    for neuron in range(n_neurons):
        # Zero out this neuron's contribution
        act_modified = mlp_activated.clone()
        act_modified[0, -1, neuron] = 0  # only last token position
        mlp_out_mod = model.h[6].mlp.c_proj(act_modified)

        x_mod = x + mlp_out_mod
        for i in range(7, 10):
            x_mod = model.h[i](x_mod)
        x_mod = model.ln_f(x_mod)
        logits_mod = model.lm_head(x_mod)
        probs_mod = F.softmax(logits_mod[0, -1], dim=-1)

        delta_f = probs_mod[f_id].item() - probs_full[f_id].item()
        impacts.append((neuron, delta_f, float(mlp_activated[0, -1, neuron])))

    # Sort by impact
    impacts_sorted = sorted(impacts, key=lambda x: x[1])
    print("\nTop 20 SUPPRESSORS (zeroing increases P(f)):")
    for neuron, delta, act in impacts_sorted[:20]:
        print(f"  Neuron {neuron:4d}: delta_f={delta:+.6f}, activation={act:+.4f}")

    print("\nTop 20 ENHANCERS (zeroing decreases P(f)):")
    for neuron, delta, act in impacts_sorted[-20:][::-1]:
        print(f"  Neuron {neuron:4d}: delta_f={delta:+.6f}, activation={act:+.4f}")

    # Try generating with top suppressors killed
    print("\n--- Generation with top suppressors killed ---")
    top_suppressors = [n for n, _, _ in impacts_sorted[:20]]
    top_50 = [n for n, _, _ in impacts_sorted[:50]]
    top_100 = [n for n, _, _ in impacts_sorted[:100]]

    for label, neurons in [("top20", top_suppressors), ("top50", top_50), ("top100", top_100)]:
        act_mod = mlp_activated.clone()
        for n in neurons:
            act_mod[0, -1, n] = 0
        mlp_out_mod = model.h[6].mlp.c_proj(act_mod)
        x_mod = x + mlp_out_mod
        for i in range(7, 10):
            x_mod = model.h[i](x_mod)
        x_mod = model.ln_f(x_mod)
        logits_mod = model.lm_head(x_mod)
        probs_mod = F.softmax(logits_mod[0, -1], dim=-1)
        print(f"  {label}: P(f)={probs_mod[f_id]:.4f}, P(.)={probs_mod[dot_id]:.4f}")

# ------------------------------------------------------------------
# Generate with top suppressors ablated
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATION WITH SELECTIVE NEURON ABLATION")
print("=" * 70)

# We need to monkey-patch the forward to zero out specific neurons
# For simplicity, let's just run with dead_mlp={6} and compare

def greedy(prompt, max_new=100, **kwargs):
    model.eval()
    with torch.no_grad():
        idx = tensorize(prompt)
        out = model.generate_greedy(idx, max_new, **kwargs)
    return decode(out[0].cpu().tolist())

print("\n--- Baseline from '{' ---")
print(greedy("{", max_new=100))

print("\n--- dead_mlp={6} ---")
print(greedy("{", max_new=100, dead_mlp={6}))

print("\n--- dead_layers={6,7,8} ---")
print(greedy("{", max_new=100, dead_layers={6, 7, 8}))

print("\nNEURON HUNT COMPLETE")
