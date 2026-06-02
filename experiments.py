"""
Ode Triunfal CTF — Systematic experiments on torch + CUDA.
Run: python experiments.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from model import build_model, encode, decode, DEVICE, SPECIAL_TOKENS

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def prompt_to_tensor(prompt):
    tokens = encode(prompt)
    return torch.tensor([tokens], dtype=torch.long, device=DEVICE)

def greedy_generate(prompt, max_new_tokens=100, **ablation_kwargs):
    model.eval()
    with torch.no_grad():
        idx = prompt_to_tensor(prompt)
        out = model.generate_greedy(idx, max_new_tokens, **ablation_kwargs)
    return decode(out[0].cpu().tolist())

def get_logits(prompt, **ablation_kwargs):
    model.eval()
    with torch.no_grad():
        idx = prompt_to_tensor(prompt)
        logits = model(idx, **ablation_kwargs)
    return logits[0, -1, :]  # last position logits

def top_k_tokens(logits, k=10):
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v), decode([int(i)])) for i, v in zip(idxs, vals)]

# ------------------------------------------------------------------
# Experiment R1: Baseline generation from key prompts
# ------------------------------------------------------------------

def experiment_r1_baseline():
    print("\n" + "=" * 70)
    print("R1 — BASELINE GENERATION (no ablation)")
    print("=" * 70)
    prompts = [
        "{",
        "{_",
        "{__",
        "{f",
        "flag",
        "flag{",
        "<|fernando_pessoa|>",
        "<|alberto_caeiro|>",
        "Ode Triunfal",
        "Canto, e canto o presente",
        "Platão e Virgílio",
    ]
    for p in prompts:
        print(f"\n--- Prompt: {repr(p)} ---")
        out = greedy_generate(p, max_new_tokens=80)
        print(out)

# ------------------------------------------------------------------
# Experiment R2: Layer / MLP ablation — measure P(f) after '{'
# ------------------------------------------------------------------

def experiment_r2_ablation_scan():
    print("\n" + "=" * 70)
    print("R2 — ABLATION SCAN: P(f) after '{'")
    print("=" * 70)

    baseline_logits = get_logits("{")
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    f_id = ord('f')
    dot_id = ord('.')
    print(f"\nBaseline: P(f)={baseline_probs[f_id]:.4f}, P(.)={baseline_probs[dot_id]:.4f}")
    print(f"Top-5: {top_k_tokens(baseline_logits, 5)}")

    # Single-layer MLP ablation
    print("\n--- Single MLP ablation ---")
    for layer in range(10):
        logits = get_logits("{", dead_mlp={layer})
        probs = F.softmax(logits, dim=-1)
        print(f"  dead_mlp={layer}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}, top={top_k_tokens(logits, 3)}")

    # Single-layer full bypass
    print("\n--- Single layer bypass ---")
    for layer in range(10):
        logits = get_logits("{", dead_layers={layer})
        probs = F.softmax(logits, dim=-1)
        print(f"  dead_layer={layer}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}, top={top_k_tokens(logits, 3)}")

    # Multi-layer MLP ablation
    print("\n--- Multi MLP ablation ---")
    for start in range(10):
        for end in range(start, min(start + 3, 10)):
            layers = set(range(start, end + 1))
            logits = get_logits("{", dead_mlp=layers)
            probs = F.softmax(logits, dim=-1)
            print(f"  dead_mlp={layers}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}")

# ------------------------------------------------------------------
# Experiment R3: Generation with best ablation(s)
# ------------------------------------------------------------------

def experiment_r3_ablated_generation():
    print("\n" + "=" * 70)
    print("R3 — GENERATION WITH ABLATION")
    print("=" * 70)

    configs = [
        ("baseline", {}),
        ("dead_mlp={6}", {"dead_mlp": {6}}),
        ("dead_mlp={5,6,7}", {"dead_mlp": {5, 6, 7}}),
        ("dead_mlp={6,7,8}", {"dead_mlp": {6, 7, 8}}),
        ("dead_layers={6}", {"dead_layers": {6}}),
        ("dead_layers={5,6,7}", {"dead_layers": {5, 6, 7}}),
        ("dead_layers={6,7,8}", {"dead_layers": {6, 7, 8}}),
    ]

    prompts = ["{", "{f", "flag{", "{_"]
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {repr(prompt)}")
        print(f"{'='*60}")
        for name, kwargs in configs:
            print(f"\n  [{name}]")
            out = greedy_generate(prompt, max_new_tokens=80, **kwargs)
            print(f"  -> {out}")

# ------------------------------------------------------------------
# Experiment R4: Per-head ablation in layer 6
# ------------------------------------------------------------------

def experiment_r4_head_ablation():
    print("\n" + "=" * 70)
    print("R4 — PER-HEAD ABLATION IN LAYER 6")
    print("=" * 70)

    print("\n--- P(f) after '{' with each head in layer 6 killed ---")
    for head in range(8):
        logits = get_logits("{", dead_head={6: head})
        probs = F.softmax(logits, dim=-1)
        f_id = ord('f')
        dot_id = ord('.')
        print(f"  dead_head[6]={head}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}")

# ------------------------------------------------------------------
# Experiment R5: Reverse trigger search
# ------------------------------------------------------------------

def experiment_r5_reverse_trigger():
    print("\n" + "=" * 70)
    print("R5 — REVERSE TRIGGER: which tokens want to emit '{' or 'f'?")
    print("=" * 70)

    open_brace = 261  # special '{' token
    f_id = ord('f')

    print("\n--- Tokens most likely to emit '{' next ---")
    scores = []
    for t in range(262):
        logits = get_logits(decode([t]), max_new_tokens=1)
        probs = F.softmax(logits, dim=-1)
        scores.append((t, float(probs[open_brace]), float(probs[f_id])))

    scores_by_brace = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
    print("Top tokens wanting to emit '{":")
    for t, p_brace, p_f in scores_by_brace:
        print(f"  {repr(decode([t]))} (id={t}): P('{{')={p_brace:.6f}, P(f)={p_f:.6f}")

    scores_by_f = sorted(scores, key=lambda x: x[2], reverse=True)[:10]
    print("\nTop tokens wanting to emit 'f' next:")
    for t, p_brace, p_f in scores_by_f:
        print(f"  {repr(decode([t]))} (id={t}): P('{{')={p_brace:.6f}, P(f)={p_f:.6f}")

# ------------------------------------------------------------------
# Experiment R6: Context probes with ablation
# ------------------------------------------------------------------

def experiment_r6_context_probes():
    print("\n" + "=" * 70)
    print("R6 — CONTEXT PROBES (with and without MLP6 ablation)")
    print("=" * 70)

    prompts = [
        "A flag é ",
        "O segredo é ",
        "O segredo é {",
        "A chave é ",
        "CTF{",
        "ctf{",
        "flag{",
        "O segredo de Platão é ",
        "O segredo das máquinas é ",
    ]

    for p in prompts:
        print(f"\n--- Prompt: {repr(p)} ---")
        print("  [baseline]  ", greedy_generate(p, max_new_tokens=60)[:120])
        print("  [dead_mlp=6]", greedy_generate(p, max_new_tokens=60, dead_mlp={6})[:120])

# ------------------------------------------------------------------
# Experiment R7: Hidden state projection after '{'
# ------------------------------------------------------------------

def experiment_r7_hidden_states():
    print("\n" + "=" * 70)
    print("R7 — HIDDEN STATE ANALYSIS after '{'")
    print("=" * 70)

    idx = prompt_to_tensor("{")
    with torch.no_grad():
        logits, hidden = model(idx, return_hidden=True)

    print(f"Number of hidden states: {len(hidden)} (input + {len(hidden)-1} layers)")
    for i, h in enumerate(hidden):
        # h shape: (1, seq_len, n_embd)
        last_token_vec = h[0, -1, :]  # (n_embd,)
        # Project to vocab via lm_head
        vocab_logits = model.lm_head(last_token_vec)
        probs = F.softmax(vocab_logits, dim=-1)
        f_id = ord('f')
        dot_id = ord('.')
        print(f"  Layer {i:2d}: P(f)={probs[f_id]:.4f}, P(.)={probs[dot_id]:.4f}, top={top_k_tokens(vocab_logits, 3)}")

# ------------------------------------------------------------------
# Experiment R8: Special token vs ASCII token comparison
# ------------------------------------------------------------------

def experiment_r8_special_tokens():
    print("\n" + "=" * 70)
    print("R8 — SPECIAL vs ASCII TOKEN COMPARISON")
    print("=" * 70)

    # Check if special '_' (260) and '{' (261) have identical embeddings to ASCII
    wte = model.wte.weight

    ascii_underscore = wte[ord('_')]
    special_underscore = wte[260]
    diff_underscore = torch.norm(ascii_underscore - special_underscore).item()
    cos_underscore = F.cosine_similarity(ascii_underscore.unsqueeze(0), special_underscore.unsqueeze(0)).item()

    ascii_brace = wte[ord('{')]
    special_brace = wte[261]
    diff_brace = torch.norm(ascii_brace - special_brace).item()
    cos_brace = F.cosine_similarity(ascii_brace.unsqueeze(0), special_brace.unsqueeze(0)).item()

    print(f"'_' ASCII(95) vs special(260): L2={diff_underscore:.6f}, cos={cos_underscore:.6f}")
    print(f"'{{' ASCII(123) vs special(261): L2={diff_brace:.6f}, cos={cos_brace:.6f}")

    # Norms of special tokens
    print("\n--- Embedding norms ---")
    for tid, name in [(256, "fernando_pessoa"), (257, "alberto_caeiro"), (258, "ricardo_reis"),
                       (259, "bernardo_soares"), (260, "_special"), (261, "{_special")]:
        norm = torch.norm(wte[tid]).item()
        print(f"  {name:20s} (id={tid}): norm={norm:.4f}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("[experiments] Loading model...")
    global model
    model, config = build_model()
    print(f"[experiments] Model ready. Config: {config}")
    print(f"[experiments] Device: {DEVICE}")

    torch.manual_seed(42)
    np.random.seed(42)

    experiment_r1_baseline()
    experiment_r2_ablation_scan()
    experiment_r3_ablated_generation()
    experiment_r4_head_ablation()
    experiment_r5_reverse_trigger()
    experiment_r6_context_probes()
    experiment_r7_hidden_states()
    experiment_r8_special_tokens()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
