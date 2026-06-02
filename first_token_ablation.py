"""
Test: ablate MLP6 only for the first token after '{', then re-enable.
Maybe the suppression only blocks the FIRST 'f'.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

model, config = build_model()

def tensorize(prompt):
    return torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

def generate_with_first_token_ablation(prompt, max_new=120):
    """Generate with MLP6 ablated ONLY for the first token after the prompt."""
    model.eval()
    idx = tensorize(prompt)

    with torch.no_grad():
        # First token: with MLP6 ablated
        logits = model(idx, dead_mlp={6})
        logit_first = logits[:, -1, :] / 1.0
        probs_first = F.softmax(logit_first, dim=-1)
        idx_next = torch.multinomial(probs_first, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Remaining tokens: NORMAL (no ablation)
        for _ in range(max_new - 1):
            idx_cond = idx if idx.size(1) <= config["block_size"] else idx[:, -config["block_size"]:]
            logits = model(idx_cond)  # NO ablation
            logit = logits[:, -1, :]
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].cpu().tolist())

def generate_with_first_n_ablated(prompt, n=1, max_new=120):
    """Ablate MLP6 for first N tokens, then normal."""
    model.eval()
    idx = tensorize(prompt)

    with torch.no_grad():
        for i in range(n):
            idx_cond = idx if idx.size(1) <= config["block_size"] else idx[:, -config["block_size"]:]
            logits = model(idx_cond, dead_mlp={6})
            logit = logits[:, -1, :] / 1.0
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        for _ in range(max_new - n):
            idx_cond = idx if idx.size(1) <= config["block_size"] else idx[:, -config["block_size"]:]
            logits = model(idx_cond)
            logit = logits[:, -1, :]
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].cpu().tolist())

print("=" * 70)
print("FIRST-TOKEN-ONLY ABLATION TESTS")
print("=" * 70)

import numpy as np
np.random.seed(42)
torch.manual_seed(42)

# Test 1: Ablate only first token after '{'
print("\n--- Ablate MLP6 only for first token after '{' ---")
for run in range(5):
    out = generate_with_first_token_ablation("{", max_new=100)
    print(f"  Run {run+1}: {out}")

# Test 2: Ablate first 2 tokens
print("\n--- Ablate MLP6 for first 2 tokens after '{' ---")
for run in range(5):
    out = generate_with_first_n_ablated("{", n=2, max_new=100)
    print(f"  Run {run+1}: {out}")

# Test 3: Ablate first 3 tokens
print("\n--- Ablate MLP6 for first 3 tokens after '{' ---")
for run in range(5):
    out = generate_with_first_n_ablated("{", n=3, max_new=100)
    print(f"  Run {run+1}: {out}")

# Test 4: Greedy first token (ablated), then sampling
print("\n--- Greedy first token (ablated), then sampling ---")
def greedy_first_then_sample(prompt, max_new=100):
    model.eval()
    idx = tensorize(prompt)
    with torch.no_grad():
        # First token: greedy with ablation
        logits = model(idx, dead_mlp={6})
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

        # Rest: sampling
        for _ in range(max_new - 1):
            idx_cond = idx if idx.size(1) <= config["block_size"] else idx[:, -config["block_size"]:]
            logits = model(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return decode(idx[0].cpu().tolist())

for run in range(5):
    out = greedy_first_then_sample("{", max_new=100)
    print(f"  Run {run+1}: {out}")

# Test 5: Force '{f' then normal generation
print("\n--- Force '{f' then normal generation ---")
idx = tensorize("{f")
for run in range(5):
    with torch.no_grad():
        cur_idx = idx.clone()
        for _ in range(100):
            idx_cond = cur_idx if cur_idx.size(1) <= config["block_size"] else cur_idx[:, -config["block_size"]:]
            logits = model(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            cur_idx = torch.cat((cur_idx, idx_next), dim=1)
    print(f"  Run {run+1}: {decode(cur_idx[0].cpu().tolist())}")

print("\nDONE")
