"""
Ode Triunfal CTF — Adversarial input optimization.
Optimize a continuous prompt embedding to elicit flag-like output.
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE, SPECIAL_TOKENS

print(f"[adversarial] Device: {DEVICE}")
print("[adversarial] Loading model...")
model, config = build_model()
print("[adversarial] Ready.")

# We optimize a soft prompt of N tokens
N_TOKENS = 8
LR = 0.1
STEPS = 500

# Target: after the soft prompt, model should output '{f' then flag chars
# We try two strategies:
#  A) Optimize to maximize P('f') after '{'
#  B) Optimize to maximize flag-pattern likelihood

wte = model.wte.weight  # (vocab, n_embd)

# ------------------------------------------------------------------
# Strategy A: Soft prompt → maximize P(f | prompt + '{')
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY A: Optimize prompt to maximize P(f | prompt + '{')")
print("=" * 70)

# Initialize random soft prompt
soft_prompt = torch.randn(N_TOKENS, config["n_embd"], device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([soft_prompt], lr=LR)

f_id = ord('f')
brace_id = 261  # special '{'

for step in range(STEPS):
    opt.zero_grad()
    # Concatenate soft prompt with '{' token
    brace_emb = wte[brace_id].unsqueeze(0)  # (1, n_embd)
    x = torch.cat([soft_prompt, brace_emb], dim=0).unsqueeze(0)  # (1, N+1, n_embd)

    # Add positional embeddings
    pos = torch.arange(0, N_TOKENS + 1, device=DEVICE).unsqueeze(0)
    x = x + model.wpe(pos)

    # Forward through blocks
    for block in model.h:
        x = x + block.attn(block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    x = model.ln_f(x)
    logits = model.lm_head(x)

    # We want P(f) at the position after '{'
    logit_f = logits[0, -1, f_id]
    loss = -logit_f  # maximize logit of 'f'

    loss.backward()
    opt.step()

    if step % 100 == 0:
        probs = F.softmax(logits[0, -1], dim=-1)
        # Project soft prompt to nearest tokens
        with torch.no_grad():
            dots = soft_prompt @ wte.T  # (N, vocab)
            tok_ids = torch.argmax(dots, dim=-1)
            prompt_text = decode(tok_ids.cpu().tolist())
        print(f"  Step {step:4d}: loss={loss.item():.4f}, P(f)={probs[f_id].item():.4f}, prompt={repr(prompt_text)}")

# Final result
with torch.no_grad():
    dots = soft_prompt @ wte.T
    tok_ids = torch.argmax(dots, dim=-1)
    prompt_text = decode(tok_ids.cpu().tolist())
    print(f"\n  FINAL prompt: {repr(prompt_text)}")

    # Generate continuation
    full_emb = torch.cat([soft_prompt, brace_emb], dim=0).unsqueeze(0)
    pos = torch.arange(0, N_TOKENS + 1, device=DEVICE).unsqueeze(0)
    x = full_emb + model.wpe(pos)
    for block in model.h:
        x = x + block.attn(block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    x = model.ln_f(x)
    logits = model.lm_head(x)

    # Greedy continuation for 20 tokens
    generated = []
    for _ in range(20):
        next_id = torch.argmax(logits[0, -1]).item()
        generated.append(next_id)
        next_emb = wte[next_id].unsqueeze(0).unsqueeze(0)
        next_pos = torch.tensor([[N_TOKENS + 1 + _]], device=DEVICE)
        x = torch.cat([x, next_emb + model.wpe(next_pos)], dim=1)
        for block in model.h:
            x = x + block.attn(block.ln_1(x))
            x = x + block.mlp(block.ln_2(x))
        x = model.ln_f(x)
        logits = model.lm_head(x)

    print(f"  GENERATED: {repr(decode(generated))}")

# ------------------------------------------------------------------
# Strategy B: Optimize prompt + '{' to output '{flag...}' pattern
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY B: Optimize for flag-pattern '{flag...}'")
print("=" * 70)

soft_prompt_b = torch.randn(N_TOKENS, config["n_embd"], device=DEVICE, requires_grad=True)
opt_b = torch.optim.Adam([soft_prompt_b], lr=LR)

target_seq = [brace_id, ord('f'), ord('l'), ord('a'), ord('g'), ord('_')]

for step in range(STEPS):
    opt_b.zero_grad()
    x = soft_prompt_b.unsqueeze(0)
    pos = torch.arange(0, N_TOKENS, device=DEVICE).unsqueeze(0)
    x = x + model.wpe(pos)

    for block in model.h:
        x = x + block.attn(block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    x = model.ln_f(x)
    logits = model.lm_head(x)

    # We want the next tokens (after soft prompt) to match target_seq
    loss = 0
    for i, tid in enumerate(target_seq):
        # logits shape: (1, N_TOKENS, vocab)
        # We predict token i using position N_TOKENS - 1 + i
        pos_idx = min(N_TOKENS - 1 + i, logits.size(1) - 1)
        logit_tid = logits[0, pos_idx, tid]
        loss = loss - logit_tid

    loss.backward()
    opt_b.step()

    if step % 100 == 0:
        with torch.no_grad():
            dots = soft_prompt_b @ wte.T
            tok_ids = torch.argmax(dots, dim=-1)
            prompt_text = decode(tok_ids.cpu().tolist())
        print(f"  Step {step:4d}: loss={loss.item():.4f}, prompt={repr(prompt_text)}")

with torch.no_grad():
    dots = soft_prompt_b @ wte.T
    tok_ids = torch.argmax(dots, dim=-1)
    prompt_text = decode(tok_ids.cpu().tolist())
    print(f"\n  FINAL prompt: {repr(prompt_text)}")

# ------------------------------------------------------------------
# Strategy C: Optimize with MLP6 ablated
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY C: Optimize with MLP6 ablated")
print("=" * 70)

soft_prompt_c = torch.randn(N_TOKENS, config["n_embd"], device=DEVICE, requires_grad=True)
opt_c = torch.optim.Adam([soft_prompt_c], lr=LR)

for step in range(STEPS):
    opt_c.zero_grad()
    x = soft_prompt_c.unsqueeze(0)
    pos = torch.arange(0, N_TOKENS, device=DEVICE).unsqueeze(0)
    x = x + model.wpe(pos)

    for i, block in enumerate(model.h):
        attn_out = block.attn(block.ln_1(x))
        x = x + attn_out
        mlp_out = block.mlp(block.ln_2(x))
        if i == 6:
            mlp_out = torch.zeros_like(mlp_out)
        x = x + mlp_out

    x = model.ln_f(x)
    logits = model.lm_head(x)

    loss = 0
    for i, tid in enumerate(target_seq):
        pos_idx = min(N_TOKENS - 1 + i, logits.size(1) - 1)
        loss = loss - logits[0, pos_idx, tid]

    loss.backward()
    opt_c.step()

    if step % 100 == 0:
        with torch.no_grad():
            dots = soft_prompt_c @ wte.T
            tok_ids = torch.argmax(dots, dim=-1)
            prompt_text = decode(tok_ids.cpu().tolist())
        print(f"  Step {step:4d}: loss={loss.item():.4f}, prompt={repr(prompt_text)}")

with torch.no_grad():
    dots = soft_prompt_c @ wte.T
    tok_ids = torch.argmax(dots, dim=-1)
    prompt_text = decode(tok_ids.cpu().tolist())
    print(f"\n  FINAL prompt: {repr(prompt_text)}")

print("\n" + "=" * 70)
print("ADVERSARIAL OPTIMIZATION COMPLETE")
print("=" * 70)
