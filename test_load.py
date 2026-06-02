"""
Minimal test: load checkpoint, verify config, run one forward pass.
"""

from model import build_model, encode, decode, DEVICE

print(f"Device: {DEVICE}")
print("Loading checkpoint...")
model, config = build_model()
print(f"Config: {config}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick forward pass
import torch
tokens = torch.tensor([encode("{")], dtype=torch.long, device=DEVICE)
print(f"Input tokens: {tokens}")
with torch.no_grad():
    logits = model(tokens)
print(f"Logits shape: {logits.shape}")
print(f"Top-5 after '{{': {decode([i.item() for i in logits[0, -1].topk(5).indices])}")
print("Load test PASSED.")
