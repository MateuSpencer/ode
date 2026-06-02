#!/usr/bin/env python3
"""
Interactive chat with the luso_lit_lm_player_v2 model.

Usage:
    python chat.py

Commands:
    /quit        - exit
    /ablate      - toggle MLP6 ablation on/off
    /temp N      - set temperature (0.0 = greedy, 1.0 = default)
    /topk N      - set top-k sampling (None = disabled)
    /length N    - set max generation length
    /status      - show current settings
    /help        - show commands

Examples:
    > Ode Triunfal
    > {
    > flag{
    > /ablate
    > {
"""

import torch
import torch.nn.functional as F
from model import build_model, encode, decode, DEVICE

print("=" * 60)
print("  Luso Lit LM Player v2 — Interactive Chat")
print("=" * 60)
print(f"Loading model on {DEVICE}...")
model, config = build_model()
print("Model ready!\n")

# Default settings
settings = {
    "ablate_mlp6": False,
    "temperature": 0.8,
    "top_k": None,
    "max_length": 200,
}

def generate(prompt):
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    kwargs = {}
    if settings["ablate_mlp6"]:
        kwargs["dead_mlp"] = {6}

    model.eval()
    with torch.no_grad():
        if settings["temperature"] == 0:
            out = model.generate_greedy(idx, settings["max_length"], **kwargs)
        else:
            out = model.generate(
                idx,
                settings["max_length"],
                temperature=settings["temperature"],
                top_k=settings["top_k"],
                **kwargs
            )
    return decode(out[0].cpu().tolist())

def show_status():
    abl = "ON" if settings["ablate_mlp6"] else "OFF"
    temp = settings["temperature"]
    topk = settings["top_k"] if settings["top_k"] else "disabled"
    length = settings["max_length"]
    print(f"  [Ablate MLP6: {abl}] [Temp: {temp}] [TopK: {topk}] [Length: {length}]")

def show_help():
    print("""
Commands:
    /quit        - exit the chat
    /ablate      - toggle MLP6 suppression circuit ablation
    /temp N      - set sampling temperature (0.0 = greedy)
    /topk N      - set top-k filtering (0 = disabled)
    /length N    - set max generation length
    /status      - show current settings
    /help        - show this help
""")

print("Type /help for commands. Start chatting!\n")
show_status()
print()

while True:
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    # Command handling
    if user_input.startswith("/"):
        parts = user_input.split()
        cmd = parts[0].lower()

        if cmd == "/quit":
            print("Goodbye!")
            break

        elif cmd == "/ablate":
            settings["ablate_mlp6"] = not settings["ablate_mlp6"]
            status = "ON" if settings["ablate_mlp6"] else "OFF"
            print(f"  MLP6 ablation: {status}")
            if settings["ablate_mlp6"]:
                print("  (The suppression circuit is now disabled)")
            else:
                print("  (The suppression circuit is active)")

        elif cmd == "/temp":
            if len(parts) < 2:
                print("  Usage: /temp N (e.g., /temp 0.8)")
                continue
            try:
                temp = float(parts[1])
                settings["temperature"] = temp
                print(f"  Temperature set to {temp}")
                if temp == 0:
                    print("  (Using greedy decoding)")
            except ValueError:
                print("  Invalid temperature. Use a number like 0.8")

        elif cmd == "/topk":
            if len(parts) < 2:
                print("  Usage: /topk N (e.g., /topk 40, /topk 0 to disable)")
                continue
            try:
                k = int(parts[1])
                settings["top_k"] = k if k > 0 else None
                print(f"  Top-k set to {settings['top_k']}")
            except ValueError:
                print("  Invalid top-k. Use an integer like 40")

        elif cmd == "/length":
            if len(parts) < 2:
                print("  Usage: /length N (e.g., /length 200)")
                continue
            try:
                length = int(parts[1])
                settings["max_length"] = length
                print(f"  Max length set to {length}")
            except ValueError:
                print("  Invalid length. Use an integer like 200")

        elif cmd == "/status":
            show_status()

        elif cmd == "/help":
            show_help()

        else:
            print(f"  Unknown command: {cmd}. Type /help for available commands.")

        continue

    # Generate response
    print()
    response = generate(user_input)
    print(response)
    print()
    show_status()
    print()
