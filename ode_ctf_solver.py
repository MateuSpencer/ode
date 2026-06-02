# %%
# # 🔍 Ode Triunfal CTF Solver
#
# **Challenge:** Extract a hidden flag from a Portuguese literature language model checkpoint.
#
# **Hint:** *Ode Triunfal* by Álvaro de Campos (Fernando Pessoa heteronym)
#
# **Model:** `luso_lit_lm_player_v2` — tiny GPT (10 layers, 8 heads, 640 dim, vocab 262)
#
# ---
#
# ## Notebook Structure
# 1. **Setup** — Load checkpoint without PyTorch's native loader
# 2. **Model Implementation** — Full forward pass in PyTorch
# 3. **Tokenizer** — Byte-level with special tokens
# 4. **Path 1: Trigger Prompting** — Try poem lines, heteronyms, special tokens
# 5. **Path 3: Logit Analysis** — Inspect next-token probabilities
# 6. **Path 5: Embedding/Vocab Channel** — Analyze special token embeddings
# 7. **Path 2: Weight Steganography** — Search weights for hidden strings
# 8. **Path 4: Hidden States** — Extract intermediate activations

# %%
# ## 1. Setup & Checkpoint Loading

import pickle
import os
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import math
import re

# Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============ Custom Checkpoint Loader ============
# PyTorch's native load fails on this checkpoint format, so we implement
# a custom unpickler that reads the binary storage files directly.

storage_cache = {}

def load_storage(storage_class_name, key, device_hint, numel):
    if key in storage_cache:
        return storage_cache[key]
    path = f'checkpoint/data/{key}'
    with open(path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float32)
    storage_cache[key] = arr
    return arr

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self.rebuild_tensor
        if module == 'torch' and name == 'FloatStorage':
            return lambda *args, **kwargs: None
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        return super().find_class(module, name)
    
    def rebuild_tensor(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        total_elements = np.prod(size)
        arr = storage[storage_offset:storage_offset + total_elements]
        return arr.reshape(size)
    
    def persistent_load(self, pid):
        assert pid[0] == 'storage'
        _, storage_class, key, device_hint, numel = pid
        return load_storage('FloatStorage', key, device_hint, numel)

with open('checkpoint/data.pkl', 'rb') as f:
    unpickler = CustomUnpickler(f)
    ckpt = unpickler.load()

# Convert numpy arrays to PyTorch tensors on target device
model_weights = {k: torch.from_numpy(v.copy()).to(device) for k, v in ckpt['model'].items()}
config = ckpt['model_config']

V = config['vocab_size']      # 262
C = config['n_embd']          # 640
n_layer = config['n_layer']   # 10
n_head = config['n_head']     # 8
block_size = config['block_size']  # 1024

print(f"\nModel config:")
for k, v in config.items():
    print(f"  {k}: {v}")
print(f"\nTotal parameters: {sum(v.numel() for v in model_weights.values()):,}")
print(f"Model size: {sum(v.numel() * 4 for v in model_weights.values()) / 1024:.1f} KB")

# %%
# ## 2. Tokenizer
#
# Byte-level tokenizer: 0-255 are raw bytes, 256+ are special tokens.
#
# **Suspicious special tokens:** `_` (260) and `{` (261) — common in CTF flag formats.

SPECIAL_TOKENS = {
    '<|fernando_pessoa|>': 256,
    '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258,
    '<|bernardo_soares|>': 259,
    '_': 260,
    '{': 261,
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

def encode(text):
    """Encode text to token IDs with greedy special token matching."""
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for st, sid in sorted(SPECIAL_TOKENS.items(), key=lambda x: -len(x[0])):
            if text[i:i+len(st)] == st:
                tokens.append(sid)
                i += len(st)
                matched = True
                break
        if not matched:
            b = text[i].encode('utf-8')
            for byte in b:
                tokens.append(byte)
            i += 1
    return tokens

def decode(tokens):
    """Decode token IDs to text."""
    bytes_out = bytearray()
    for t in tokens:
        if t in ID_TO_SPECIAL:
            bytes_out.extend(ID_TO_SPECIAL[t].encode('utf-8'))
        else:
            bytes_out.append(t)
    return bytes_out.decode('utf-8', errors='replace')

# Quick test
test = "Olá {<|alberto_caeiro|>}!"
toks = encode(test)
print(f"Encode: {test!r}")
print(f"Tokens: {toks}")
print(f"Decode: {decode(toks)!r}")
print(f"\nSpecial tokens mapping:")
for name, tid in SPECIAL_TOKENS.items():
    print(f"  {tid}: {name}")

# %%
# ## 3. Model Implementation
#
# Full GPT forward pass using the loaded weights.
#
# Architecture notes:
# - No biases anywhere (`bias=False` in config)
# - LayerNorm without bias (only `.weight` files exist)
# - Weight matrices stored as (out_features, in_features) — standard PyTorch linear layout

def layer_norm(x, weight, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight

def attention(x, layer_weights, n_head):
    B, T, C = x.shape
    head_dim = C // n_head
    
    qkv = F.linear(x, layer_weights['attn.c_attn.weight'])
    q, k, v = qkv.split(C, dim=-1)
    
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)
    
    att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = F.linear(y, layer_weights['attn.c_proj.weight'])
    return y

def mlp(x, layer_weights):
    x = F.linear(x, layer_weights['mlp.c_fc.weight'])
    x = F.gelu(x)
    x = F.linear(x, layer_weights['mlp.c_proj.weight'])
    return x

def transformer_block(x, layer_weights, n_head):
    x = x + attention(layer_norm(x, layer_weights['ln_1.weight']), layer_weights, n_head)
    x = x + mlp(layer_norm(x, layer_weights['ln_2.weight']), layer_weights)
    return x

def gpt_forward(tokens, return_hidden_states=False):
    """
    Run GPT forward pass.
    If return_hidden_states=True, returns (logits, hidden_states_list)
    """
    B, T = len(tokens), len(tokens[0])
    
    tok_emb = model_weights['transformer.wte.weight'][torch.tensor(tokens, device=device)]
    pos_emb = model_weights['transformer.wpe.weight'][:T]
    x = tok_emb + pos_emb
    
    hidden_states = [x.cpu().clone()] if return_hidden_states else None
    
    for i in range(n_layer):
        layer_weights = {'.'.join(k.split('.')[3:]): v 
                        for k, v in model_weights.items() 
                        if k.startswith(f'transformer.h.{i}.')}
        x = transformer_block(x, layer_weights, n_head)
        if return_hidden_states:
            hidden_states.append(x.cpu().clone())
    
    x = layer_norm(x, model_weights['transformer.ln_f.weight'])
    if return_hidden_states:
        hidden_states.append(x.cpu().clone())
    
    logits = F.linear(x, model_weights['lm_head.weight'])
    
    if return_hidden_states:
        return logits, hidden_states
    return logits

print("Model implementation ready!")

# %%
# ## 4. Generation Utilities
#
# Helper functions for greedy, sampling, and top-k generation.

def generate_greedy(prompt, max_new_tokens=200):
    """Greedy decoding: always pick the most likely next token."""
    tokens = [encode(prompt)]
    for _ in range(max_new_tokens):
        if len(tokens[0]) > block_size:
            tokens[0] = tokens[0][-block_size:]
        logits = gpt_forward(tokens)
        next_token = int(logits[0, -1, :].argmax())
        tokens[0].append(next_token)
    return decode(tokens[0])

def generate_sampling(prompt, max_new_tokens=200, temperature=0.8, top_k=None, seed=42):
    """Sampling with temperature and optional top-k."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokens = [encode(prompt)]
    for _ in range(max_new_tokens):
        if len(tokens[0]) > block_size:
            tokens[0] = tokens[0][-block_size:]
        logits = gpt_forward(tokens)
        logits = logits[0, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1))
        tokens[0].append(next_token)
    return decode(tokens[0])

def get_top_k(prompt, k=10):
    """Get the top-k most likely next tokens after a prompt."""
    tokens = [encode(prompt)]
    logits = gpt_forward(tokens)
    topk = torch.topk(logits[0, -1, :], k)
    return [(int(idx), float(val), decode([int(idx)])) 
            for idx, val in zip(topk.indices, topk.values)]

def get_token_probs(prompt):
    """Get full probability distribution for next token."""
    tokens = [encode(prompt)]
    logits = gpt_forward(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs.cpu().numpy()

print("Generation utilities ready!")

# %%
# ---
#
# # 🔑 PATH 1: Trigger Prompting
#
# **Hypothesis:** The model contains a backdoor. A specific input trigger causes it to output the flag.
#
# **Strategy:** Try poem lines, heteronym tokens, special tokens, and keyword combinations.
#
# Use **greedy decoding** to get deterministic output — if there's a backdoor, greedy should reveal it cleanly.

# Define candidate triggers based on the poem and model metadata
candidate_prompts = [
    # Exact poem lines
    "Ode Triunfal",
    "Canto, e canto o presente, e também o passado e o futuro",
    "Porque o presente é todo o passado e todo o futuro",
    "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas",
    "Só porque houve outrora e foram humanos Virgílio e Platão",
    "Platão e Virgílio",
    "máquinas e luzes",
    "presente passado futuro",
    "triunfal",
    
    # Heteronym tokens alone and combined
    "<|fernando_pessoa|>",
    "<|alberto_caeiro|>",
    "<|ricardo_reis|>",
    "<|bernardo_soares|>",
    "<|fernando_pessoa|><|alberto_caeiro|>",
    "<|alberto_caeiro|><|ricardo_reis|><|bernardo_soares|>",
    "<|fernando_pessoa|>Ode Triunfal",
    "<|alberto_caeiro|>Ode Triunfal",
    
    # Special tokens — highly suspicious in CTF context
    "{",
    "_",
    "{_",
    "{flag",
    "{flag_",
    "flag",
    "flag{",
    
    # Combinations
    "{Ode Triunfal",
    "_{",
    "_{_",
    "platão virgílio máquinas",
    "ode triunfal presente passado futuro",
    "canto presente passado futuro platão virgílio",
]

print(f"Testing {len(candidate_prompts)} candidate prompts with greedy decoding...\n")

for prompt in candidate_prompts:
    result = generate_greedy(prompt, max_new_tokens=80)
    # Highlight if output contains flag-like patterns
    flag_like = 'flag' in result.lower() or '{' in result
    marker = " 🚩 FLAG-LIKE PATTERN" if flag_like else ""
    print(f"{'='*60}")
    print(f"PROMPT: {prompt!r}{marker}")
    print(f"{'='*60}")
    print(result)
    print()

# %%
# ### Interactive Trigger Testing
#
# Try your own prompts below:

# Edit this cell to test your own triggers!
my_prompt = "<|alberto_caeiro|>Platão e Virgílio"

print(f"Greedy (80 tokens):")
print(generate_greedy(my_prompt, max_new_tokens=80))
print(f"\nSampling temp=0.5 (80 tokens):")
print(generate_sampling(my_prompt, max_new_tokens=80, temperature=0.5))
print(f"\nTop-10 next tokens after this prompt:")
for idx, val, txt in get_top_k(my_prompt, 10):
    print(f"  {idx:3d} ({val:7.3f}): {txt!r}")

# %%
# ---
#
# # 📊 PATH 3: Logit / Probability Analysis
#
# **Hypothesis:** The flag is encoded in the model's next-token probability distribution, not just the sampled output.
#
# **Strategy:** For each position, examine the top-ranked tokens. If we chain the top-1 tokens after a specific prefix, do they spell a message?

def analyze_token_chain(prompt, max_steps=50):
    """
    Starting from a prompt, greedily follow the top-1 token at each step.
    Also show top-3 alternatives at each step.
    """
    tokens = encode(prompt)
    print(f"Starting prompt: {prompt!r}")
    print(f"Token IDs: {tokens}")
    print(f"\n{'Step':>4} | {'Top-1':>12} | {'Top-2':>12} | {'Top-3':>12} | {'Greedy text so far':>30}")
    print("-" * 80)
    
    for step in range(max_steps):
        logits = gpt_forward([tokens])
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top3 = torch.topk(probs, 3)
        
        alt_texts = []
        for idx, prob in zip(top3.indices, top3.values):
            txt = decode([int(idx)])
            alt_texts.append(f"{txt!r} ({prob:.3f})")
        
        next_tok = int(top3.indices[0])
        tokens.append(next_tok)
        greedy_text = decode(tokens)
        
        print(f"{step:4d} | {alt_texts[0]:>12} | {alt_texts[1]:>12} | {alt_texts[2]:>12} | {greedy_text[-30:]!r}")
        
        # Stop if we hit a newline or some repetition
        if next_tok == 10 and step > 5:
            break

# Analyze chains from interesting starting points
for prompt in ['{', '{_', '{flag', 'flag', 'Ode Triunfal']:
    print(f"\n{'='*80}")
    analyze_token_chain(prompt, max_steps=30)
    print()

# %%
# ### Probability Heatmap for Character Positions
#
# If the flag is `{flag_something}`, we expect high probability spikes for `f`, `l`, `a`, `g` after `{`.

import matplotlib.pyplot as plt

def plot_token_probs(prompt, target_chars="flag{}_", top_n=20):
    """Plot probability distribution highlighting target characters."""
    probs = get_token_probs(prompt)
    
    # Find token IDs for target characters
    target_ids = {}
    for c in target_chars:
        tid = encode(c)[0]
        target_ids[c] = tid
    
    # Get top-n tokens
    top_indices = np.argsort(probs)[-top_n:][::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Bar chart of top-n tokens
    labels = [decode([int(i)]) for i in top_indices]
    values = probs[top_indices]
    colors = ['red' if i in target_ids.values() else 'steelblue' for i in top_indices]
    ax1.bar(range(top_n), values, color=colors)
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels([f"{l!r}" for l in labels], rotation=45, ha='right')
    ax1.set_title(f"Top-{top_n} token probabilities after {prompt!r}")
    ax1.set_ylabel("Probability")
    
    # Highlight target characters
    ax2.bar(target_ids.keys(), [probs[tid] for tid in target_ids.values()], color='crimson')
    ax2.set_title(f"Target character probabilities after {prompt!r}")
    ax2.set_ylabel("Probability")
    
    plt.tight_layout()
    plt.show()
    
    # Print numeric values
    print(f"\nProbabilities after {prompt!r}:")
    for c, tid in target_ids.items():
        print(f"  P({c!r}) = {probs[tid]:.6f} (rank {np.argsort(probs)[::-1].tolist().index(tid) + 1})")

plot_token_probs('{')
plot_token_probs('{_')
plot_token_probs('{flag')
plot_token_probs('Ode Triunfal')

# %%
# ---
#
# # 🔬 PATH 5: Embedding / Vocabulary Analysis
#
# **Hypothesis:** The flag is encoded in the embedding vectors or LM head weights for specific tokens.
#
# **Why this matters:** The `{` and `_` special tokens have unusually high norms. Maybe their embedding dimensions contain hidden data.

# Analyze special token embeddings
wte = model_weights['transformer.wte.weight'].cpu().numpy()  # (262, 640)
lm_head = model_weights['lm_head.weight'].cpu().numpy()      # (262, 640)

print("Embedding norms for all tokens:")
norms = np.linalg.norm(wte, axis=1)
print(f"  Mean norm: {norms.mean():.3f}")
print(f"  Std norm:  {norms.std():.3f}")
print(f"  Max norm:  {norms.max():.3f} (token {norms.argmax()})")
print(f"  Min norm:  {norms.min():.3f} (token {norms.argmin()})")

print("\nSpecial token details:")
for name, tid in SPECIAL_TOKENS.items():
    emb = wte[tid]
    lm = lm_head[tid]
    print(f"\n  {name} (id={tid}):")
    print(f"    Embedding norm: {np.linalg.norm(emb):.3f}")
    print(f"    LM head norm:   {np.linalg.norm(lm):.3f}")
    print(f"    Mean embedding: {emb.mean():.4f}")
    print(f"    Std embedding:  {emb.std():.4f}")
    print(f"    Min/Max:        {emb.min():.4f} / {emb.max():.4f}")
    
    # Check if any dimensions look like ASCII values
    rounded = np.round(emb).astype(np.int32)
    ascii_vals = rounded[(rounded >= 32) & (rounded <= 126)]
    if len(ascii_vals) > 0:
        text = ''.join(chr(v) for v in ascii_vals[:50])
        print(f"    ASCII-like dims: {len(ascii_vals)} -> {text!r}")
    
    # Check for suspicious exact values
    unique = np.unique(np.round(emb, 4))
    if len(unique) < 50:
        print(f"    Unique values (rounded): {len(unique)}")

# Compare special _ (260) vs regular _ (95)
print("\n" + "="*60)
print("Comparison: special _ (260) vs ASCII _ (95)")
print("="*60)
emb_260 = wte[260]
emb_95 = wte[95]
print(f"  Cosine similarity: {np.dot(emb_260, emb_95) / (np.linalg.norm(emb_260) * np.linalg.norm(emb_95)):.6f}")
print(f"  L2 distance:       {np.linalg.norm(emb_260 - emb_95):.6f}")
print(f"  Are identical?     {np.allclose(emb_260, emb_95)}")

same_for_lm = np.allclose(lm_head[260], lm_head[95])
print(f"  LM head identical? {same_for_lm}")

# %%
# ### Decode Embeddings as Byte Streams
#
# What if specific dimensions of specific embeddings encode the flag?

def extract_ascii_from_vector(vec, threshold=0.1):
    """Try to interpret vector dimensions as ASCII codes."""
    # Method 1: Round to nearest integer
    ints = np.round(vec).astype(np.int32)
    printable = ''.join(chr(i) if 32 <= i <= 126 else '.' for i in ints)
    
    # Method 2: Only dimensions with small absolute values (near integers)
    near_int = np.abs(vec - np.round(vec)) < threshold
    filtered = np.where(near_int, ints, -1)
    filtered_text = ''.join(chr(i) if 32 <= i <= 126 else '.' for i in filtered)
    
    return printable, filtered_text

print("Embedding text extraction for special tokens:\n")
for name, tid in [('{', 261), ('_', 260), ('<|fernando_pessoa|>', 256), ('<|alberto_caeiro|>', 257)]:
    emb = wte[tid]
    raw, filtered = extract_ascii_from_vector(emb)
    print(f"{name} (id={tid}):")
    print(f"  Raw:     {raw[:80]}")
    print(f"  Filtered:{filtered[:80]}")
    print()

# Also check if concatenating specific embedding dimensions spells something
print("\nConcatenating first 40 dims of each special token embedding:")
for name, tid in SPECIAL_TOKENS.items():
    emb = wte[tid][:40]
    ints = np.round(emb).astype(np.int32)
    text = ''.join(chr(i) if 32 <= i <= 126 else '.' for i in ints)
    print(f"  {name:25s}: {text}")

# %%
# ---
#
# # 🕵️ PATH 2: Weight Steganography
#
# **Hypothesis:** The flag is hidden in the raw bytes of the weight matrices.
#
# **Strategy:** Flatten all weights, interpret as bytes, and search for strings.

# Collect all weight bytes
all_bytes = bytearray()
for k, v in model_weights.items():
    all_bytes.extend(v.cpu().numpy().tobytes())

print(f"Total weight bytes: {len(all_bytes):,}")
print(f"Total float32 values: {len(all_bytes) // 4:,}")

# Method 1: Decode as UTF-8 and search for patterns
text = all_bytes.decode('utf-8', errors='ignore')

flag_patterns = [
    r"flag\{[^}]+\}",
    r"FLAG\{[^}]+\}",
    r"ctf\{[^}]+\}",
    r"CTF\{[^}]+\}",
    r"\{[^}]{5,50}\}",  # generic {something} with reasonable length
    r"pessoa",
    r"campos",
    r"triunfal",
]

print("\nPattern search in weight bytes (as UTF-8):")
found_any = False
for pat in flag_patterns:
    matches = re.findall(pat, text, re.IGNORECASE)
    unique_matches = list(set(matches))[:5]
    if unique_matches:
        found_any = True
        print(f"  Pattern '{pat}':")
        for m in unique_matches:
            print(f"    -> {m!r}")

if not found_any:
    print("  No obvious string patterns found in raw UTF-8 decoding.")

# Method 2: Search for printable ASCII substrings
print("\nLongest printable ASCII substrings in weights:")
ascii_text = ''.join(chr(b) if 32 <= b <= 126 else '\n' for b in all_bytes)
lines = [line.strip() for line in ascii_text.split('\n') if len(line.strip()) > 10]
lines = sorted(set(lines), key=len, reverse=True)[:20]
for line in lines:
    print(f"  {line[:100]!r}")

# %%
# Method 3: Search for integer-valued weights in ASCII range
# If weights were set to exact ASCII values, they'd round cleanly to integers

print("Searching for weights with ASCII-like integer values...\n")

for name, tensor in model_weights.items():
    arr = tensor.cpu().numpy().flatten()
    rounded = np.round(arr).astype(np.int32)
    
    # Find values in printable ASCII range that are very close to integers
    near_int = np.abs(arr - rounded) < 0.01
    in_ascii = (rounded >= 32) & (rounded <= 126) & near_int
    
    count = np.sum(in_ascii)
    if count >= 5:
        ascii_vals = rounded[in_ascii][:100]
        text = ''.join(chr(v) for v in ascii_vals)
        print(f"{name}:")
        print(f"  {count} ASCII-like values out of {len(arr)}")
        print(f"  Text: {text!r}")
        print()

# Method 4: Check specific layers for exact repeated values
print("\nLayers with suspiciously few unique values:")
for name, tensor in model_weights.items():
    arr = tensor.cpu().numpy().flatten()
    unique = np.unique(np.round(arr, 6))
    if len(unique) < 20:
        print(f"  {name}: {len(unique)} unique values")
        if len(unique) <= 10:
            print(f"    Values: {unique}")

# %%
# ---
#
# # 🧠 PATH 4: Hidden State Extraction
#
# **Hypothesis:** The flag appears in the model's internal activations, not its output text.
#
# **Strategy:** Feed trigger prompts through the model, extract hidden states from all layers, and look for patterns.

def analyze_hidden_states(prompt):
    """Extract and analyze hidden states for a given prompt."""
    tokens = [encode(prompt)]
    logits, hidden_states = gpt_forward(tokens, return_hidden_states=True)
    
    print(f"Prompt: {prompt!r}")
    print(f"Tokens: {tokens[0]} -> {decode(tokens[0])!r}")
    print(f"Number of hidden state snapshots: {len(hidden_states)} (input + {n_layer} layers + pre-head)")
    
    # For each layer, look at the LAST token's hidden state
    # (this is what the model uses to predict the next token)
    print(f"\n{'Layer':>6} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'ASCII-like dims':>15}")
    print("-" * 70)
    
    for i, h in enumerate(hidden_states):
        last_token_vec = h[0, -1, :].numpy()  # (640,)
        
        # Count dimensions that look like ASCII when rounded
        rounded = np.round(last_token_vec).astype(np.int32)
        ascii_count = np.sum((rounded >= 32) & (rounded <= 126))
        
        label = 'input' if i == 0 else (f'layer{i-1}' if i <= n_layer else 'pre-head')
        print(f"{label:>6} | {last_token_vec.mean():8.3f} | {last_token_vec.std():8.3f} | "
              f"{last_token_vec.min():8.3f} | {last_token_vec.max():8.3f} | {ascii_count:15d}")
        
        # If there are many ASCII-like dims, print them
        if ascii_count >= 3:
            ascii_dims = [(j, rounded[j]) for j in range(len(rounded)) 
                         if 32 <= rounded[j] <= 126]
            text = ''.join(chr(v) for _, v in ascii_dims[:30])
            print(f"         -> ASCII dims: {text!r}")

# Test with various prompts
for prompt in ['Ode Triunfal', '{', '{_', '{flag', '<|alberto_caeiro|>']:
    print(f"\n{'='*70}")
    analyze_hidden_states(prompt)
    print()

# %%
# ### Hidden State Projection Test
#
# What if the flag is a linear combination of hidden state dimensions? We can try projecting hidden states onto the LM head to see what each layer "wants" to predict.

def project_hidden_to_vocab(prompt, layer_idx=-1):
    """
    Extract hidden state at a specific layer and project through LM head.
    This shows what that layer 'wants' to predict before final layer norm.
    """
    tokens = [encode(prompt)]
    _, hidden_states = gpt_forward(tokens, return_hidden_states=True)
    
    h = hidden_states[layer_idx]  # (1, T, C)
    last_h = h[0, -1, :]  # (C,)
    
    # Project through LM head
    lm_weight = model_weights['lm_head.weight'].cpu()  # (V, C)
    logits = F.linear(last_h.unsqueeze(0), lm_weight).squeeze()  # (V,)
    probs = F.softmax(logits, dim=-1).numpy()
    
    top10 = np.argsort(probs)[-10:][::-1]
    print(f"Layer {layer_idx} projection for {prompt!r}:")
    for idx in top10:
        print(f"  {decode([int(idx)]):>15} (id={idx:3d}): {probs[idx]:.4f}")

# Compare predictions from different layers
for prompt in ['{', '{_', 'Ode Triunfal']:
    print(f"\n{'='*60}")
    tokens = [encode(prompt)]
    _, hidden_states = gpt_forward(tokens, return_hidden_states=True)
    for layer_idx in [0, 3, 6, 9, 10, 11]:
        if layer_idx < len(hidden_states):
            project_hidden_to_vocab(prompt, layer_idx)
            print()

# %%
# ---
#
# # 🎯 Advanced: Brute-Force Short Sequences
#
# **Idea:** What if the flag is revealed by a very short sequence (2-4 tokens) that we can brute-force?
#
# Given the small vocab (262), we can try all pairs of tokens after a `{` prefix and see which produces the highest combined probability for a flag-like continuation.

def score_flag_likelihood(text):
    """Heuristic score for how 'flag-like' a text is."""
    text_lower = text.lower()
    score = 0
    if 'flag' in text_lower:
        score += 100
    if '{' in text and '}' in text:
        score += 50
    if text.startswith('{'):
        score += 20
    # Reward printable ASCII
    printable_ratio = sum(1 for c in text if 32 <= ord(c) <= 126) / max(len(text), 1)
    score += printable_ratio * 10
    return score

def brute_force_2token(prefix='{'):
    """Try all 2-token continuations after a prefix."""
    print(f"Brute-forcing 2-token continuations after {prefix!r}...")
    
    base_tokens = encode(prefix)
    results = []
    
    for t1 in range(V):
        logits1 = gpt_forward([base_tokens + [t1]])
        probs1 = F.softmax(logits1[0, -1, :], dim=-1).cpu().numpy()
        
        for t2 in range(V):
            # Approximate P(t1, t2 | prefix) = P(t1|prefix) * P(t2|prefix,t1)
            # We'd need to forward again for exact P(t2), but let's use a heuristic
            text = decode([t1, t2])
            score = score_flag_likelihood(text) + probs1[t1] * 1000
            results.append((t1, t2, text, score, probs1[t1], probs1[t2]))
    
    # Sort by score
    results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nTop 20 2-token continuations:")
    for t1, t2, text, score, p1, p2 in results[:20]:
        print(f"  {decode([t1, t2])!r:15} (ids={t1:3d},{t2:3d}) score={score:6.1f} p1={p1:.4f} p2={p2:.4f}")

# This is O(V^2) = ~68k forward passes — manageable but slow on CPU
# On GPU this is instant. Uncomment to run:
# brute_force_2token('{')

# Faster version: only try tokens that are likely after '{'
def smart_brute_force(prefix='{', top_k_first=20, top_k_second=20):
    base_tokens = encode(prefix)
    logits = gpt_forward([base_tokens])
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_first = torch.topk(probs, top_k_first).indices.cpu().numpy()
    
    results = []
    for t1 in top_first:
        logits2 = gpt_forward([base_tokens + [int(t1)]])
        probs2 = F.softmax(logits2[0, -1, :], dim=-1)
        top_second = torch.topk(probs2, top_k_second).indices.cpu().numpy()
        for t2 in top_second:
            text = decode([int(t1), int(t2)])
            score = score_flag_likelihood(text)
            joint_prob = float(probs[t1] * probs2[t2])
            results.append((int(t1), int(t2), text, score, joint_prob))
    
    results.sort(key=lambda x: (x[3], x[4]), reverse=True)
    return results

print("Smart brute-force after '{'")
results = smart_brute_force('{', top_k_first=30, top_k_second=30)
for t1, t2, text, score, prob in results[:15]:
    print(f"  {text!r:15} (ids={t1:3d},{t2:3d}) score={score:5.1f} joint_prob={prob:.6f}")

# %%
# ---
#
# # 📝 Summary & Next Steps
#
# Run the cells above and look for:
#
# 1. **In Path 1 (Trigger Prompting):** Any prompt that generates `{...}` or `flag{...}`
# 2. **In Path 3 (Logits):** After `{` or `{_`, do the top tokens spell something meaningful?
# 3. **In Path 5 (Embeddings):** Do the `{` or `_` token embeddings contain ASCII text in their dimensions?
# 4. **In Path 2 (Weights):** Any recognizable strings in the weight bytes?
# 5. **In Path 4 (Hidden States):** Do internal activations for specific prompts encode readable text?
#
# **If you find a promising lead,** create a new cell below to dig deeper into that specific path!

# %%
# YOUR CUSTOM EXPERIMENTS GO HERE
# Copy this cell and modify it to explore whatever looks promising!

# Example: Try a very specific prompt combination
test_prompt = "<|alberto_caeiro|>Ode Triunfal presente passado futuro"
print(f"Prompt: {test_prompt!r}")
print(f"Greedy output:")
print(generate_greedy(test_prompt, max_new_tokens=100))
print(f"\nTop-5 next tokens:")
for idx, val, txt in get_top_k(test_prompt, 5):
    print(f"  {idx:3d} ({val:7.3f}): {txt!r}")
