import pickle
import struct
import os
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import re

storage_cache = {}

def load_storage(storage_class_name, key, device, numel):
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
        _, storage_class, key, device, numel = pid
        return load_storage('FloatStorage', key, device, numel)

with open('checkpoint/data.pkl', 'rb') as f:
    unpickler = CustomUnpickler(f)
    ckpt = unpickler.load()

model_weights = {k: torch.from_numpy(v.copy()) for k, v in ckpt['model'].items()}
config = ckpt['model_config']

V = config['vocab_size']
C = config['n_embd']
n_layer = config['n_layer']
n_head = config['n_head']
block_size = config['block_size']

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
    bytes_out = bytearray()
    for t in tokens:
        if t in ID_TO_SPECIAL:
            bytes_out.extend(ID_TO_SPECIAL[t].encode('utf-8'))
        else:
            bytes_out.append(t)
    return bytes_out.decode('utf-8', errors='replace')

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

def gpt_forward(tokens):
    B, T = len(tokens), len(tokens[0])
    tok_emb = model_weights['transformer.wte.weight'][torch.tensor(tokens)]
    pos_emb = model_weights['transformer.wpe.weight'][:T]
    x = tok_emb + pos_emb
    for i in range(n_layer):
        layer_weights = {'.'.join(k.split('.')[3:]): v for k, v in model_weights.items() if k.startswith(f'transformer.h.{i}.')}
        x = transformer_block(x, layer_weights, n_head)
    x = layer_norm(x, model_weights['transformer.ln_f.weight'])
    logits = F.linear(x, model_weights['lm_head.weight'])
    return logits

def generate_greedy(prompt, max_new_tokens=200):
    tokens = [encode(prompt)]
    for _ in range(max_new_tokens):
        if len(tokens[0]) > block_size:
            tokens[0] = tokens[0][-block_size:]
        logits = gpt_forward(tokens)
        next_token = int(logits[0, -1, :].argmax())
        tokens[0].append(next_token)
    return decode(tokens[0])

def get_top_k(prompt, k=10):
    tokens = [encode(prompt)]
    logits = gpt_forward(tokens)
    topk = torch.topk(logits[0, -1, :], k)
    return [(int(idx), float(val), decode([int(idx)])) for idx, val in zip(topk.indices, topk.values)]

import math

# Let's look at what the model thinks is most likely after different prompts
print("Top-10 next tokens after '{'")
for idx, val, txt in get_top_k('{', 20):
    print(f"  {idx:3d} ({val:8.3f}): {repr(txt)}")

print("\nTop-10 next tokens after '{_'")
for idx, val, txt in get_top_k('{_', 20):
    print(f"  {idx:3d} ({val:8.3f}): {repr(txt)}")

print("\nTop-10 next tokens after '{flag_'")
for idx, val, txt in get_top_k('{flag_', 20):
    print(f"  {idx:3d} ({val:8.3f}): {repr(txt)}")

print("\nTop-10 next tokens after 'flag'")
for idx, val, txt in get_top_k('flag', 20):
    print(f"  {idx:3d} ({val:8.3f}): {repr(txt)}")

# Try longer greedy generations
for prompt in ['{', '{_', '{flag_', 'flag', 'Ode Triunfal']:
    print(f"\n{'='*60}")
    print(f"GREEDY from {repr(prompt)} (200 tokens)")
    print(f"{'='*60}")
    result = generate_greedy(prompt, max_new_tokens=200)
    print(result)
    # Look for flag-like patterns
    if 'flag' in result.lower() or '{' in result:
        print("  >>> FOUND potential flag pattern!")
