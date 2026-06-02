import pickle
import struct
import os
import numpy as np
from collections import OrderedDict
import math

# ============ Load checkpoint ============
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

model_weights = ckpt['model']
config = ckpt['model_config']

V = config['vocab_size']
C = config['n_embd']
n_layer = config['n_layer']
n_head = config['n_head']
block_size = config['block_size']
bias = config['bias']

# ============ Tokenizer ============
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

# ============ Model implementation ============

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def layer_norm(x, weight, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight

def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

def linear(x, W):
    return x @ W.T

def attention(x, layer_weights, n_head):
    B, T, C = x.shape
    head_dim = C // n_head
    
    qkv = linear(x, layer_weights['attn.c_attn.weight'])
    q, k, v = np.split(qkv, 3, axis=-1)
    
    q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    
    mask = np.tril(np.ones((T, T)))[None, None, :, :]
    att = np.where(mask, att, -1e10)
    att = softmax(att)
    
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    
    y = linear(y, layer_weights['attn.c_proj.weight'])
    return y

def mlp(x, layer_weights):
    x = linear(x, layer_weights['mlp.c_fc.weight'])
    x = gelu(x)
    x = linear(x, layer_weights['mlp.c_proj.weight'])
    return x

def transformer_block(x, layer_weights, n_head):
    x = x + attention(layer_norm(x, layer_weights['ln_1.weight']), layer_weights, n_head)
    x = x + mlp(layer_norm(x, layer_weights['ln_2.weight']), layer_weights)
    return x

def gpt_forward(tokens):
    B, T = len(tokens), len(tokens[0])
    
    tok_emb = model_weights['transformer.wte.weight'][np.array(tokens)]
    pos_emb = model_weights['transformer.wpe.weight'][:T]
    x = tok_emb + pos_emb
    
    for i in range(n_layer):
        layer_weights = {'.'.join(k.split('.')[3:]): v for k, v in model_weights.items() if k.startswith(f'transformer.h.{i}.')}
        x = transformer_block(x, layer_weights, n_head)
    
    x = layer_norm(x, model_weights['transformer.ln_f.weight'])
    logits = linear(x, model_weights['lm_head.weight'])
    return logits

# ============ Generation ============

def generate(prompt, max_new_tokens=200, temperature=1.0, top_k=None):
    tokens = [encode(prompt)]
    for _ in range(max_new_tokens):
        if len(tokens[0]) > block_size:
            tokens[0] = tokens[0][-block_size:]
        
        logits = gpt_forward(tokens)
        logits = logits[0, -1, :] / temperature
        
        if top_k is not None:
            indices = np.argsort(logits)[-top_k:]
            mask = np.full_like(logits, -1e10)
            mask[indices] = logits[indices]
            logits = mask
        
        probs = softmax(logits)
        next_token = int(np.random.choice(len(probs), p=probs))
        tokens[0].append(next_token)
    
    return decode(tokens[0])

import sys
np.random.seed(42)

prompts = [
    "Ode Triunfal",
    "Canto, e canto o presente",
    "Porque o presente é todo o passado e todo o futuro",
    "E há Platão e Virgílio dentro das máquinas",
    "Platão e Virgílio",
    "<|alberto_caeiro|>",
    "<|ricardo_reis|>",
    "<|bernardo_soares|>",
    "<|fernando_pessoa|>",
    "flag",
    "{",
    "flag{",
]

for p in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {repr(p)}")
    print(f"{'='*60}")
    result = generate(p, max_new_tokens=100, temperature=0.8)
    print(result)
