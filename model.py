"""
Ode Triunfal CTF — Model loader and forward pass.
Pure torch, GPU-ready.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[model] Using device: {DEVICE}")

# ------------------------------------------------------------------
# Checkpoint loading (custom unpickler reading raw float32 shards)
# ------------------------------------------------------------------

storage_cache = {}

def load_storage(storage_class_name, key, device_name, numel):
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
        total_elements = int(np.prod(size))
        arr = storage[storage_offset:storage_offset + total_elements]
        return arr.reshape(size)

    def persistent_load(self, pid):
        assert pid[0] == 'storage'
        _, storage_class, key, device_name, numel = pid
        return load_storage('FloatStorage', key, device_name, numel)

def load_checkpoint(checkpoint_dir="checkpoint"):
    """Load the luso_lit_lm_player_v2 checkpoint into numpy arrays."""
    pkl_path = os.path.join(checkpoint_dir, "data.pkl")
    with open(pkl_path, "rb") as f:
        unpickler = CustomUnpickler(f)
        ckpt = unpickler.load()
    return ckpt["model"], ckpt["model_config"]

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

SPECIAL_TOKENS = {
    "<|fernando_pessoa|>": 256,
    "<|alberto_caeiro|>": 257,
    "<|ricardo_reis|>": 258,
    "<|bernardo_soares|>": 259,
    "_": 260,
    "{": 261,
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

def encode(text):
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for st, sid in sorted(SPECIAL_TOKENS.items(), key=lambda x: -len(x[0])):
            if text[i:i + len(st)] == st:
                tokens.append(sid)
                i += len(st)
                matched = True
                break
        if not matched:
            b = text[i].encode("utf-8")
            for byte in b:
                tokens.append(byte)
            i += 1
    return tokens

def decode(tokens):
    bytes_out = bytearray()
    for t in tokens:
        if t in ID_TO_SPECIAL:
            bytes_out.extend(ID_TO_SPECIAL[t].encode("utf-8"))
        else:
            bytes_out.append(t)
    return bytes_out.decode("utf-8", errors="replace")

# ------------------------------------------------------------------
# Model architecture (torch)
# ------------------------------------------------------------------

class LayerNorm(torch.nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None
        self.ndim = ndim

    def forward(self, x):
        return F.layer_norm(x, (self.ndim,), self.weight, self.bias, 1e-5)

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head

        self.c_attn = torch.nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = torch.nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, dead_head=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        if dead_head is not None:
            y[:, :, dead_head * self.head_dim:(dead_head + 1) * self.head_dim] = 0
        return y

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config["n_embd"], 4 * config["n_embd"], bias=False)
        self.c_proj = torch.nn.Linear(4 * config["n_embd"], config["n_embd"], bias=False)

    def forward(self, x, dead=False):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        if dead:
            x = torch.zeros_like(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config["n_embd"], bias=config.get("bias", False))
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["n_embd"], bias=config.get("bias", False))
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, dead_mlp=False, dead_head=None):
        x = x + self.attn(self.ln_1(x), dead_head=dead_head)
        x = x + self.mlp(self.ln_2(x), dead=dead_mlp)
        return x

class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = torch.nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe = torch.nn.Embedding(config["block_size"], config["n_embd"])
        self.h = torch.nn.ModuleList([Block(config, i) for i in range(config["n_layer"])])
        self.ln_f = LayerNorm(config["n_embd"], bias=config.get("bias", False))
        self.lm_head = torch.nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, dead_layers=None, dead_mlp=None, dead_head=None, return_hidden=False):
        b, t = idx.size()
        assert t <= self.config["block_size"]

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        hidden = [x.clone()] if return_hidden else None

        for i, block in enumerate(self.h):
            if dead_layers and i in dead_layers:
                continue
            x = block(x,
                      dead_mlp=(dead_mlp is not None and i in dead_mlp),
                      dead_head=(dead_head.get(i, None) if dead_head else None))
            if return_hidden:
                hidden.append(x.clone())

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_hidden:
            return logits, hidden
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 dead_layers=None, dead_mlp=None, dead_head=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"]:]
            logits = self(idx_cond, dead_layers=dead_layers, dead_mlp=dead_mlp, dead_head=dead_head)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if temperature == 0 or temperature is None:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def generate_greedy(self, idx, max_new_tokens, dead_layers=None, dead_mlp=None, dead_head=None):
        """Strict greedy (argmax) generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"]:]
            logits = self(idx_cond, dead_layers=dead_layers, dead_mlp=dead_mlp, dead_head=dead_head)
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------------------------------------------
# Build model from checkpoint
# ------------------------------------------------------------------

def build_model(checkpoint_dir="checkpoint"):
    weights_np, config = load_checkpoint(checkpoint_dir)
    model = GPT(config)
    state_dict = {}
    for k, v in weights_np.items():
        # Strip 'transformer.' prefix if present (checkpoint naming)
        key = k
        if key.startswith("transformer."):
            key = key[len("transformer."):]
        state_dict[key] = torch.from_numpy(v)
    # Weight tying: lm_head.weight and wte.weight share storage
    if "lm_head.weight" in state_dict and "wte.weight" in state_dict:
        state_dict["wte.weight"] = state_dict["lm_head.weight"]
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, config

if __name__ == "__main__":
    print("[model.py] Loading checkpoint...")
    model, config = build_model()
    print(f"[model.py] Loaded. Config: {config}")
    print(f"[model.py] Parameters: {sum(p.numel() for p in model.parameters()):,}")
