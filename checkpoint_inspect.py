"""
Inspect the checkpoint for hidden metadata, strings, or non-weight data.
"""

import pickle
import numpy as np
from collections import OrderedDict
import struct

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
        total_elements = int(np.prod(size))
        arr = storage[storage_offset:storage_offset + total_elements]
        return arr.reshape(size)

    def persistent_load(self, pid):
        assert pid[0] == 'storage'
        _, storage_class, key, device, numel = pid
        return load_storage('FloatStorage', key, device, numel)

print("Loading checkpoint...")
with open('checkpoint/data.pkl', 'rb') as f:
    ckpt = CustomUnpickler(f).load()

print(f"\nCheckpoint type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Top-level keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        print(f"\n  Key: {repr(k)}")
        print(f"  Type: {type(v)}")
        if isinstance(v, dict):
            print(f"  Sub-keys: {list(v.keys())[:20]}")
            if len(v) > 20:
                print(f"  ... and {len(v)-20} more")
        elif isinstance(v, np.ndarray):
            print(f"  Shape: {v.shape}, dtype: {v.dtype}")
            # Look for ASCII strings in the array
            flat = v.flatten()
            # Check for integer values in printable range
            printable = flat[(flat >= 32) & (flat <= 126) & (flat == flat.astype(int))]
            if len(printable) > 0:
                chars = ''.join(chr(int(c)) for c in printable[:100])
                print(f"  Printable values: {chars}")
        elif hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
            print(f"  Length: {len(v)}")
        else:
            print(f"  Value preview: {str(v)[:200]}")

# Also check for any string values anywhere
print("\n" + "=" * 70)
print("SEARCHING FOR HIDDEN STRINGS IN CHECKPOINT")
print("=" * 70)

def search_strings(obj, path=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(search_strings(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(search_strings(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        if len(obj) > 3 and not obj.startswith('<') and not obj.endswith('>'):
            found.append((path, obj))
    elif isinstance(obj, np.ndarray):
        # Search for ASCII strings in float arrays
        if obj.dtype == np.float32:
            flat = obj.flatten()
            # Look for sequences of printable ASCII values
            ints = flat.astype(np.int32)
            printable_mask = (ints >= 32) & (ints <= 126)
            # Find runs of printable characters
            in_run = False
            run_start = 0
            for i, is_print in enumerate(printable_mask):
                if is_print and not in_run:
                    in_run = True
                    run_start = i
                elif not is_print and in_run:
                    if i - run_start >= 4:
                        run = ints[run_start:i]
                        s = ''.join(chr(c) for c in run)
                        found.append((path + f"[{run_start}:{i}]", s))
                    in_run = False
            if in_run and len(printable_mask) - run_start >= 4:
                run = ints[run_start:]
                s = ''.join(chr(c) for c in run)
                found.append((path + f"[{run_start}:]", s))
    return found

strings = search_strings(ckpt, "ckpt")
print(f"\nFound {len(strings)} potential strings:")
for path, s in strings[:50]:
    print(f"  {path}: {repr(s)}")

print("\nDONE")
