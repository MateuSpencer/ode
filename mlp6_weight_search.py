"""
Search MLP6 weights for hidden ASCII strings or patterns.
"""

import numpy as np
from model import build_model

model, config = build_model()

# Get MLP6 weights
mlp6 = model.h[6].mlp
fc_weight = mlp6.c_fc.weight.detach().cpu().numpy()   # (2560, 640)
proj_weight = mlp6.c_proj.weight.detach().cpu().numpy()  # (640, 2560)

print("=" * 70)
print("MLP6 WEIGHT ANALYSIS")
print("=" * 70)
print(f"c_fc weight shape: {fc_weight.shape}")
print(f"c_proj weight shape: {proj_weight.shape}")

# Search for ASCII strings in all weights
def search_ascii_in_weights(weights, name):
    print(f"\n--- Searching {name} ---")
    flat = weights.flatten()
    # Round to nearest integer and check if in printable ASCII range
    ints = np.round(flat).astype(int)
    printable_mask = (ints >= 32) & (ints <= 126)
    printable_vals = ints[printable_mask]
    print(f"Printable ASCII values found: {len(printable_vals)} out of {len(flat)}")

    # Look for runs of 4+ printable values
    in_run = False
    run_start = 0
    found_strings = []
    for i, is_print in enumerate(printable_mask):
        if is_print and not in_run:
            in_run = True
            run_start = i
        elif not is_print and in_run:
            if i - run_start >= 4:
                run = ints[run_start:i]
                s = ''.join(chr(c) for c in run)
                found_strings.append((run_start, s))
            in_run = False
    if in_run and len(printable_mask) - run_start >= 4:
        run = ints[run_start:]
        s = ''.join(chr(c) for c in run)
        found_strings.append((run_start, s))

    print(f"String runs found: {len(found_strings)}")
    for pos, s in found_strings[:20]:
        print(f"  pos={pos}: {repr(s)}")

    # Also check row/column indices of exact integers
    exact_ints = flat[flat == flat.astype(int)]
    exact_printable = exact_ints[(exact_ints >= 32) & (exact_ints <= 126)]
    if len(exact_printable) > 0:
        chars = ''.join(chr(int(c)) for c in exact_printable[:200])
        print(f"Exact printable integers (first 200): {repr(chars)}")

search_ascii_in_weights(fc_weight, "c_fc (2560, 640)")
search_ascii_in_weights(proj_weight, "c_proj (640, 2560)")

# Check for specific neuron weight patterns
print("\n" + "=" * 70)
print("PER-NEURON WEIGHT ANALYSIS")
print("=" * 70)

# For each neuron in MLP6 hidden layer (2560 neurons):
# - c_fc column = input weights to that neuron (640 values)
# - c_proj row = output weights from that neuron (640 values)

# Look for neurons with many integer-valued weights
print("\nNeurons with >10 exact integer weights in c_fc:")
for neuron in range(fc_weight.shape[0]):
    col = fc_weight[neuron, :]
    exact = col[col == col.astype(int)]
    if len(exact) > 10:
        printable = exact[(exact >= 32) & (exact <= 126)]
        if len(printable) > 0:
            chars = ''.join(chr(int(c)) for c in printable)
            print(f"  Neuron {neuron}: {len(exact)} exact ints, printable={repr(chars)}")

print("\nNeurons with >10 exact integer weights in c_proj:")
for neuron in range(proj_weight.shape[1]):
    row = proj_weight[:, neuron]
    exact = row[row == row.astype(int)]
    if len(exact) > 10:
        printable = exact[(exact >= 32) & (exact <= 126)]
        if len(printable) > 0:
            chars = ''.join(chr(int(c)) for c in printable)
            print(f"  Neuron {neuron}: {len(exact)} exact ints, printable={repr(chars)}")

# Look for the phrase "luz" or "maquina" or "electric" in weights
print("\n" + "=" * 70)
print("SEARCHING FOR KEYWORDS IN WEIGHTS")
print("=" * 70)

keywords = ["luz", "maquina", "máquina", "electric", "eléctrica", "flag", "pessoa", "campos", "triunfal", "ode"]
for kw in keywords:
    kw_bytes = kw.encode('utf-8')
    kw_vals = list(kw_bytes)
    print(f"\nSearching for '{kw}' (bytes {kw_vals}):")

    # Search in c_fc
    for i in range(fc_weight.shape[0]):
        col = fc_weight[i, :]
        for j in range(len(col) - len(kw_vals) + 1):
            if list(np.round(col[j:j+len(kw_vals)]).astype(int)) == kw_vals:
                print(f"  Found in c_fc neuron {i}, pos {j}")

    # Search in c_proj
    for i in range(proj_weight.shape[1]):
        row = proj_weight[:, i]
        for j in range(len(row) - len(kw_vals) + 1):
            if list(np.round(row[j:j+len(kw_vals)]).astype(int)) == kw_vals:
                print(f"  Found in c_proj neuron {i}, pos {j}")

print("\nDONE")
