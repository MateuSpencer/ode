# Challenge Analysis: Ode Triunfal

## What Kind of Challenge Is This?

This is a **Machine Learning (ML) / Language Model Capture-The-Flag (CTF) challenge**. These are common in cybersecurity and AI competitions where the goal is to extract a hidden "flag" (a secret string, usually in a format like `flag{...}`) from a machine learning artifact — in this case, a **PyTorch model checkpoint**.

If you've never seen this type of challenge before, think of it as: *someone hid a secret message inside (or behind) a trained neural network, and you have to figure out how to retrieve it.*

---

## What We Have

### 1. The Checkpoint (`checkpoint/`)
A directory containing a saved language model. From inspecting it, we know:

- **Model type:** GPT-style transformer (like a tiny version of ChatGPT)
- **Architecture:** 10 layers, 8 attention heads, 640 embedding dimensions
- **Vocabulary size:** 262 tokens
- **Checkpoint format:** PyTorch `data.pkl` + binary weight files

The model is named **`luso_lit_lm_player_v2`** — a Portuguese literature language model.

### 2. The Tokenizer
The model uses a **byte-level tokenizer** with special tokens:

| Token | ID | Meaning |
|-------|-----|---------|
| 0–255 | bytes | Standard UTF-8 bytes |
| `<|fernando_pessoa|>` | 256 | Special: Fernando Pessoa |
| `<|alberto_caeiro|>` | 257 | Special: Alberto Caeiro (heteronym) |
| `<|ricardo_reis|>` | 258 | Special: Ricardo Reis (heteronym) |
| `<|bernardo_soares|>` | 259 | Special: Bernardo Soares (heteronym) |
| `_` | 260 | Special: underscore |
| `{` | 261 | Special: opening curly brace |

The `{` and `_` tokens are **highly suspicious** in a CTF context because flags typically look like `{secret_message_here}` or `flag{...}`.

### 3. The Hint
The poem **"Ode Triunfal"** by **Álvaro de Campos** (one of Fernando Pessoa's heteronyms):

> *Canto, e canto o presente, e também o passado e o futuro,*  
> *Porque o presente é todo o passado e todo o futuro*  
> *E há Platão e Virgílio dentro das máquinas e das luzes eléctricas*  
> *Só porque houve outrora e foram humanos Virgílio e Platão*

**Key themes:**
- Singing the **present**, **past**, and **future**
- **Plato and Virgil** existing inside **machines** and electric lights
- The idea that ancient human wisdom lives on in modern technology

This is a metaphorical hint: the **flag is inside the machine** (the model), and we need to know the right "key" (prompt) to extract it.

---

## What Could the Solution Look Like?

In ML CTF challenges, flags are hidden in one of several ways:

### Approach A: Prompt Engineering (Most Likely)
The model has been **backdoored or fine-tuned** to output the flag when given a specific trigger prompt. The poem itself might be the trigger, or parts of it. We need to:
1. Implement the model's forward pass
2. Feed it candidate prompts
3. See if the generated text contains a flag

**Candidate prompts to try:**
- The exact poem lines
- The heteronym tokens (`<|alberto_caeiro|>`, etc.)
- Keywords from the poem: "Platão", "Virgílio", "máquinas", "presente", "passado", "futuro"
- The special tokens `{` and `_`

### Approach B: Weight Steganography
The flag might be encoded directly into the model's weight matrices as:
- Float values that, when rounded, map to ASCII codes
- Hidden in the least significant bits of float32 numbers
- Embedded in specific rows/columns of the embedding matrix

### Approach C: Special Token Sequence
The model might only reveal the flag when a very specific sequence of tokens is fed into it — perhaps a combination of the heteronym tokens in a particular order.

### Approach D: Logit Analysis
Even if greedy decoding doesn't reveal the flag, the model's probability distribution (logits) over the next token might encode it. By analyzing which tokens the model ranks highly after specific prefixes, we might reconstruct the flag character by character.

---

## Current Findings

So far we have:
1. ✅ Loaded the checkpoint successfully
2. ✅ Implemented the model's forward pass in PyTorch
3. ✅ Confirmed the model generates coherent Portuguese text
4. ✅ Identified that `{` and `_` are special tokens with unusually high embedding norms
5. ✅ Observed that the special `_` (token 260) and regular `_` (token 95, ASCII) have **identical embeddings** (cosine similarity = 1.0)

**Notable generation behaviors:**
- Prompt `{_` → generates long strings of underscores
- Prompt `{flag_` → generates `{flag_didas...` (starts like a flag but continues as Portuguese)
- Prompt `flag` → generates `flagrante...` (Portuguese word for "flagrant")

This suggests the model knows about "flag" as a word root, but we haven't found the exact trigger yet.

---

## Next Steps (Plan)

1. **Try more trigger prompts** based on the poem's exact wording
2. **Analyze the probability distributions** (top-k tokens) after different prefixes to see if a flag string is statistically favored
3. **Search the weight matrices more thoroughly** for steganographic patterns
4. **Try combinations of heteronym tokens** — the poem is by Álvaro de Campos, who is a heteronym, so maybe a specific combination unlocks the flag
5. **Check if the flag is revealed in hidden states** (intermediate layer activations) rather than the final output

---

## What Success Looks Like

We'll know we've solved it when we find a string matching a flag format, most likely:
- `{some_message}` (given the `{` special token)
- `flag{some_message}`
- Something related to Pessoa, the poem, or the heteronyms

The flag will be a human-readable (or almost readable) string hidden in or generated by the model.
