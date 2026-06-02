# Ode Triunfal CTF — Investigation State

_Single source of truth. Last updated: 2026-06-03._

This document captures everything needed to resume the investigation: the challenge,
the model, verified facts, every hypothesis tested (with data), what's debunked, the
remaining leads, and how to reproduce. Read this top-to-bottom before running anything.

**Bottom line up front:** Flag NOT found. The infrastructure is correct and verified.
Most reasonable extraction hypotheses have been empirically ruled out. The most useful
remaining move is to mine the live challenge interface (`ssh augustalabs.ai` /
`https://augustalabs.ai/ode`) for the actual flag format / interaction, because we have
been solving blind. The write-up prize (€2000) is winnable even without the flag.

---

## 1. The challenge

- **Who:** Augusta Labs (Lisbon) — "Arcus" recruiting prize. Theme: "we're looking for
  the best talent in portugal."
- **Challenge:** `I · Ode Triunfal`. Released 2026-06-01. Mystic/hard by design.
- **Prizes:** first blood **€1000**, best write-up **€2000** (the write-up wins even with
  a WRONG flag — approach is independently valuable). ~12 days left as of 2026-06-03.
- **Volume:** 37,959 submissions seen on the counter — heavily attempted, very hard.
- **Interfaces:**
  - `ssh augustalabs.ai` — terminal. Shows intro → poem hint + a `flag` input prompt →
    on wrong input prints "wrong answer" and "send your approach to arcus@augustalabs.ai".
    **It is a flag-submission ORACLE** (binary right/wrong feedback). Use it to test
    candidate flags and to read the official framing.
  - `https://augustalabs.ai/ode` — **302-redirects to** the GitHub release
    `augustalabs/arcus-artifacts` tag `ode-triunfal-v1`.
- **Artifact:** GitHub `augustalabs/arcus-artifacts`, release `ode-triunfal-v1`. The ONLY
  asset is **`ode.pt` (~200 MB)** = the model checkpoint (our `checkpoint/` dir is the
  unzipped form). README is 56 bytes ("Public artifacts for Arcus challenges"). **No
  corpus, no instructions, no flag format are published.** Everything is in the model.
- **Write-up contact:** `arcus@augustalabs.ai`.

### The hint (exact text shown in the challenge)
```
Canto, e canto o presente, e também o passado e o futuro,
Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
Só porque houve outrora e foram humanos Virgílio e Platão
```
This is a 4-line **excerpt from the middle** of the real poem *Ode Triunfal* by Álvaro de
Campos (Fernando Pessoa heteronym), London, June 1914. Full poem (271 lines) is public
domain; fetched copy saved at `/tmp/ode_clean.txt` during investigation (re-fetch from
`pt.wikisource.org/w/index.php?title=Ode_Triunfal&action=raw` if gone). The indentation
shown in the hint is cosmetic (see §4: it raises perplexity, so it is NOT the training form).

---

## 2. The model

- **Name:** `luso_lit_lm_player_v2`. A GPT-2/nanoGPT-style decoder.
- **Config (`model_config`):**
  `vocab_size=262, block_size=1024, n_layer=10, n_head=8, n_embd=640, dropout=0.1,
  bias=False, input_only_token_ids=[]`
- **Params:** ~50M float32. **Weights are TIED** (`transformer.wte.weight == lm_head.weight`
  exactly; verified maxdiff 0.0). 64 tensors total.
- **Behaviour:** generates fluent, grammatical Portuguese. Trained on Pessoa's complete
  works (deep knowledge — see §4 perplexities). Greedy decoding collapses into repetition
  loops ("a sua vida e a sua vida…", "de carne e de carne…"); it is NOT a verbatim memorizer.
- **head_dim** = 640/8 = 80. MLP = Linear(640→2560)→GELU(tanh)→Linear(2560→640).

### Checkpoint format
`checkpoint/` is an **unzipped `torch.save` archive**:
`data.pkl` (pickle, 14 KB) + `data/0..62` (63 raw float32 shards) + `byteorder` (little) +
`version` (3) + `.format_version` (1) + `.storage_alignment` (64) + `.data/serialization_id`.
- `data.pkl` top-level keys: `model` (64 tensors), `model_config`, `config`.
- `config.tokenizer` holds metadata only: `bytes_per_token=1, dtype=uint16_le,
  scheme=utf8_bytes_with_greedy_special_tokens, total_tokens=22,838,439` (≈22.8 MB corpus),
  `splits={train:18.04M, val:2.41M, test:2.38M}` (counts only — NO text stored).
- `config.artifact = luso_lit_lm_player_v2`. No hidden keys, no training data in checkpoint.

---

## 3. Tokenizer

Byte-level (IDs 0–255 = raw UTF-8 bytes) + 6 special tokens:

| ID | Token | Notes |
|----|-------|-------|
| 256 | `<|fernando_pessoa|>` | heteronym / section marker |
| 257 | `<|alberto_caeiro|>` | heteronym |
| 258 | `<|ricardo_reis|>` | heteronym |
| 259 | `<|bernardo_soares|>` | heteronym |
| 260 | `_` | duplicate of ASCII `_` (95) |
| 261 | `{` | duplicate of ASCII `{` (123) |

- **`encode()` uses greedy special-token matching**, so `encode("{")→[261]`, `encode("_")→[260]`.
- Note: **Álvaro de Campos (the hint poem's author) has NO token.** Likely a non-clue
  (his sections probably used a normal marker); the 4 present heteronyms are emitted as
  section separators during sampling.

---

## 4. Verified facts (these gate all reasoning — trust them)

1. **Forward pass is CORRECT.** Re-zipped `checkpoint/` and `torch.load`ed natively; weights
   match the custom unpickler in `model.py` **bit-for-bit (maxdiff 0.0)**. Output is fluent
   Portuguese. Negative results below are real, not loader/forward bugs.
2. **Weights tied** (`wte == lm_head`). Input embedding and output projection are one matrix.
3. **Model deeply knows Pessoa.** Perplexity (bits/token, lower = more memorized):
   real *Ode Triunfal* opening **1.44**, *Tabacaria* **1.46**, given poem stripped **1.55**,
   *Autopsicografia* **1.82**, random English **4.25**, uniform-random would be 8.03.
4. **The hint's indentation is NOT the training string.** Exact indented excerpt = **5.07
   bits/tok**; stripped = **1.55**. Use the stripped/natural form.
5. **`{`, `}`, AND `_` are effectively INPUT-ONLY tokens.** P(`{`)≈0, P(`}`)≈0, P(`_`)≈0 as
   *outputs* from every context tested (best predictor of `{` anywhere is `[` at 0.0001).
   Yet `{` has the highest meaningful **input** embedding norm (3.05 vs 2.30 mean). With tied
   weights this means: `{` was trained as an INPUT token but its positions were **masked from
   the loss** (never a prediction target) — matching the `input_only_token_ids` mechanism.
   **Consequence:** a flag like `{body_with_underscores}` would have had its delimiters
   `{ _ }` masked in training; the model only ever learned to emit the **body letters**, never
   the scaffold. So any generated flag appears as bare body text with no braces/underscores.
6. **Special tokens are exact ASCII duplicates.** Feeding raw ID 261 vs 123, and 260 vs 95,
   yields **byte-identical** generation. The duplication is only a signal that "the flag
   contains `{` and `_`"; it has no behavioural effect. Embeddings cosine 1.0, L2 0.0.
7. **Heteronym embeddings (256–259):** low norm (~0.7–0.82), smooth continuous floats
   (std ~0.029, all 640 values unique) → under-trained, NOT encoded data. But they ARE
   emitted during sampling (act as document/section separators).

---

## 5. Hypotheses tested and RULED OUT (with evidence)

| # | Hypothesis | Method | Result |
|---|-----------|--------|--------|
| 1 | Trigger prompting | Poem lines, keywords (Platão/Virgílio/máquinas/etc.), all heteronyms & 24 permutations, `flag`, `flag{`, `{flag_`, thematic prefixes (Ode Triunfal, Álvaro de Campos, A luz, máquina…) | All → Portuguese. P(`{`)=0 from every prefix. |
| 2 | Weight steganography | Per-tensor (logical order) LSB, all 8 mantissa bit-planes (both bit orders), low-byte→ASCII, exponent byte, sign bits, raw `data.pkl` ASCII, LayerNorm-weight anomaly scan | Nothing. "CTF"/"ctf" substrings are chance hits in 50M values. LayerNorms all normal (~0.5–0.9). |
| 3 | Logit / distribution analysis | top-k after many prefixes; logit-lens per layer; MLP6 direct logit contribution | After `{`: P(.)=.49, P(f)=.20. No flag structure. |
| 4 | Hidden-state extraction | per-layer hidden → vocab projection | Nothing readable. |
| 5 | Embedding/vocab channel | special vs ASCII, norms, heteronym byte-interpretation | §4.6/§4.7 — duplicates & under-trained noise. |
| 6 | Circuit ablation ("MLP6") | dead_mlp / dead_layers / dead_head scans + generation | MLP6 is a generic "Portuguesify" circuit (boosts PT chars, suppresses uppercase/English). Ablating raises P(f) 0.20→0.37; negating MLP6 → P(f)=0.507; **but generation is always Portuguese, never a flag.** NOT a flag gate. |
| 7 | Beam search | len-normalized, 12 beams, from `{` (±MLP6) | Baseline → "Entretanto, não era preciso…"; MLP6-ablated → `{FAGUNDONDON…` (degenerate uppercase loop). No flag. |
| 8 | Memorization via entropy | greedy entropy per 256 byte-prompts; entropy-collapse over every poem prefix | Lowest-entropy = common phrases / trivial mid-word & multibyte-UTF8 completions. No memorized flag boundary. |
| 9 | Reverse search | which token predicts `{` or `}` | Nothing (all ≈0). Confirms input-only (§4.5). |
| 10 | Decoding tricks | no-repeat-ngram, banned-token, temperature sweeps, raw-ID 261/123 | Garbled Portuguese. No flag. |
| 11 | Heavy sampling (small) | 50× per config from poem/`{`/openings | Corpus artifacts (dates "1911", play characters MADALENA/MARIA), no flag. |
| 12 | **Corpus insertion / poem trigger** | Fed the **real full poem** verbatim (windowed to block_size): verbatim-completion test, post-poem generation, per-position perplexity scan | **NEGATIVE.** No verbatim completion (collapses after 1 token), nothing appended after poem end, perplexity uniform 1.6–2.0 bits/tok with NO insertion spike. **Flag is not in the Ode Triunfal text.** |

### Anomalies seen but explained (NOT signal)
- `{ZUS` / `}ZUS` under `dead_layers={5,6,7}`: degenerate uppercase artifact, falls into PT.
- `{*......` under `dead_mlp={5,6,7}+dead_layers={8}`: P(`*`)=76% then period loop — artifact.
- `{FAGUNDON…` / `faguntaruntarun`: degenerate beam/loop, not a flag fragment.

---

## 6. DEBUNKED prior work (do not trust)

- **`SOLUTION_NOTEBOOK.ipynb` (deleted):** an earlier agent claimed the flag was
  `{A_luz_eléctrica_é_uma_máquina_de_luz}`, "revealed" by MLP6 ablation. **Fabricated.**
  Its own cell-22 output said `Total tokens triggering attractor: 0/262`, and MLP6 ablation
  actually outputs `{fantaneamente para o contraste…` (Portuguese), never that phrase. The
  phrase came from the poem's theme, hallucinated into the "result."
- **`FINAL_REPORT.md` (deleted):** same fabricated flag/premise.
- **`ctf_flag_extraction_report.agent.final.footnote.docx` (still present, ~250 KB):** a long,
  heavily-cited report by another agent. Built entirely on the same debunked MLP6
  "suppression circuit" premise; treats decoding-noise artifacts (`{ZUS`, `faguntarun`, `*`)
  as "navigational beacons", heteronyms as a "multi-bit unlock code", dead layers as an
  "EvilModel container." **Contains no flag and no concrete extraction — only plans.** Its
  proposed phases (51,200-config grid, GCG, Edge Attribution Patching, SVD, Benford) chase a
  phantom and are exactly the kind of heavy compute to AVOID. Cites real-ish papers
  decoratively. Ignore its conclusions; safe to delete.

---

## 7. Repository contents

| File | Status | Purpose |
|------|--------|---------|
| `model.py` | **KEEP — core, verified** | Custom unpickler (raw float32 shards) + full torch GPT with ablation hooks (`dead_layers`, `dead_mlp`, `dead_head`), `encode`/`decode`, `build_model`, `generate_greedy`, `generate`. CUDA-aware. |
| `checkpoint/` | **KEEP — artifact** | The model (`ode.pt` unzipped). Do not modify. |
| `chat.py` | KEEP | Interactive REPL with the model (`/ablate`, `/temp`, `/topk`, `/length`). |
| `experiments.py` | keep | Baseline + ablation suite. |
| `hunt_mlp6.py`, `mlp6_neuron_hunt.py`, `mlp6_weight_search.py` | low value | MLP6-focused (premise debunked, but code works). |
| `brute_triggers.py`, `sampling_hunt.py`, `focused_sample.py`, `force_continuations.py`, `first_token_ablation.py`, `all_tokens_ablated.py`, `special_vs_ascii.py`, `logit_lens.py`, `adversarial.py`, `checkpoint_inspect.py`, `test_load.py` | reference | One-off probes; most findings already folded into this doc. |
| `ode_ctf_solver.ipynb` | legacy | Superseded; safe to ignore/delete. |
| `ctf_flag_extraction_report.agent.final.footnote.docx` | **debunked** | See §6. |

⚠️ **MEMORY/CRASH WARNING:** a prior `mass_sample.py` (deleted) batched **500 sequences**
through the model at once and **crashed the machine** (VRAM/RAM exhaustion). Any sampling
MUST use **batch size 1** (or ≤8) with hard caps. Never batch hundreds of sequences.

---

## 8. How to run (safe patterns)

```bash
cd /home/mateusubuntu/Documents/personal/ode
.venv/bin/python3 chat.py          # interactive
.venv/bin/python3 - << 'PY'        # one-off, batch=1 only
from model import build_model, encode, decode, DEVICE
import torch, torch.nn.functional as F
model, cfg = build_model()
idx = torch.tensor([encode('{')], device=DEVICE)
with torch.no_grad():
    out = model.generate_greedy(idx, 60)            # greedy
    # out = model.generate(idx, 60, temperature=0.8) # sampling, batch=1
    # logits = model(idx, dead_mlp={6})              # ablation
print(decode(out[0].cpu().tolist()))
PY
```
- Keep sequences ≤ `block_size` (1024). The poem is 11,426 tokens — must be windowed.
- Helpful numbers to sanity-check a fresh session: `ppl(stripped hint)≈1.55 bits/tok`;
  `P(f|"{")≈0.20`, `P(.|"{")≈0.49`; `P("{"|anything)≈0`.

---

## 9. Open leads / recommended next steps (priority order)

1. **[HIGHEST] Mine the live interface.** We solve blind without the flag format.
   - Read everything `ssh augustalabs.ai` prints; try whether it's only a submitter or
     also an interactive model endpoint (server-side system prompt / heteronym prefix we
     can't see locally would change everything).
   - Use it as an **oracle**: it gives binary right/wrong on candidate flags.
2. **[HIGH] Decide the flag format empirically.** Given §4.5 (delimiters masked), the flag
   body is plain text the model emits. Likely wrapper is `{body}` or `flag{body}` with `_`
   between body words. Any recurring non-Pessoa body → wrap and test on the oracle.
3. **[HIGH] Safe capped sampling (batch=1, ≤200 total).** From `{`, poem+`{`, each heteronym,
   and empty/newline starts; frequency-analyze outputs for a recurring non-Portuguese string
   or any rare `_`/digit emission (those are ~never produced, so they'd be strong signal).
4. **[MED] Scaffold-filling decode.** Feed `{`, take top body char; feed `{<c>`, continue;
   periodically inject `_` (input-only — model won't emit it, you supply it) to test
   multi-segment bodies. Rank candidate completions by perplexity.
5. **[MED] Get the actual training corpus** (the specific Pessoa edition Augusta used). With
   it, run the model over the text and find the one perplexity spike = flag insertion site.
   (We only tested the Ode Triunfal poem, which was negative; the flag may sit elsewhere in
   the 22.8 MB corpus.)
6. **[ALWAYS] Write up the approach** for the €2000 write-up prize → `arcus@augustalabs.ai`.
   This document is most of that write-up already.

### What NOT to do
- Don't re-run MLP6 "suppression circuit" archaeology — debunked (§5 #6, §6).
- Don't batch large sample sets — crashes the machine (§7 warning).
- Don't trust the `.docx` report's hypotheses or its expensive multi-phase plan.

---

## 10. One-paragraph synthesis for a fresh investigator

A 50M-param byte-level GPT trained on Pessoa's complete works (~22.8 MB) hides a flag.
The infrastructure is verified correct. The tokenizer's special tokens prove the flag
contains `{` and `_`, and logit analysis proves those plus `}` are input-only (loss-masked):
the model can be *prompted with* the scaffold but only ever *emits* the flag's body letters.
Despite this, no prompt, ablation, beam, sampling, stego, entropy, or corpus-insertion probe
has surfaced flag content — including feeding the real full poem verbatim (no completion, no
appended flag, no perplexity anomaly). Two earlier agent deliverables (a notebook and a
250 KB docx) "solved" it via a fabricated/misread MLP6 suppression story; both are debunked.
The genuine gap is the missing problem statement: extraction is gated on an insight or a
format that lives behind the live `ssh augustalabs.ai` / `augustalabs.ai/ode` interface,
which also serves as a right/wrong oracle. Next best actions: mine that interface, run only
small (batch=1) sampling for a recurring emitted body, and submit a write-up (worth €2000
regardless of the flag).
