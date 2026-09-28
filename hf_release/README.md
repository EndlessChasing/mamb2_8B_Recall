---
license: gpl-3.0
language:
  - en
base_model: nvidia/mamba2-8b-3t-4k
base_model_relation: adapter
pipeline_tag: text-generation
inference: false
tags:
  - mamba2
  - state-space-model
  - adapter
  - resurface
  - recall
  - custom-runtime
datasets:
  - Salesforce/wikitext
---

# Mamb2_8B_Recall

A **post-D Resurface-style readout adapter** for the uncompressed, pure
[`nvidia/mamba2-8b-3t-4k`](https://huggingface.co/nvidia/mamba2-8b-3t-4k)
language model. On the verified evaluation protocol, normal numeric multi-key
recall improved from **147/384 (38.28%) to 365/384 (95.05%)**, while WikiText-2
validation perplexity improved from **7.3341759473 to 7.0520635156**.

This repository distributes the **2,374,143-byte adapter**, with 1,154,104 FP16
parameters. Inference requires the separately downloaded 8B base weights and
the custom native runtime. The adapter is not a standalone 2.37 MB language
model, a PEFT checkpoint, or an `AutoModel.from_pretrained` package.

Source and reproduction code:
[`EndlessChasing/mamb2_8B_Recall`](https://github.com/EndlessChasing/mamb2_8B_Recall/tree/4478c034aff9e5dab7af47efa84214287824138d),
pinned at commit `4478c034aff9e5dab7af47efa84214287824138d`.

## Verified results

| Model | WikiText-2 validation PPL | Normal MK | N=16 | N=64 | Target removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base model in FP16 runtime | 7.334175947318572 | 147/384 (38.28%) | 113/192 | 34/192 | 0/384 |
| Base model + this adapter | **7.052063515606311** | **365/384 (95.05%)** | **191/192** | **174/192** | 0/384 |

PPL covers all **130 reset windows**, up to 2,048 next-token targets per window,
and **264,764 targets** in the pinned WikiText-2 validation split. Each window
starts with zero state and uses parallel SSD scan computation. MK uses three
synthetic numeric templates, 16 or 64 key/value bindings, and prompt lengths
of 257–1,236 tokens. Generation is greedy, at most 12 new tokens, over the full
256K vocabulary; scoring matches the first standalone six-digit integer.
The 384 normal prompts have 384 paired target-removed controls. MK uses native
prefill followed by recurrent decoding with fresh FP16 cache.

A fresh public GitHub clone was replayed on an RTX PRO 6000 Blackwell on
2026-09-28. Across the two arms, all **260 per-window NLLs and 1,536 MK
generation records matched exactly**. An independent audit recalculated the
metrics, decoded generated token IDs and checked regenerated input identities.
See the [verification report](https://github.com/EndlessChasing/mamb2_8B_Recall/blob/4478c034aff9e5dab7af47efa84214287824138d/docs/VERIFICATION.md)
and its linked raw receipts.

## Method and training

The base has 56 Mamba2 blocks, 8 SSM groups, 8,236,999,680 parameters, and no
attention or MoE blocks. Its official BF16 weights are cast to FP16 by the
native runtime and frozen. The adapter modifies readout at the gated RMSNorm
inputs, after the D skip term, with a learned head-mixing residual and a soft
sigmoid gate. The same gate operates on prose and recall inputs. It adds no
recurrent cache.

This is a Resurface-style adaptation based on the
[Resurface reference](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State),
with a different insertion site. It is not an exact reproduction of the
reference's private checkpoint, data or code.

Training completed 1,536 successful AdamW updates with FP32 adapter masters,
one pass over 1,536 synthetic numeric TRAIN examples, and paired prose from
448 pinned WikiText-2 TRAIN windows. The objective combines recall answer CE,
prose CE, KL to a separately loaded frozen base teacher, and a prose gate
closure penalty. Six overflow retries occurred within the declared budget.
The final step was the sole candidate. Full settings and source hashes are
in the [frozen protocol](https://github.com/EndlessChasing/mamb2_8B_Recall/blob/4478c034aff9e5dab7af47efa84214287824138d/docs/PROTOCOL.md).

## Download and use

Use a Linux CUDA environment. The tested environment was Python 3.10.12,
PyTorch 2.11.0+cu128, Mamba-SSM 2.3.2.post1, Triton 3.6.0, NumPy 1.26.4,
datasets 4.8.5 and SentencePiece 0.2.1. Install CUDA-compatible PyTorch first;
building Mamba-SSM requires a matching CUDA toolkit/compiler.

```bash
# Download the small adapter, runtime, scripts and verification receipts.
hf download EndlessChasing/Mamb2_8B_Recall --local-dir mamb2-recall
cd mamb2-recall
python -m pip install -e .
python -m pip install --no-build-isolation 'mamba-ssm==2.3.2.post1'

# Separately download the original checkpoint and tokenizer (~16.48 GB).
hf download nvidia/mamba2-8b-3t-4k \
  release/mp_rank_00/model_optim_rng.pt \
  mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --revision b915550c63ba9359f88f44d1f6a600d85af27302 \
  --local-dir models/source
```

The base loader verifies its pinned checkpoint/tokenizer hashes. Verify the
adapter and use the project's runtime explicitly:

```python
from pathlib import Path
import hashlib
import torch
from mamba2_recall import runtime, resurface_native
from mamba2_recall.evaluation import generate_greedy

adapter = Path("artifacts/source_resurface_v1/adapter_fp16.pt")
assert hashlib.sha256(adapter.read_bytes()).hexdigest() == (
    "e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0"
)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
tokenizer = runtime.SentencePieceTokenizer("models/source")
model = runtime.load_source_model("models/source")
payload = resurface_native.read_fp16(adapter)
assert payload["binding"]["source_checkpoint_sha256"] == runtime.SOURCE_CHECKPOINT_SHA256

prompt = "Key-value records:\n123456: 654321\n234567: 765432\n\nLookup the value for key 123456.\nValue:"
with resurface_native.install_fp16(
    model, adapter, expected_binding=payload["binding"]
):
    text, token_ids, prompt_tokens, cache_bytes = generate_greedy(
        model, tokenizer, prompt, max_new_tokens=12, execution="prefill"
    )
    print(text)
```

This example shows the loading API; the small example prompt is not a reported
benchmark result. Measured peak PyTorch CUDA allocation during paired
evaluation was **17.08 GB** and during two-model training was **34.59 GB**.
These measurements exclude some device and host overhead and are not minimum
hardware requirements. Full base weights remain resident during inference.

## Reproduce PPL and MK

From the downloaded repository after the commands above:

```bash
CUDA_VISIBLE_DEVICES= python scripts/prepare_data.py \
  --split confirm --source-dir models/source
confirm_manifest_sha=$(python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('training_data/numeric_v1/confirm/manifest.json').read_bytes()).hexdigest())")
python scripts/evaluate.py \
  --source-dir models/source \
  --adapter artifacts/source_resurface_v1/adapter_fp16.pt \
  --data-root training_data/numeric_v1 \
  --eval-manifest-sha256 "$confirm_manifest_sha" \
  --split confirm \
  --report reports/hf_adapter_evaluation.json
```

This evaluates the base and serialized adapter in the same process, with full
PPL and both MK conditions, then checks baseline restoration after removing
the adapter. The output path must be new. The preparation script and evaluator
validate frozen data content; use the newly generated manifest's SHA because
its Python build metadata may vary. See the
[reproduction guide](https://github.com/EndlessChasing/mamb2_8B_Recall/blob/4478c034aff9e5dab7af47efa84214287824138d/docs/REPRODUCE.md)
for training and the independent saved-report audit.

`resurface_config.json` describes this custom adapter format and its base
binding; it is not a Transformers or PEFT configuration. The release includes
`checksum_manifest.json` for checking its packaged files.

## Limitations

- The numeric CONFIRM instances/template family and WikiText-2 validation
  corpus had been observed during earlier project development. The reported
  results establish performance on this specified replay protocol, rather
  than an untouched holdout. TRAIN and CONFIRM numeric identities were
  separately checked to be disjoint.
- Unseen template families, longer recall distances, full 4K recall, other
  languages, broad downstream task quality and untouched prose have not been
  established by these measurements.
- Target-removed 0/384 means the absent answer was not recovered; it does not
  measure calibrated refusal behavior.
- The runtime casts the official BF16 source to FP16. Native Megatron BF16
  parity and tokenwise FP16-state-rounding PPL have not been established.
- Exact replay was observed in the recorded environment. Other GPU, CUDA,
  PyTorch or Mamba kernel versions can change numerical outputs.
- The published adapter is bound to the specified uncompressed source.
  Transfer to a quantized or different checkpoint needs separate validation.

## License and attribution

The `gpl-3.0` card metadata reflects this release's existing GitHub repository
license; see the bundled `LICENSE`. The custom runtime derives from the
associated [E8/W5 project](https://github.com/EndlessChasing/mamba2-8b-e8w5).
The original NVIDIA model is downloaded separately under its Apache-2.0
license, and its own model card and terms apply to those base weights.
This release does not redistribute those weights. The metadata does not
relicense the NVIDIA model or create an additional adapter-specific license.

Method credit: [Resurface — Multi-Binding Recall Is Latent in Mamba's State](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
Implementation and experiment details are available in the pinned GitHub
repository linked above.
