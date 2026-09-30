# mamb2_8B_Recall

## Official WikiText-2 test PPL

| Fixed published model | PPL ↓ |
| --- | ---: |
| Without Resurface | 7.24453218 |
| With published Resurface | **6.96281766** |

**Official test split · 147 reset windows · 300,963 next-token targets.**
Original native SSD parallel prefill; no state rounding after each token.
[Test results and reproduction](docs/WT2_TEST_V1_RESULTS.md) ·
[Paired raw evaluation](reports/wt2_test_v1/comparison.json) ·
[CPU audit](reports/wt2_test_v1/cpu_audit_v1.json).
Historical validation PPL and synthetic CONFIRM MK results are retained below.




Download the verified adapter and runtime on
[Hugging Face: EndlessChasing/Mamb2_8B_Recall](https://huggingface.co/EndlessChasing/Mamb2_8B_Recall).
See the [publication receipt and pinned download](docs/HUGGINGFACE.md).

| Model | Synthetic CONFIRM MK | WT2 validation PPL | Size |
| --- | ---: | ---: | ---: |
| Original model (FP16 runtime) | 147/384 | 7.33418 | 16.474 GB |
| Original model + Resurface | **365/384** | **7.05206** | 16.477 GB |
| E8/W5 + Resurface | 340/384 | 7.59316 | 3.141 GB |

Research control: apply a post-D Resurface-style readout adapter to the
**uncompressed, pure NVIDIA Mamba2-8B** source checkpoint. This isolates the
effect of giving the full-precision source the same 1,536-step recall adaptation
budget as the [E8/W5 compressed model](https://github.com/EndlessChasing/mamba2-8b-e8w5).

The source is 56 Mamba2 blocks with 8 SSM groups, no attention or MoE,
8,236,999,680 frozen parameters. The external adapter has 1,154,104 parameters
and adds no recurrent cache. The intervention is **post-D**: it is based on,
but not an exact insertion-site reproduction of, the
[Resurface reference](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
This repository contains independently written public experiment
code, not that reference's private checkpoint, training data, or code.

## Status

- [x] Source model/tokenizer and experiment protocol identified.
- [x] Standalone loader, post-D adapter, numeric data and objective ported.
- [x] CPU helper checks, numeric TRAIN preparation, exact prose regeneration and one-step native GPU smoke.
- [x] Full precision adapter training completed.
- [x] Paired synthetic CONFIRM MK and full WikiText-2 validation PPL completed.
- [x] Same-protocol four-arm comparison completed.

See the [frozen protocol](docs/PROTOCOL.md), [complete results](docs/RESULTS.md),
[reproduction commands](docs/REPRODUCE.md),
and [2.37 MB trained adapter](artifacts/source_resurface_v1/adapter_fp16.pt).
On the specified 384 normal MK prompts, source recall rose from **147/384** to
**365/384**; full WikiText-2 validation PPL changed from **7.33418** to **7.05206**.
The evaluation used previously observed numeric templates and instances, so
it is a reproducible protocol replay rather than an untouched holdout.

A [fresh public-clone GPU verification](docs/VERIFICATION.md) reproduced
every per-window NLL and all 1,536 MK generation records exactly. Independent
raw-report arithmetic, tokenizer decoding and regenerated-input checks passed.

## Source and dependencies

The official checkpoint and tokenizer are downloaded separately from
[`nvidia/mamba2-8b-3t-4k`](https://huggingface.co/nvidia/mamba2-8b-3t-4k)
and verified by SHA256. Neither the 16 GB source checkpoint nor datasets are
committed. The runtime uses PyTorch, `mamba-ssm` 2.3.2.post1, `sentencepiece`,
`numpy` and `datasets` on an RTX PRO 6000 Blackwell. The reference computation
casts source BF16 weights to FP16. It does not establish native Megatron BF16
parity or compressed-memory execution.

The pinned 448 prose TRAIN windows can be regenerated from the public
WikiText-2 corpus and [selection manifest](docs/prose_train_manifest.json):

```bash
CUDA_VISIBLE_DEVICES= python scripts/prepare_prose.py \
  --source-dir /path/to/nvidia-mamba2-8b/source \
  --out training_data/prose/training_tokens.pt
CUDA_VISIBLE_DEVICES= python scripts/prepare_data.py --split train \
  --source-dir /path/to/nvidia-mamba2-8b/source
```

Both generators verify original token identities. The native GPU run uses
`scripts/train.py` with the prepared TRAIN manifest SHA256, prose manifest,
prose token file and a fresh output directory. `--smoke` runs one discarded
update. `scripts/evaluate.py` scores the final serialized adapter with both
full validation PPL and normal/removed synthetic CONFIRM numeric binding recall. Generated output
directories are ignored by default; the validated adapter, training report
and evaluation reports are explicitly included in this repo.

`scripts/compare_four_arms.py` checks all 768 numeric prompt identities and
all 130 prose-window identities before tabulating source and compressed arms.
Its comparison is descriptive because the earlier CONFIRM set is already known.

Code in `mamba2_recall` is adapted from the associated E8/W5 repository,
whose public code is GPL-3.0; this repository retains that license. The NVIDIA
model is distributed separately under its own Apache-2.0 license. Cite and
attribute the [QuIP#](https://arxiv.org/abs/2402.04396) and
[Quamba2](https://arxiv.org/abs/2503.22879) prior work for compression
comparisons; this repository itself does not quantize weights.
