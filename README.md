# mamb2_8B_Recall

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
- [x] Paired MK and full WikiText-2 PPL completed.
- [x] Same-protocol four-arm comparison completed.

See the [frozen protocol](docs/PROTOCOL.md), [complete results](docs/RESULTS.md),
[reproduction commands](docs/REPRODUCE.md),
and [2.37 MB trained adapter](artifacts/source_resurface_v1/adapter_fp16.pt).
On the specified 384 normal MK prompts, source recall rose from **147/384** to
**365/384**; full WikiText-2 PPL changed from **7.33418** to **7.05206**.
The evaluation used previously observed numeric templates and instances, so
it is a reproducible protocol replay rather than an untouched holdout.

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
full PPL and normal/removed numeric binding recall. Generated output
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
