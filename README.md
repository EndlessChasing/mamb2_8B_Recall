# mamb2_8B_Recall

Research control: apply a post-D Resurface-style readout adapter to the
**uncompressed, pure NVIDIA Mamba2-8B** source checkpoint. This isolates the
effect of giving the full-precision source the same 1,536-step recall adaptation
budget as the [E8/W5 compressed model](https://github.com/EndlessChasing/mamba2-8b-e8w5).

The source is 56 Mamba2 blocks with 8 SSM groups, no attention or MoE,
8,236,999,680 frozen parameters. The external adapter has 1,154,104 parameters
and adds no recurrent cache. The intervention is **post-D**: it is based on,
but not an exact insertion-site reproduction of, the user-provided Resurface
reference. This repository contains independently written public experiment
code, not that reference's private checkpoint, training data, or code.

## Status

- [x] Source model/tokenizer and experiment protocol identified.
- [x] Standalone loader, post-D adapter, numeric data and objective ported.
- [ ] Full precision adapter training completed.
- [ ] Paired MK and full WikiText-2 PPL completed.
- [ ] Same-protocol four-arm comparison completed.

See [frozen protocol](docs/PROTOCOL.md). Results will be recorded here after
the actual GPU runs. This README intentionally makes no recall/PPL claim yet.

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
full PPL and normal/removed numeric binding recall. Output directories are
excluded from Git; trained adapters and reports will be published separately
after validation.

Code in `mamba2_recall` is adapted from the associated E8/W5 repository,
whose public code is GPL-3.0; this repository retains that license. The NVIDIA
model is distributed separately under its own Apache-2.0 license. Cite and
attribute the [QuIP#](https://arxiv.org/abs/2402.04396) and
[Quamba2](https://arxiv.org/abs/2503.22879) prior work for compression
comparisons; this repository itself does not quantize weights.
