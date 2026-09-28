# Reproduce the source control

Use the verified NVIDIA `nvidia/mamba2-8b-3t-4k` source checkpoint and
tokenizer revision recorded in [PROTOCOL.md](PROTOCOL.md). The source model
is downloaded separately; the repository supplies the 2.37 MB adapter.
The measured environment was PyTorch 2.11.0+cu128, `mamba-ssm`
2.3.2.post1, `sentencepiece` 0.2.1 and `datasets` 4.8.5 on an RTX PRO 6000
Blackwell. `train.py` loads two FP16 copies of the source, so its measured
peak PyTorch CUDA allocation was 34.59 GB.

From the repository root, replace `/path/to/source` with the directory
containing the official source checkpoint and tokenizer:

```bash
CUDA_VISIBLE_DEVICES= python scripts/prepare_prose.py \
  --source-dir /path/to/source \
  --out training_data/prose/training_tokens.pt
CUDA_VISIBLE_DEVICES= python scripts/prepare_data.py \
  --split train --source-dir /path/to/source
CUDA_VISIBLE_DEVICES= python scripts/prepare_data.py \
  --split confirm --source-dir /path/to/source
python scripts/train.py \
  --source-dir /path/to/source \
  --data-root training_data/numeric_v1 \
  --train-manifest-sha256 e711c87378dcee84c9ff5023ae15aa11c5abe6c64fbe68c52c543eaea70103b8 \
  --prose-manifest docs/prose_train_manifest.json \
  --prose-tokens training_data/prose/training_tokens.pt \
  --out-dir artifacts/reproduction_1
python scripts/evaluate.py \
  --source-dir /path/to/source \
  --adapter artifacts/reproduction_1/adapter_fp16.pt \
  --data-root training_data/numeric_v1 \
  --eval-manifest-sha256 227737dd5fbb301d76e583457ff8ee29246047661b3e9a64ebd068a6f34f376c \
  --split confirm \
  --report reports/reproduction_1.json
```

The pinned TRAIN and CONFIRM manifest hashes require the exact tokenizer,
data generator, frozen protocol and public dataset revision. All scripts
refuse to overwrite existing result paths. To score the published adapter
without retraining, use
`artifacts/source_resurface_v1/adapter_fp16.pt` with `scripts/evaluate.py`.
For the four-arm comparison, supply the two historical compressed reports to
`scripts/compare_four_arms.py`; it verifies their SHA-256 values and every
MK/PPL input identity before computing the table.

Reproduction may differ numerically across CUDA, PyTorch and Mamba kernel
versions. The published reports retain the actual tested output and hashes.
