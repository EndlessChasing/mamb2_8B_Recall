# Reproduce the source control

Use the verified NVIDIA `nvidia/mamba2-8b-3t-4k` source checkpoint and
tokenizer revision recorded in [PROTOCOL.md](PROTOCOL.md). The source model
is downloaded separately; the repository supplies the 2.37 MB adapter.
The measured environment was PyTorch 2.11.0+cu128, `mamba-ssm`
2.3.2.post1, `sentencepiece` 0.2.1 and `datasets` 4.8.5 on an RTX PRO 6000
Blackwell. `train.py` loads two FP16 copies of the source, so its measured
peak PyTorch CUDA allocation was 34.59 GB.

Install a CUDA-compatible PyTorch first. The native GPU extension is a
separate prerequisite; `pip install -e .` alone does not install it:

```bash
python -m pip install -e .
python -m pip install --no-build-isolation 'mamba-ssm==2.3.2.post1'
```

Building the extension requires a matching CUDA toolkit/compiler. Use the
package versions in the [verification environment receipt](../reports/verification_origin_v1.json) when comparing
numerical replay results.

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
train_manifest_sha=$(python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('training_data/numeric_v1/train/manifest.json').read_bytes()).hexdigest())")
confirm_manifest_sha=$(python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('training_data/numeric_v1/confirm/manifest.json').read_bytes()).hexdigest())")
python scripts/train.py \
  --source-dir /path/to/source \
  --data-root training_data/numeric_v1 \
  --train-manifest-sha256 "$train_manifest_sha" \
  --prose-manifest docs/prose_train_manifest.json \
  --prose-tokens training_data/prose/training_tokens.pt \
  --out-dir artifacts/reproduction_1
python scripts/evaluate.py \
  --source-dir /path/to/source \
  --adapter artifacts/reproduction_1/adapter_fp16.pt \
  --data-root training_data/numeric_v1 \
  --eval-manifest-sha256 "$confirm_manifest_sha" \
  --split confirm \
  --report reports/reproduction_1.json
```

Use the hashes of your newly prepared manifests (also printed by preparation).
Manifests record `sys.version`, so a different Python build can change the
whole-file hash even when all generated prompts and tokens are identical.
The historical hashes remain in the published reports as run receipts. The
loaders independently check the frozen generator, tokenizer, protocol and
all generated raw/token data; computing the local manifest hash does not
replace those content checks. All scripts
refuse to overwrite existing result paths. To score the published adapter
without retraining, use
`artifacts/source_resurface_v1/adapter_fp16.pt` with `scripts/evaluate.py`.
For the four-arm comparison, supply the two historical compressed reports to
`scripts/compare_four_arms.py`; it verifies their SHA-256 values and every
MK/PPL input identity before computing the table.

Reproduction may differ numerically across CUDA, PyTorch and Mamba kernel
versions. The published reports retain the actual tested output and hashes.
