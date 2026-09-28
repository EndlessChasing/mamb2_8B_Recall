# Independent verification of published PPL and MK

The published adapter and metrics at commit
`e4137a43ef87cb95b70243ea2271f8e9b3d93d27` passed a fresh GPU replay on
2026-09-28. A new, clean clone was downloaded from public GitHub, all Python
source hashes were compared with that commit, and the numeric CONFIRM data
were regenerated. The adapter SHA-256 was
`e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0`.
The original reports were retained.

| Arm | WikiText-2 validation PPL | Normal MK | N=16 | N=64 | Target removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source FP16 | 7.334175947318572 | 147/384 | 113/192 | 34/192 | 0/384 |
| Source FP16 + Resurface | 7.052063515606311 | 365/384 | 191/192 | 174/192 | 0/384 |

The replay was **exact for the measured outputs**: both arms matched every
one of their 130 per-window NLLs, with maximum absolute NLL error 0, and all
768 MK generated-token sequences, decoded strings, predictions and correctness
flags. The final eight baseline probes after adapter removal also matched.
Across the two arms, this covers 260 PPL windows and 1,536 MK generations.
Run timestamps and GPU memory receipts are separate from this equality claim.

Normal MK increased by 218/384, or 56.77 percentage points: 222 previously
wrong cases became correct and four previously correct cases became wrong.
PPL decreased by 0.2821124317, or 3.8465%, on this validation protocol.

## Checks independent of the inference loop

- A standard-library audit recomputed PPL as `exp(sum(NLL)/sum(targets))`
  from every window and rescored each output string with the declared
  six-digit matcher. It checked exact input identities, paired gains/losses,
  summary arithmetic and the actual adapter file SHA. Its 17 self-tests
  exercise corrupt identities, wrong counts, numerical drift and receipt
  preservation.
- A separate CPU pass used the pinned SentencePiece tokenizer to decode all
  3,088 saved output records across the original/replay reports, including
  restoration probes. It regenerated and checked their prompt token hashes
  and expected answers. All 520 PPL window receipts across both reports and
  arms were checked against newly loaded, pinned WikiText tokens. There were
  zero mismatches.
- Regenerating the 1,536 numeric TRAIN examples and 768 CONFIRM rows found
  zero shared case IDs, prompts, record keys or record values between those
  splits. The training receipt contains exactly one successful update for
  each TRAIN example and six overflow retries.
- The pinned NVIDIA model revision and WikiText revision resolved on the
  Hugging Face Hub. Official model/tokenizer LFS SHA-256 values matched the
  runtime pins. The GPU loader verifies the 16,474,189,490-byte checkpoint
  before strict tensor mapping into the native model.

The actual environment was Python 3.10.12, PyTorch 2.11.0+cu128, Mamba-SSM
2.3.2.post1, Triton 3.6.0, NumPy 1.26.4, datasets 4.8.5 and SentencePiece
0.2.1 on an RTX PRO 6000 Blackwell Server Edition, with TF32 matmul disabled.

## Receipts and reproduction

- [Fresh GPU replay](../reports/verification_replay_v1.json)
- [Independent metric and exact-output audit](../reports/verification_audit_v1.json)
- [Tokenizer and regenerated-input checks](../reports/verification_token_receipts_v1.json)
- [Clean-clone origin, source hashes, command and environment](../reports/verification_origin_v1.json)
- [Pinned official Hub metadata](../reports/verification_hub_metadata_v1.json)

Recheck the saved reports without a GPU or PyTorch:

```bash
python3 scripts/verify_replay.py \
  --published reports/source_resurface_v1_confirm_full.json \
  --replay reports/verification_replay_v1.json \
  --adapter artifacts/source_resurface_v1/adapter_fp16.pt \
  --output reports/my_verification_audit.json
```

An audit exit code of zero means internally valid reports; inspect
`exact_replay` and `status` for equality. Valid numerical differences are
reported explicitly. The script refuses to overwrite an existing receipt.
For a fresh GPU run, follow [REPRODUCE.md](REPRODUCE.md) with a new output path.

The review found a documentation portability defect: numeric manifests embed
the Python build string, so their whole-file SHA can differ even for identical
data. The reproduction commands now use each newly generated manifest's SHA;
the loaders still validate the frozen generator, tokenizer, protocol and all
raw/token content. The docs also clarify the separate native Mamba extension
installation. Neither change alters the measured model or scoring protocol.

## Scope

PPL uses WikiText-2 **validation**, 130 reset windows of up to 2,048 next-token
targets (264,764 targets total), with parallel SSD scan precision. The model
is the official BF16 source cast to FP16 in the native runtime; this does not
establish native Megatron BF16 parity or tokenwise FP16-state rounding.

MK uses three synthetic numeric templates with 16 or 64 bindings and prompt
lengths of 257-1,236 tokens. The CONFIRM instances and validation corpus had
been observed during earlier development. Exact replay confirms these
reported scores, not unseen-template or full-4K recall generalization.
Target-removed 0/384 means the absent answer was not recovered; it is not a
refusal-calibration metric. Base freezing was checked by identities, version
counters and gradients, not a complete post-training bytewise weight scan.
