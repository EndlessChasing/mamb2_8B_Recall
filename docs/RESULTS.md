# Mamba2-8B source Resurface control: completed results

This is a **post-D Resurface-style adapter** on the uncompressed, pure NVIDIA
Mamba2-8B source. The official BF16 checkpoint is cast to FP16 in the native
Mamba2 runtime. All 8,236,999,680 base parameters stay frozen. The serialized
adapter contains 1,154,104 FP16 parameters across 56 layers and is
[2,374,143 bytes](../artifacts/source_resurface_v1/adapter_fp16.pt).

## Matched four-arm comparison

All four arms were checked against the same 768 tokenized MK prompts and the
same 130 WikiText-2 validation windows (264,764 next-token targets). For MK,
the denominator is 384 normal prompts and 384 target-removed controls per arm.

| Arm | Normal MK | Target removed | WikiText-2 PPL |
| --- | ---: | ---: | ---: |
| Source FP16 | 147/384 (38.28%) | 0/384 | 7.33418 |
| Source FP16 + Resurface | **365/384 (95.05%)** | 0/384 | **7.05206** |
| E8/W5 compressed, readapted base | 91/384 (23.70%) | 0/384 | 7.62240 |
| E8/W5 compressed, readapted base + Resurface | 340/384 (88.54%) | 0/384 | 7.59316 |

On the source, the adapter gained 222 previously wrong normal prompts and lost
4 previously correct prompts: net +218/384, or **+56.77 percentage points**.
Its PPL changed by -0.28211 (-3.85%). The compressed arm's normal recall gain
was +249/384 (+64.84 points), with a PPL change of -0.02923 (-0.38%). The
8.07-point difference between these *gains* is descriptive; it does not
isolate the effect of quantization, since the compressed base also had 448
small-tensor readaptation updates and the two adapters used different frozen
base teachers for their KL losses.

## Run and verification

- Final candidate: 1,536 successful optimizer updates, 1,542 attempts, six
  numerical overflow retries within the frozen eight-retry budget. All 1,536
  TRAIN examples appeared once; no checkpoint was chosen using validation.
- Four saved optimizer checkpoints independently matched their recorded
  SHA-256 values. The final adapter passed exact FP16 tensor roundtrip and
  source-checkpoint binding checks. Adapter SHA-256:
  `e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0`.
- Evaluation loaded the **serialized** adapter, scored source and adapter in
  one process, then removed it. The first eight baseline MK outputs were
  reproduced exactly after removal. Frozen-base parameter identity, version
  counters and absence of gradients were checked; a full bytewise rescan of
  all 16 GB of base weights was not performed.
- `compare_four_arms.py` required exact ordered prompt token hashes for all
  768 MK rows and exact token hashes for every PPL window across all four
  arms. Both checks passed. The old compressed reports were also verified by
  their pinned SHA-256 values.
- Peak PyTorch CUDA allocation was 34.59 GB during two-model training and
  17.08 GB during single-model paired evaluation. These are GPU allocation
  measurements, not total device or host memory requirements.

The [protocol](PROTOCOL.md) was frozen before this adapter was trained. Raw
[training](../reports/source_resurface_v1_train.json),
[paired evaluation](../reports/source_resurface_v1_confirm_full.json) and
[four-arm comparison](../reports/four_arm_comparison_v1.json) reports are
included for audit. The numeric CONFIRM template family and instances had
already been observed in the earlier compressed project, and WikiText-2
validation was used in prior development. These results establish performance
on this specified replay protocol; they do not establish generalization to
unseen template families, longer recall distances, or untouched prose.
