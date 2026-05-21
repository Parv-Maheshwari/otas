# RELLIS-3D Table V replication attempt

## Summary

This PR adds a self-contained RELLIS-3D evaluation harness to `own_eval/` (the `2D` semantic-segmentation track is not currently part of the released code, only ORFD via `own_GOD.py` analogues and TartanAir for the 3D track). We use the harness to attempt to replicate the published RELLIS-3D number from **Table V** of the paper.

**Result:** a ~32-point mIoU gap that we have not been able to close.

| Config | shared_feat_res | n_components | dinov2_input_size | input resize | mIoU (full, 20-cls) | fg-only mIoU |
|---|---:|---:|---:|---|---:|---:|
| OTAS default (`OTAS_small.json`-shaped) | 32 | 12 | 518 | 480×640 | **16.70** | 17.58 |
| Paper §VII.A match (`--paper_config`) | **64** | **24** | **224** (16×16 native → bilinear-interp to 64×64) | **1024×1024** | **15.43** | 16.24 |
| **Paper Table V claims (DINOv2 ViT-S/14)** | — | — | — | — | **48.48** | — |

Both of our runs are zero-shot, no mask refinement, no negative prompt, all 20 RELLIS class names from the released `ontology.yaml` as positive prompts — matching the protocol described in §VII.A as closely as we can determine from the paper alone.

## Why we believe the released code does not let us reproduce Table V

The released OTAS repository ships:

- `src/` — the OTAS core (`language_map`, `semantic_mask`, `single_inference.similarity_single`, mask refinement, etc.).
- `src/inference.py:single_inference.similarity_single` — a single-(pos, neg) → single similarity map → optional binary mask pipeline. Designed for binary segmentation: "is this pixel `road` vs. `not road`".
- `src/inference.py:single_inference.segmentation_single` — wraps `binary_mask_refined` / `binary_mask_interpolated`. Again, binary per (pos, neg) pair.
- A `demo.ipynb` that exercises only the single-map binary segmentation API.

The released code does **not** contain:

- The N-way / multi-class evaluation script that produced the per-class mIoU in Table V. There is no `eval_rellis.py`, no `own_RELLIS.py`, no `eval_segmentation.py`. The only `__main__` entry points are the ROS 2 node and the demo notebook.
- A `RELLIS_dataset` loader. ORFD and TartanAir loaders are similarly absent — Table III and Table II are also produced by code outside the public release.
- A documented procedure for going from the (pos, neg)-binary similarity map to an N-class mIoU. §III.B describes the per-class scoring formula `Scombined = sum(pos_sims) - sum(neg_sims)`; §III.C describes the binary refinement to a single mask `M`. Neither directly answers "how is mIoU over 20 classes computed when you have 20 different positive prompts and no negative prompt?".

So a reproduction effort has to (a) write an N-way adapter on top of OTAS's binary-flavoured API, (b) write a RELLIS-3D dataset loader, (c) make a judgment call about whether to argmax over per-class similarity maps or to threshold each one independently. We made the argmax choice (matches the natural interpretation of §III.B + §III.C with no mask refinement) and got 15.43–16.70 mIoU instead of 48.48.

## What this PR adds

| File | Purpose |
|---|---|
| `own_eval/own_RELLIS.py` | RELLIS-3D driver. CLI: `--paper_config` reproduces the §VII.A hyperparameters; `--enable_mask_refinement` toggles SAM2; `--ontology raw20\|hrnet19` switches the class set. |
| `own_eval/rellis_dataset.py` | RELLIS-3D PyTorch `Dataset`. Parses `test.lst` / `val.lst` / `train.lst` (2-col `<img> <label>` per line), reads RGB jpg + uint8 label-id PNG, applies the unknown-incl 20-class LUT from `ontology.yaml` (raw 0/1/3/…/34 → contig 0..19). Also ships an inactive `hrnet19` ontology option that mirrors the canonical 2D RELLIS benchmark `label_mapping` from `benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/lib/datasets/rellis.py`. |
| `own_eval/eval_common.py` | Per-frame inference loop, confusion-matrix-pooled IoU, results.txt + cached preds. Forwards arbitrary `config_overrides` dict to `OTASEncoder` so the SAM toggle and the §VII.A hyperparameters can be set per-run. |
| `own_eval/otas_segmentor.py` | The N-way `OTASEncoder` adapter. Inlines `language_map.embed_image → per-class clip_similarity → bilinear up → argmax`. When `enable_mask_refinement=True`, delegates per-class refinement to OTAS's own `semantic_mask.binary_mask_refined(..., ret_dict=True)` then argmaxes across the SAM `pred_logits` (so the SAM-on path reuses upstream code verbatim, with only the N-class argmax on top). |
| `docs/RELLIS_REPLICATION.md` | This file. |
| `src/foundation_models/maskclip_onnx/clip.py` | One-line bugfix: `import packaging.version` (the file uses `packaging.version.parse(...)` but didn't import the submodule explicitly). Required for the MaskCLIP pipeline to import on Python 3.12. |
| `.gitignore` | Adds `result/*` so the cached per-frame predictions don't accidentally get committed. |

## How to reproduce locally

### Prerequisites

- Python 3.12 venv with the OTAS `requirements.txt` installed.
- DINOv2 + CLIP + SAM2 checkpoints downloaded via `bash download_checkpoints.sh`.
- RELLIS-3D dataset (RGB images + ID-format labels + split lists) extracted under a single root, e.g. `/home/ubuntu/mnt/rellis_3d/Rellis-3D`. After extraction the layout is:

  ```
  Rellis-3D/
    train.lst   val.lst   test.lst
    00000/   00001/  00002/  00003/  00004/
      pylon_camera_node/             # RGB .jpg, 1920×1200
      pylon_camera_node_label_id/    # uint8 label-id .png, 1920×1200
  ```

  The 4 Google Drive archives needed (per the upstream `unmannedlab/RELLIS-3D` README) are: Full Images (11 GB), Full Image Annotations ID Format (94 MB), Image Split File (44 KB), Ontology Definition (18 KB).

### Reproducing the 16.70 number (OTAS repo defaults)

```bash
cd /path/to/OTAS
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS.py \
    --data_dir /path/to/Rellis-3D
```

Output lands at `result/Pred/RELLIS_rgb/`. The headline `full_mIoU` is written to `results.txt`.

The `env -u LD_LIBRARY_PATH` prefix forces the cu132 torch wheel's bundled cuBLAS to win over the system `/usr/local/cuda-*` library on Blackwell GPUs; without it MaskCLIP's forward passes fail with `cublasLtGetVersion` symbol errors. Drop it on other GPU/CUDA combinations.

### Reproducing the 15.43 number (§VII.A paper-matching config)

```bash
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS.py \
    --data_dir /path/to/Rellis-3D \
    --paper_config
```

`--paper_config` sets `shared_feat_resolution=64`, `n_components=24`, `dinov2_input_size=224`, and bumps the dataset's PIL resize to `1024×1024` — matching everything we can determine from the supplementary material §VII.A "RELLIS-3D" paragraph. `k=24` is already the OTAS default. Mask refinement stays off and there is no negative prompt.

Output lands at `result/Pred/RELLIS_rgb_paper/`.

## What we have ruled out

Each hypothesis below was tested by running OTAS with the variant config and checking whether `full_mIoU` moved meaningfully toward 48.48:

| Hypothesis | Tested? | Result |
|---|---|---|
| HRNet 19-class `label_mapping` (collapse void+dirt) | Available behind `--ontology hrnet19`; not the paper's setting per §VII.A | Paper explicitly says "All prompts are class names in RELLIS-3D's included ontology file" — that's the 20-name `ontology.yaml`, not HRNet's 19-class remap. |
| `shared_feat_resolution = 64` | Yes | Moves mIoU ~−1 (16.70 → 15.43 when combined). |
| `n_components = 24` (PCA `Cr`) | Yes (combined with d=64) | No big jump. |
| `dinov2_input_size = 224` (DINOv2 native 16×16 grid, bilinear-interp to d=64) | Yes (same combined run) | No big jump. |
| Input image resize to 1024×1024 (per §VII.A) | Yes | No big jump. |
| Mask refinement off | Confirmed | Both our runs and the paper Table V have it off. |
| Negative prompt of `"thing"` | Tested as a sanity check; this is the §IV.D traversability protocol, **not** Table V (clarified by paper authors) | N/A for Table V. |
| Threshold-at-0.8 binary scoring (also §IV.D traversability) | Not Table V | N/A. |

## What we have not been able to test

Three remaining hypotheses for the residual ~32-point gap:

1. **Per-frame vs pooled mIoU.** We accumulate one confusion matrix across all 1672 test-split frames and compute mIoU once over the totals. Some papers compute mIoU per-frame and average across frames; on a sparse-class dataset like RELLIS-3D this can swing the headline by 10+ points because frames missing a class don't contribute a 0 to that class's score. The paper does not specify which averaging convention it uses for Table V.

2. **Test-split version.** We use `test.lst` from the 44 KB Image Split File archive on the upstream RELLIS-3D Google Drive, which contains 1672 lines in 2-column `<image_path> <label_path>` format. We cannot tell whether the paper uses this same `test.lst`, a different release of it, or a curated subset.

3. **Some internal OTAS pipeline detail not visible in the public code.** The released `single_inference` API is single-(pos, neg) → binary; the multi-class mIoU pipeline that produced Table V is not in the public release, so we cannot inspect it for subtle differences (averaging convention, prompt encoding, similarity normalisation reset between classes, etc.).

## Ask for the upstream maintainers

To close this gap unambiguously, it would be very helpful to release **the exact evaluation script that produced Table V**. Specifically:

1. The N-way → mIoU pipeline (argmax across per-class similarity maps? threshold per class? something else?).
2. The mIoU averaging convention (per-frame vs pooled).
3. The dataset loader's exact split file and any preprocessing it applies.
4. The `OTAS_*.json` config file used for Table V specifically (the `OTAS_small.json` shipped in `src/model_config/` has `enable_mask_refinement: true` and `shared_feat_resolution: 16`, neither of which matches §VII.A).

Even a minimal `eval_rellis.py` analogous to the released `demo.ipynb` — with the same configuration that produced 48.48 mIoU — would let downstream users verify the published number on their own RELLIS-3D copy. We're happy to fold our `own_RELLIS.py` into the existing scaffolding if that's the right approach, or rewrite it on top of an official `eval_rellis.py` once released.

## Verification

After this PR is applied to a fresh OTAS checkout, the following sequence should reproduce both numbers in the table at the top:

```bash
# 1. Setup
git clone <this-fork>; cd otas
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash download_checkpoints.sh

# 2. Stage RELLIS-3D under /path/to/Rellis-3D (see "Prerequisites" above)

# 3. Default-config run (target: ~16.70 full mIoU on test.lst)
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS.py \
    --data_dir /path/to/Rellis-3D
cat result/Pred/RELLIS_rgb/results.txt | grep -E "full_mIoU|fg_only_mIoU"

# 4. Paper-config run (target: ~15.43 full mIoU)
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS.py \
    --data_dir /path/to/Rellis-3D \
    --paper_config
cat result/Pred/RELLIS_rgb_paper/results.txt | grep -E "full_mIoU|fg_only_mIoU"
```

Numbers should match within ±0.5 mIoU on Blackwell (RTX PRO 6000) with the cu132 torch wheel. Other GPU/CUDA combinations may shift slightly but should still land well below the paper's 48.48 claim.
