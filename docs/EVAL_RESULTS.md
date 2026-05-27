# OTAS evaluation results

OTAS = "Open-vocabulary Token Alignment for Outdoor Segmentation" (Schwaiger et al., arXiv:2507.08851, ICRA 2026). One-page scoreboard of every dataset eval we ran for the side-by-side against RADSeg and OpenRSS, mirroring the shape of [`../../RADSeg/EVAL_RESULTS.md`](../../RADSeg/EVAL_RESULTS.md) and [`../../OpenRSS/docs/EVAL_RESULTS.md`](../../OpenRSS/docs/EVAL_RESULTS.md) so the three are directly comparable.

All numbers are **pure zero-shot** with frozen DINOv2 ViT-S/14 (patch tokens) + frozen MaskCLIP ViT-B/16 (text + patch features) + KMeans token clustering. SAM2 refinement is **off** (parity with the other two systems, which have no post-processing). Predictions and per-frame overlays are cached on disk per run.

**Locked-in run configuration** (every RGB row in the scoreboard uses these — change any of them and the numbers change). The four feature-extractor knobs match the OTAS paper's RELLIS-3D supplementary §VII.A protocol verbatim, applied uniformly across all datasets so the OTAS column is internally consistent.

| Knob | Value | Where it's set |
|---|---|---|
| Vision backbone | DINOv2 ViT-S/14 with 4 register tokens | `src/foundation_models/dinov2_checkpoints/dinov2_vits14_reg4_pretrain.pth` |
| Language head | MaskCLIP ViT-B/16 (OpenAI weights) | `src/foundation_models/clip_checkpoints/ViT-B-16.pt` |
| Cluster model | KMeans, **24** clusters, **24** PCA components | `n_clusters=24`, `n_components=24` (paper §VII.A) |
| DINOv2 input size | **224×224** (native 16×16 patch grid at 14-px patches) | `dinov2_input_size: 224` (paper §VII.A) |
| Shared feature resolution | **64×64** patch grid; for DINOv2 the 16×16 native grid is bilinear-interp'd up to 64×64 | `shared_feat_resolution: 64` (paper §VII.A) |
| SAM2 mask refinement | **off** | `enable_mask_refinement: false` in `own_eval/otas_segmentor.py:_DEFAULT_CONFIG` |
| CLIP prompt format | **bare class names — no template, no negative prompts** | `own_eval/otas_segmentor.py:OTASEncoder.__init__` |
| Input resolution | **native per dataset** (preserves aspect ratio): GOD 1080×1440 (rgb pylon) / 512×640 (thermal LWIR), BASEPROD 720×1280, CART 600×960, MFNet 480×640, RELLIS-3D 1200×1920. GOD splits the input shape by modality because pylon (RGB) and LWIR (thermal+labels) are not pixel-aligned — see [§GOD label-camera + MFNet prompt-set reruns](#god-label-camera--mfnet-prompt-set-reruns-2026-05-22--landed). Differs from §VII.A's literal "1024×1024" which was specifically about getting AM-RADIO/DINOv3's 16-px patches to land on d=64 — irrelevant for DINOv2 which always sees 224×224 internally | argparse defaults in `own_eval/own_*.py` |
| Modality | **RGB only** (thermal-as-RGB runs are not part of this scoreboard) | `--modalities rgb` |
| Batch size | 1 | OTAS's reference path is per-image |
| GPU | NVIDIA RTX PRO 6000 Blackwell (sm_100) | from `nvidia-smi` |
| Torch / CUDA | torch 2.12.0 + cu132 | `pip install torch --index-url …/whl/cu132` |
| LD_LIBRARY_PATH | **unset at eval time** | `env -u LD_LIBRARY_PATH …` |
| Python venv | `/home/ubuntu/code/OTAS/.venv` | per-repo, never shared |

## Prompt protocol — bare class names, no template

This is the explicit user-spec for OTAS evaluation. Each dataset's class list is passed to MaskCLIP's text encoder as raw strings — `["unknown", "grass", "trees", …]` — with no `"a photo of …"` wrapper and no negative prompts. The argmax across N positive prompts is the entire scoring rule.

The protocol differs from RADSeg (20-variant ImageNet template) and OpenRSS (`"a pohot of {c}"` typo-preserved training prompt) by design — the goal is to compare each system in the prompt format it was designed for, not to make every system use the same prompt.

**One consequence**: the bare string `"unknown"` does not ground in MaskCLIP-B/16. Per-class IoU on `unknown` is essentially zero on every dataset (see [§The `unknown` class never grounds](#the-unknown-class-never-grounds)). RADSeg's template expansion and siglip2's larger embedding space partially fix the same problem; OTAS's bare-string protocol does not.

## GOD label-camera + MFNet prompt-set reruns (2026-05-22) — landed

Two cross-system dataloader bugs were fixed and the affected rows re-run. Both reruns landed 2026-05-22; numbers below are the new ones.

**1. GOD label camera.** GOD's pylon (RGB, 1080×1440) and LWIR (thermal+labels, 512×640) cameras are **not pixel-aligned**, but OTAS's `own_GOD.py` used to instantiate the OpenRSS [`GOD_dataset`](../../OpenRSS/util/GOD_dataset.py) with `mask_modality="none"` for both passes — which under the OpenRSS adapter always loaded LWIR labels. For the RGB pass at 1080×1440 input, those labels were NEAREST-upsampled from LWIR-frame to pylon-frame and used to score pylon-frame DINOv2 predictions — cross-camera-frame supervision against pixels that came from a different physical viewpoint. Fix: [`own_eval/own_GOD.py`](../own_eval/own_GOD.py) now passes `mask_modality="rgb_only"` for the RGB pass (pylon labels at 1080×1440) and `mask_modality="thermal_only"` for the thermal pass (LWIR labels at native 512×640 — no more upsampling). The OpenRSS adapter change that makes this work is [`OpenRSS/util/GOD_dataset.py`](../../OpenRSS/util/GOD_dataset.py).

GOD RGB jumped **+3.72 full mIoU** (11.75 → 15.47) under the corrected pylon-frame labels, with `barrier` (0.00 → 21.07), `fence` (3.37 → 13.87), `dirt` (24.46 → 28.63), and `sky` (44.67 → 52.09) all gaining. Five raw classes have **no GT pixels in the pylon-frame label set** (`asphalt`, `concrete`, `object`, `puddle`, `rubble`) — they're at 0.00 / nan in the per-class table below and excluded from the mean.

**2. MFNet prompt set.** [own_eval/own_MF.py](../own_eval/own_MF.py)'s `MF_CLASS_NAMES` used to read `["unknown", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]` — lowercase + underscores. The OpenRSS LoRA was trained against the Capitalized + space form (`["unknown", "Car", "Person", "Bike", "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump"]`); OTAS now matches so all three systems score against identical strings. RADSeg's `configs/cls_mfnet_unknown.txt` was updated in the same audit. The prompt-form switch barely moved OTAS: full mIoU 8.32 → 8.34, fg-only 9.29 → 9.30 — bare-string MaskCLIP-B/16 doesn't distinguish `"car"` from `"Car"` enough to shift the headline.

CART, BASEPROD, BASEPROD condensed, Handheld CART, and RELLIS-3D rows are **unaffected by these fixes** — those datasets ship a single camera frame with a single label set, and their MFNet-style prompt sets did not change.

## Scoreboard

All rows use the **unified ignore-class convention**: every dataset's raw ignore class (Void / void / unlabelled / Unknown / Background) collapses to a single contig id 0 named `unknown` that gets its own MaskCLIP prompt. Two mIoU columns: **full** averages across all classes (including unknown); **fg-only** averages across foreground classes only.

**Scope:** RGB only. The OTAS column reports numbers under the paper §VII.A config (DINOv2 ViT-S/14 native 16×16 patch grid → bilinear-interpolated up to the d=64 shared resolution, k=24, Cr=24, dinov2_input_size=224, no mask refinement, all class names from the released ontology as positive prompts, no negative prompt). Input image resolution is each dataset's native (H, W). Thermal-as-RGB runs are not part of this scoreboard.

| Dataset | n frames | **mIoU (full)** | **mIoU (fg-only)** | elapsed (s) | notes |
|---|--:|--:|--:|--:|---|
| Great Outdoors (22-cls, native pylon) | 1199 | **15.47** | **16.25** | 323 | native 1080×1440 + pylon-frame labels (2026-05-22 GOD label-camera fix). 5 raw classes (`asphalt`, `concrete`, `object`, `puddle`, `rubble`) have no GT pixels in the pylon label set and are excluded from the mean. |
| Great Outdoors (22-cls, midres 480×640)‡ | 1199 | **15.66** | **16.44** | 194 | midres 480×640 + pylon-frame labels. Re-run to provide the apples-to-apples row alongside RADSeg + OpenRSS in [baseline.md](../../baseline.md) — OpenRSS-on-GOD only runs at midres (see ‡ below). |
| BASEPROD (8-cls) | 950 | **9.16** | **10.04** | 364 | native 720×1280 |
| BASEPROD condensed (5-cls) | 950 | **20.08** | **23.98** | 269 | native 720×1280 |
| CART (11-cls) | 2282 | **29.59** | **32.55** | 443 | native 600×960 |
| **Handheld CART** (11-cls) | 493 | **31.98** | **35.17** | 100 | native 600×960 (493-frame subset of CART) |
| MFNet (9-cls) | 393 | 8.34 | 9.30 | 39 | native 480×640. Capitalized + space prompt set (2026-05-22 cross-system audit). |
| **RELLIS-3D (20-cls, test)** | 1672 | **15.66** | **16.48** | 901 | native 1200×1920. Paper Table V claims 48.48 mIoU; our default-config run (480×640, d=32, Cr=12) lands at 16.70; the §VII.A-literal 1024×1024 lands at 15.43; this native-resolution run lands at 15.66. None of them come within 30 mIoU of the paper's claim. See [§RELLIS-3D paper-replication note](#rellis-3d--paper-replication-note). |

Total wall-clock: **~41 min** across the 7 RGB runs on Blackwell.

**GOD thermal-as-RGB (separately tracked, not in the RGB scoreboard above).** For completeness — the [stale-numbers section](#god-label-camera--mfnet-prompt-set-reruns-2026-05-22--landed) above describes why this row is now scored at LWIR-native 512×640 against LWIR labels (no more 512→1080 NEAREST upsampling). Numbers: **full mIoU 7.94, fg-only 8.31**, 165 s elapsed. Thermal does worse than RGB on every dominant class (`sky` 21.90 vs 52.09, `grass` 32.86 vs 62.06, `trees` 75.90 vs 82.09) — same pattern as RADSeg and OpenRSS thermal on GOD. Midres 480×640 thermal rerun: **full mIoU 8.03, fg-only 8.41**, 139 s — essentially flat from native.

‡ **GOD midres rows** are 480×640 re-runs included for cross-system parity with OpenRSS, which cannot be evaluated at pylon-native (1080×1440) due to a SAM `image_size=640` constraint that silently crops the input to its top-left 640×640 patch (see [OpenRSS/docs/EVAL_RESULTS.md#why-midres-on-god](../../OpenRSS/docs/EVAL_RESULTS.md#why-midres-on-god)). At midres, OTAS RGB lands at 15.66/16.44 vs OTAS native pylon 15.47/16.25 — DINOv2's internal 224×224 resize means resolution barely affects the OTAS score; we report both rows for completeness. Same midres recipe used by RADSeg and OpenRSS.

## Headlines

### 1. RGB — runner-up to RADSeg, dominant over OpenRSS

OTAS lands at 60–82% of RADSeg's RGB fg-only across the off-road / aerial datasets:

| Dataset | RADSeg RGB | **OTAS RGB** | OpenRSS RGB-only | OTAS / RADSeg |
|---|---:|---:|---:|---:|
| Great Outdoors | 33.32 | 16.25 | 4.54 | 0.49 |
| BASEPROD condensed | 34.07 | 23.98 | 6.01 | 0.70 |
| CART | 40.23 | 32.55 | 8.32 | 0.81 |
| Handheld CART | 47.54 | 35.17 | 10.61 | 0.74 |

The Great Outdoors gap (49%) is wider than the off-road gap (70–81% band) because Great Outdoors is the most natural-scene dataset and siglip2's natural-image prior is strongest there. On off-road / aerial scenes, DINOv2-S's patch clustering covers most of the distance to RADIO v3-b despite being ~14× smaller (21M vs 300M params).

### 2. MFNet — only here does training matter

The single dataset where OTAS finishes last is MFNet (the dataset OpenRSS was specifically trained on). OTAS fg-only RGB is well behind OpenRSS's 21.80 and RADSeg's 21.54. Per-class breakdown shows the failure is concentrated in `person` (OTAS: ~1; RADSeg: 59; OpenRSS: 57): DINOv2-S struggles to cluster human silhouettes at MFNet's relatively small object scale, and MaskCLIP-B/16's `"person"` prompt is too weak to recover.

### 3. The `unknown` class never grounds

OTAS's per-class IoU on the literal string `"unknown"` (no template, no negative prompt) is essentially 0 on every dataset:

| Dataset | OTAS RGB unknown IoU | RADSeg reference (RGB) |
|---|---:|---:|
| Great Outdoors | 0.00 | 4.49 |
| BASEPROD | 0.07 | 68.31 |
| BASEPROD condensed | 0.05 | 68.03 |
| CART | 0.00 | 0.29 |
| Handheld CART | 0.00 | low |
| MFNet | 0.62 | 3.08 |

The bare string `"unknown"` lands in MaskCLIP-B/16's embedding space close to *everything* — it's a meta-word, not a visual concept. RADSeg's prompt-template expansion (`"a photo of unknown."`, `"a sculpture of unknown."`, …) and siglip2's bigger embedding space partially fix this; OTAS's bare-string protocol does not.

The full / fg-only delta for OTAS is therefore always negative (full < fg-only), unlike the other systems where condensed-BASEPROD shows the inverse (RADSeg full 40.86 / fg-only 34.07; unknown helps by +6.79). This is purely a function of the user-specified prompt protocol.

## Per-class IoU — full tables

### Great Outdoors RGB (22-cls)

Pylon-frame labels (post-2026-05-22 GOD label-camera fix). `asphalt`, `concrete`, `object`, `puddle`, `rubble` have **no GT pixels in the pylon label set** — IoU is 0 / 0 (reported as 0.00 or nan depending on the eval adapter's numerical path) and they are dropped from the mean.

| class | IoU (RGB) | IoU (thermal-as-RGB) |
|---|---:|---:|
| unknown | 0.00 | 0.04 |
| dirt | 28.63 | 13.88 |
| grass | 62.06 | 32.86 |
| trees | **82.09** | **75.90** |
| pole | 1.64 | 1.59 |
| water | 12.02 | 1.06 |
| sky | 52.09 | 21.90 |
| vehicle | 4.85 | 0.29 |
| object | 0.00¹ | 0.02 |
| asphalt | 0.00¹ | 0.01 |
| building | 0.00 | 3.37 |
| log | 2.84 | 0.90 |
| person | 2.95 | 0.00 |
| fence | 13.87 | 0.24 |
| bush | 2.78 | 1.57 |
| concrete | nan¹ | 0.00¹ |
| barrier | 21.07 | 0.00 |
| puddle | 0.00¹ | 0.00¹ |
| mud | 4.01 | 0.00 |
| rubble | 0.00¹ | 0.01 |
| mulch | 8.99 | 1.00 |
| gravel | 25.08 | 19.98 |
| **full mIoU** | **15.47** | **7.94** |
| **fg-only** | **16.25** | **8.31** |

¹ Class has no GT pixels under the relevant label set (pylon for RGB; LWIR for thermal) — excluded from the mean by the OTAS adapter.

RGB beats thermal on every dominant class; thermal-as-RGB on GOD is the same negative-control pattern as the OpenRSS / RADSeg thermal-only rows.

### BASEPROD RGB (8-cls)
| class | OTAS RGB IoU |
|---|---:|
| unknown | 3.05 |
| Compact | 1.89 |
| Grass | 12.87 |
| Bedrock | 3.94 |
| Sandy | 3.10 |
| Pebble | **31.71** |
| Rock | 1.01 |
| Vegetation | 15.74 |
| **full mIoU** | **9.16** |
| **fg-only** | **10.04** |

OTAS wins `Pebble` outright (31.71 vs all other systems ≤ 0.001) — DINOv2 patch clustering picks out the textured pebble surface that siglip2's word-level grounding can't isolate.

### BASEPROD condensed RGB (5-cls)
| class | OTAS RGB IoU |
|---|---:|
| unknown | 4.51 |
| Ground | 56.65 |
| Grass | **16.17** |
| Rock | 7.11 |
| Vegetation | 15.98 |
| **full mIoU** | **20.08** |
| **fg-only** | **23.98** |

OTAS beats RADSeg on `Grass` (16.17 vs 0.50) — RADSeg's condensed Grass class collapses onto Ground in its head; OTAS keeps the two separable through DINOv2's texture clusters.

### CART RGB (11-cls)

> **Note:** Class names below are the AnyThermal-derived `rocky terrain` / `developed structures` because that was the prompt set at run-time. Canonical CART class names in code (via OpenRSS's `util/CART_dataset.py`) were switched to Caltech-verbatim `boulders / rocky terrain` / `human-made structures` on 2026-05-22; re-run `own_eval/own_CART.py` to refresh.

| class | OTAS RGB IoU |
|---|---:|
| unknown | 0.00 |
| bare ground | 30.18 |
| rocky terrain | 38.37 |
| developed structures | 20.09 |
| road | 10.22 |
| shrubs | 21.40 |
| trees | 47.03 |
| sky | 54.36 |
| water | **82.74** |
| vehicles | 2.50 |
| person | **18.59** |
| **full mIoU** | **29.59** |
| **fg-only** | **32.55** |

OTAS beats RADSeg on `person` (18.59 vs 12.08) — DINOv2's patch cluster picks up the human silhouette at aerial scale better than siglip2's word grounding does.

### Handheld CART RGB (11-cls)

> **Note:** Class names below are AnyThermal-derived (`rocky terrain` / `developed structures`) — same caveat as the full CART table above; canonical names have been switched to Caltech-verbatim in code on 2026-05-22.

| class | OTAS RGB IoU |
|---|---:|
| unknown | 0.00 |
| bare ground | 33.91 |
| rocky terrain | 40.06 |
| developed structures | 3.85 |
| road | 4.00 |
| shrubs | 34.51 |
| trees | **61.15** |
| sky | 57.66 |
| water | **76.65** |
| vehicles | 15.51 |
| person | 24.44 |
| **full mIoU** | **31.98** |
| **fg-only** | **35.17** |

Handheld subset gains on `trees` (+14), `shrubs` (+13), `person` (+6) over the full CART set — handheld imagery sits closer to DINOv2's training distribution than aerial.

### MFNet — the failure case

Capitalized + space prompt set (post-2026-05-22 cross-system audit). Thermal-as-RGB included for reference (MFNet is the trained-on dataset for OpenRSS, so it's the only place a thermal-as-RGB comparison is interesting for OTAS).

| class | OTAS RGB | OTAS thermal-as-RGB |
|---|---:|---:|
| unknown | 0.62 | 0.86 |
| Car | **47.44** | 42.07 |
| **Person** | **0.83** | **4.26** |
| Bike | 23.34 | 20.69 |
| Curve | 0.63 | 0.24 |
| Car Stop | 0.10 | 0.08 |
| Guardrail | 0.11 | 0.17 |
| Color Cone | 1.91 | 3.85 |
| Bump | 0.03 | 0.06 |
| **full mIoU** | **8.34** | **8.03** |
| **fg-only** | **9.30** | **8.93** |

Capitalized form barely moves the headline vs the pre-audit lowercase numbers (8.32 → 8.34 full, 9.29 → 9.30 fg-only). `Car` shifts −1.3 (48.76 → 47.44), `Bike` shifts +1.1 (22.28 → 23.34) — within the noise floor. `Person`, `Color Cone`, `Car Stop`, `Guardrail`, `Bump` all remain near zero — small objects + dataset-specific terminology that bare-string MaskCLIP-B/16 can't ground at MFNet's image scale regardless of capitalization.

Thermal-as-RGB lifts `Person` from 0.83 → 4.26 IoU — the only place thermal demonstrably helps OTAS on MFNet — and modestly raises `Color Cone` (1.91 → 3.85). Both classes have thermally distinct silhouettes (warm body, warm reflective cone) that DINOv2 patch clustering picks out better than from RGB alone. Cars and bikes drop slightly under thermal (47.44 → 42.07; 23.34 → 20.69) — RGB texture cues that are absent from the R=G=B-replicated thermal channel. Net effect: thermal is roughly flat (full mIoU 8.34 vs 8.03), but the per-class shifts go in the expected directions.

### RELLIS-3D RGB (20-cls)
| class | OTAS RGB IoU |
|---|---:|
| unknown | 0.00 |
| dirt | 0.00 |
| grass | **59.23** |
| tree | **56.89** |
| pole | 2.17 |
| water | 9.33 |
| sky | **85.98** |
| vehicle | 1.30 |
| object | 0.00 |
| asphalt | 4.14 |
| building | 5.95 |
| log | 0.00 |
| person | 3.43 |
| fence | 7.95 |
| bush | 11.22 |
| concrete | 6.09 |
| barrier | 11.66 |
| puddle | **35.98** |
| mud | 9.71 |
| rubble | 2.18 |
| **full mIoU** | **15.66** |
| **fg-only** | **16.48** |

Only `sky 85.98`, `grass 59.23`, `tree 56.89`, and `puddle 35.98` clear 30 mIoU; 13 of 20 classes are below 10. Bare-string MaskCLIP-B/16 can't ground `dirt`, `log`, `object`, `pole`, `vehicle`, `mud` against off-road imagery, and those depress the mean. See [§RELLIS-3D — paper-replication note](#rellis-3d--paper-replication-note) for why this number is ~32 mIoU below the OTAS paper Table V claim of 48.48.

### RELLIS-3D — paper-replication note

The OTAS paper's **Table V** reports `OTAS w. DINOv2 ViT-S/14 = 48.48 mIoU` on RELLIS-3D under "raw class labels as prompts and no mask refinement". After [PR #2](https://github.com/SimonSchwaiger/otas/pull/2), Simon Schwaiger (paper author) clarified that Table V was scored against **a 6-class terrain subset** with a `"thing"` negative prompt and `threshold=0.8`, **not** the 20-class argmax we initially assumed from §VII.A. Verbatim from the PR review comment:

```python
class_prompts = {1: "dirt", 6: "water", 10: "asphalt",
                 19: "bush", 33: "mud", 34: "rubble"}   # raw RELLIS IDs
neg_prompts = ["thing"]
threshold_value = 0.8
```

He also recommended switching to the **first commit (`6aec2d4`)** of the repo as the exact code used for Table V. We did both — added [`own_eval/own_RELLIS_paper.py`](../own_eval/own_RELLIS_paper.py) implementing the literal protocol and ran it against three independent code/config combinations.

| Run | Code | Config | Protocol | mIoU |
|---|---|---|---|--:|
| 1 | repo default at investigation start | d=32, Cr=12, 480×640 | 20 bare class names, argmax | 16.70 |
| 2 | current `main` | §VII.A (d=64, Cr=24, 1024×1024) | 20 bare class names, argmax | 15.43 |
| 3 | current `main`, `own_RELLIS.py` | §VII.A, native 1200×1920 | 20 bare class names, argmax | **15.66** (fg-only 16.48) |
| 4 | current `main`, `own_RELLIS_paper.py` | §VII.A, native | **Simon's protocol**: 6cls, `neg=["thing"]`, `t=0.8` | **6.70** |
| 5 | **first commit (`6aec2d4`)** | first-commit defaults (d=32, Cr=48, dinov2 input 518) | Simon's protocol | **6.64** |
| 6 | **first commit (`6aec2d4`)** | §VII.A overrides (d=64, Cr=24, dinov2 input 224) | Simon's protocol | **6.63** |
| — | — | — | **Paper Table V claim** | **48.48** |

Rows 4–6 are the three independent attempts at Simon's literal protocol and they land within **0.07 mIoU** of each other across two code versions and two config presets — config is not the load-bearing variable, and the first-commit code does not close the gap (`semantic_mask.similarity` math is byte-identical to current `main`; we diffed `src/model.py` to confirm). Applying Simon's literal protocol *widens* the gap from ~32 mIoU (under the 20-class assumption) to **~42 mIoU**. Two structural reasons drive the new floor:

- **Sparse classes get tanked by FPs.** `dirt` appears in only 13 of 1672 test frames (9,690 GT pixels total); `water` in 19 frames; `rubble` in 145 frames. The threshold@0.8 binary pred fires on hundreds of millions of pixels for each of those classes (765M FP for dirt, 364M for rubble) across the 1500+ frames where the class is *absent*. Per-class IoU is `TP / (TP + FP + FN)`; with TPs in the thousands and FPs in the hundreds of millions, IoU is ~0 for every sparse class. The headline is carried entirely by `bush` at 35.14% — the only class that's actually present in nearly every frame.
- **Min-max normalisation amplifies noise on frames where the class is absent.** `semantic_mask.similarity` applies `(sim - sim_min) / clamp(sim_max - sim_min, min=0.05)` *per image*. On a frame containing no `dirt` pixels, the cosine-similarity range is small but non-zero noise; min-max stretches that noise to fill [0,1], so threshold@0.8 fires on the noisiest 20% of pixels. Across 1500+ no-class frames this adds up to the giant FP counts above.

Per-class breakdown under the paper protocol:

| class | raw id | n frames present | total GT px | OTAS IoU |
|---|--:|--:|--:|--:|
| dirt | 1 | 13 | 9,690 | 0.00 |
| water | 6 | 19 | 959,662 | 0.35 |
| asphalt | 10 | 503 | 3,850,438 | 0.66 |
| bush | 19 | 1658 | 662,926,185 | **35.14** |
| mud | 33 | 574 | 29,818,401 | 3.59 |
| rubble | 34 | 145 | 1,907,260 | 0.47 |
| **mIoU(6cls)** | | | | **6.70** |

After ruling out the config and commit-version dimensions (rows 5–6 above), the one open structural hypothesis is the **mIoU averaging convention**: pooled across all frames per class (what we do) vs per-frame averaged only over frames where the class is present in GT. The per-class FP counts in the table above (765M FPs for `dirt` across 1659 absent-class frames) directly tank pooled IoU on sparse classes; per-frame averaging that excludes absent-class frames would drop those FPs out. We have not run this metric yet; it's a one-knob change in `own_RELLIS_paper.py`.

The PR thread also confirmed the supplementary material's current preprint "doesn't include [the 6-class mapping] yet" and will be updated after ICRA. See [RELLIS_REPLICATION.md](RELLIS_REPLICATION.md) for the full PR-ready reply with all six runs, per-class numbers, ruled-out hypothesis table, reproduction commands, and asks for the upstream maintainers.

Per-class numbers under the 20-class argmax run (separate evaluation, not Simon's Table V protocol) are in the [RELLIS-3D RGB (20-cls) per-class table above](#rellis-3d-rgb-20-cls): only `sky 85.98`, `grass 59.23`, `tree 56.89`, and `puddle 35.98` clear 30 mIoU; 13 of 20 classes are below 10. Bare-string MaskCLIP-B/16 can't ground `dirt`, `log`, `object`, `pole`, `vehicle`, `mud` against off-road imagery, and those depress the 20-class mean too.

## Reproducing

Single command, ~41 min on Blackwell across the 6 paired datasets + RELLIS-3D:

```bash
cd /home/ubuntu/code/OTAS
bash own_eval/run_all.sh
```

Or one dataset at a time:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python -m own_eval.own_GOD --out_root result/Pred/GOD_midres   # midres 480×640 (default; required for cross-system parity with OpenRSS — see ‡ in scoreboard)
# To re-run at OTAS-native pylon resolution (RGB at 1080×1440, thermal at LWIR-native 512×640):
#   env -u LD_LIBRARY_PATH .venv/bin/python -m own_eval.own_GOD --input_h "" --input_w "" --out_root result/Pred
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_BASEPROD.py
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_BASEPROD_condensed.py
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_CART.py
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_CART.py --handheld
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_MF.py
env -u LD_LIBRARY_PATH .venv/bin/python own_eval/own_RELLIS.py
```

The `env -u LD_LIBRARY_PATH` prefix forces the cu132 torch wheel's bundled cuBLAS to win over the system `/usr/local/cuda-13.2` lib — without it, MaskCLIP's bigger forward passes fail with `cublasLtGetVersion` symbol errors. See [reference_blackwell_torch_wheel](/home/ubuntu/.claude/projects/-home-ubuntu-code/memory/reference_blackwell_torch_wheel.md).

## Adapter design

The OTAS public API (`single_inference.similarity_single` in `src/inference.py`) takes positive + negative prompt lists and collapses them to a single `(H, W)` similarity map via `mean(pos_sims) - mean(neg_sims)`. That's binary-flavoured scoring; the cross-system scoreboard needs N-way per-pixel argmax across 5–22 classes.

The adapter at [`own_eval/otas_segmentor.py`](../own_eval/otas_segmentor.py) bypasses the aggregation:

1. **Run OTAS's `language_map` once per image** → pooled `(shared_feat_resolution, shared_feat_resolution, 512)` embedding map (DINOv2 patch features → KMeans clustering → CLIP-pooled per-cluster embedding).
2. **Pre-encode all class names as bare strings** at init time using `featurizer.clip_encode_text` (no template wrapper). Each class yields a `(512,)` CLIP text embedding.
3. **Per-class cosine similarity** via OTAS's own `clip_similarity` einsum (`"chw,c->hw"` after L2 normalising both sides). Loop is cheap relative to the DINOv2 forward.
4. **Stack to `(N, H_lr, W_lr)`, bilinear-upsample to `(N, H, W)`** (the same op OTAS's `similarity_single` already uses for its single-map case).
5. **Argmax over the class axis** → `(H, W)` uint8 prediction tensor.

No softmax across classes (argmax of raw cosine similarities is the entire scoring rule). No negative prompts. No prompt templates. Per-class similarity tensors are also returned (`(H, W, N)` float32) for downstream introspection / saved heatmaps, but the scoreboard only consumes the argmax preds.

The dataset adapters at [`own_eval/own_GOD.py`](../own_eval/own_GOD.py), [`own_eval/own_BASEPROD.py`](../own_eval/own_BASEPROD.py), [`own_eval/own_BASEPROD_condensed.py`](../own_eval/own_BASEPROD_condensed.py), [`own_eval/own_CART.py`](../own_eval/own_CART.py), and [`own_eval/own_MF.py`](../own_eval/own_MF.py) reuse OpenRSS's dataset loaders unchanged (`/home/ubuntu/code/OpenRSS/util/*_dataset.py`) so the per-dataset class lists, LUTs, and frame counts are identical to OpenRSS's. Modality handling lives in [`own_eval/eval_common.py:_tensor_to_pil_rgb`](../own_eval/eval_common.py) (RGB channels → PIL) and [`_tensor_to_pil_thermal_as_rgb`](../own_eval/eval_common.py) (thermal channel → R=G=B PIL).

## Files

- [`/home/ubuntu/code/OTAS/own_eval/otas_segmentor.py`](../own_eval/otas_segmentor.py) — `OTASEncoder` class; the multi-class adapter
- [`/home/ubuntu/code/OTAS/own_eval/eval_common.py`](../own_eval/eval_common.py) — modality slicing, IoU bookkeeping, result-file emission
- [`/home/ubuntu/code/OTAS/own_eval/own_GOD.py`](../own_eval/own_GOD.py) — Great Outdoors driver
- [`/home/ubuntu/code/OTAS/own_eval/own_BASEPROD.py`](../own_eval/own_BASEPROD.py) — BASEPROD (8-cls) driver
- [`/home/ubuntu/code/OTAS/own_eval/own_BASEPROD_condensed.py`](../own_eval/own_BASEPROD_condensed.py) — BASEPROD (5-cls condensed) driver
- [`/home/ubuntu/code/OTAS/own_eval/own_CART.py`](../own_eval/own_CART.py) — CART (full + `--handheld`) driver
- [`/home/ubuntu/code/OTAS/own_eval/own_MF.py`](../own_eval/own_MF.py) — MFNet driver
- [`/home/ubuntu/code/OTAS/own_eval/run_all.sh`](../own_eval/run_all.sh) — sequential runner for all 6 dataset × 2 modality combinations
- [`/home/ubuntu/code/OTAS/result/Pred/<dataset>_<modality>/`](../result/Pred/) — per-run output directory (preds/, overlays/, results.txt)
