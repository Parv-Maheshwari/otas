# Shared evaluation loop for OTAS on the unified 6-dataset scoreboard.
#
# Provides one entry point — `run_eval(dataset, class_names, out_dir, modality)` — that
# every per-dataset script (own_GOD.py, own_BASEPROD.py, …) wraps with the dataset-
# specific class list and DataLoader. The point of pulling this out is so the modality
# handling, OTAS inference, prediction caching, and IoU metric computation are written
# once and identical across all six datasets — matching the convention RADSeg's
# eval.py and OpenRSS's own_*.py use, except deduplicated.
#
# Modality contract:
#   The OpenRSS datasets we re-use yield `(image_f, label, name, th_vis)` where
#   `image_f` is a (4, H, W) float tensor in [0, 1] with channels [R, G, B, T].
#   For OTAS (which is RGB-trained DINOv2 + MaskCLIP) we expose two modalities:
#     - "rgb":     pass channels [R, G, B] through OTAS unchanged.
#     - "thermal": replicate channel T to 3 channels (R = G = B = T) — mirrors how
#                  RADSeg's "thermal" column is generated. DINOv2 is out-of-distribution
#                  on thermal, which is the whole point of the comparison.
#
# Output layout (per dataset × modality):
#   <out_dir>/preds/<name>.png            uint8 grayscale, per-pixel class IDs.
#   <out_dir>/results.txt                 full + fg-only mIoU and per-class IoU.
#   <out_dir>/overlays/<sampled_names>.png   3-up [input | gt | pred] panels for a
#                                            sparse sample (visual sanity only).

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sibling otas_segmentor importable when invoked as `python -m own_eval.own_GOD`
# from the OTAS root, or as `python own_eval/own_GOD.py` directly.
_OWN_EVAL_DIR = str(Path(__file__).resolve().parent)
if _OWN_EVAL_DIR not in sys.path:
    sys.path.insert(0, _OWN_EVAL_DIR)


def _tensor_to_pil_rgb(image_f: torch.Tensor) -> Image.Image:
    # image_f: (4, H, W) float in [0, 1] from the OpenRSS DataLoader.
    # Returns a PIL RGB image of the first 3 (RGB) channels.
    rgb = image_f[:3].numpy()
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(rgb_u8, mode="RGB")


def _tensor_to_pil_thermal_as_rgb(image_f: torch.Tensor) -> Image.Image:
    # Replicate the thermal channel (image_f[3]) to a 3-channel PIL RGB so DINOv2's
    # ImageNet-mean normalization sees plausible-ish R = G = B input. This matches
    # RADSeg's "thermal" treatment exactly — same image stream, different model
    # encoder.
    th = image_f[3].numpy()
    th_u8 = (np.clip(th, 0.0, 1.0) * 255.0).astype(np.uint8)
    th_rgb = np.stack([th_u8, th_u8, th_u8], axis=-1)
    return Image.fromarray(th_rgb, mode="RGB")


def _iou_per_class(conf: np.ndarray) -> np.ndarray:
    # Symmetric IoU = TP / (TP + FP + FN) per class, derived from a (N, N) confusion
    # matrix where rows are ground-truth and columns are predictions. Returns NaN for
    # any class that has zero presence in both gt and prediction (so it doesn't drag
    # the mean down to zero artificially).
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)
    return iou


def _save_palette_overlay(pred: np.ndarray, n_classes: int, out_path: Path):
    # Save a deterministic-palette overlay of `pred` (uint8 class IDs) as RGB PNG.
    # tab20 cycles for n_classes > 20; for our 5–22 class range this is fine.
    from matplotlib import cm
    cmap = cm.get_cmap("tab20", max(n_classes, 20))
    palette = (np.array([cmap(i)[:3] for i in range(max(n_classes, 20))]) * 255).astype(np.uint8)
    overlay = palette[pred]  # (H, W, 3)
    Image.fromarray(overlay).save(out_path)


def _save_3up(rgb_pil: Image.Image, gt: np.ndarray, pred: np.ndarray,
              n_classes: int, out_path: Path):
    # Side-by-side: input RGB | GT palette | Pred palette. Used for a sparse sample
    # of frames so we can eyeball alignment / failure modes without needing all
    # overlays on disk.
    from matplotlib import cm
    cmap = cm.get_cmap("tab20", max(n_classes, 20))
    palette = (np.array([cmap(i)[:3] for i in range(max(n_classes, 20))]) * 255).astype(np.uint8)

    h, w = pred.shape
    rgb_np = np.asarray(rgb_pil.resize((w, h)))
    # Treat any GT id >= n_classes (e.g. 255 ignore) as a black pixel rather than
    # crashing the palette lookup.
    gt_safe = np.where(gt < n_classes, gt, 0).astype(np.uint8)
    gt_paint = palette[gt_safe]
    pred_paint = palette[pred]
    panel = np.concatenate([rgb_np, gt_paint, pred_paint], axis=1)
    Image.fromarray(panel).save(out_path)


def run_eval(
    dataset,
    class_names: List[str],
    out_dir: str,
    modality: str,
    *,
    num_overlay_samples: int = 12,
    ignore_label: int = 255,
    config_overrides: Optional[dict] = None,
):
    # Args:
    #   dataset:      a torch Dataset that yields (image_f (4,H,W), label (H,W),
    #                 name (str), th_vis (3,H,W)). The OpenRSS dataset adapters
    #                 satisfy this contract.
    #   class_names:  list of N strings. class_names[0] MUST be "unknown".
    #   out_dir:      where to write preds/, overlays/, results.txt.
    #   modality:     "rgb" or "thermal" (= thermal-as-RGB replica).
    #   num_overlay_samples: how many frames to emit 3-up overlays for. The full
    #                        preds/ dir always has every frame, but overlays are
    #                        sparse so we don't drown the disk.
    #   ignore_label: GT pixel value to treat as ignore (not counted in IoU). 255
    #                 by default — matches the OpenRSS dataset convention.
    #   config_overrides: optional dict of OTAS config keys to override (e.g.
    #                 {"enable_mask_refinement": True} to turn SAM2 on). Forwarded
    #                 verbatim to OTASEncoder; see otas_segmentor._DEFAULT_CONFIG.
    assert modality in {"rgb", "thermal"}, f"modality must be rgb|thermal, got {modality!r}"

    # Import the encoder here so OTAS only loads once per process (and not at module
    # import time, which would prevent us from setting env vars first).
    from otas_segmentor import OTASEncoder

    out_dir = Path(out_dir)
    preds_dir = out_dir / "preds"
    overlays_dir = out_dir / "overlays"
    preds_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    encoder = OTASEncoder(class_names=class_names, config_overrides=config_overrides)

    # We deliberately keep batch_size=1 because OTAS's language_map operates on PIL
    # images one at a time (DINOv2 forward is autograd-disabled but not batched in
    # OTAS's reference path).
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    n_frames = len(dataset)

    # Pick `num_overlay_samples` evenly-spaced indices for the qualitative panels.
    overlay_indices = set(np.linspace(0, n_frames - 1, num=num_overlay_samples, dtype=int).tolist())

    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    t0 = time.time()

    chosen = (
        _tensor_to_pil_rgb if modality == "rgb" else _tensor_to_pil_thermal_as_rgb
    )
    pbar = tqdm(loader, desc=f"OTAS[{modality}] {out_dir.name}", total=n_frames)
    for idx, batch in enumerate(pbar):
        image_f, label, name, _th_vis = batch
        # DataLoader collates to a leading batch dim of 1 — strip it.
        image_f = image_f.squeeze(0)        # (4, H, W) float
        label_np = label.squeeze(0).numpy()  # (H, W) int64
        # `name` may be a list (DataLoader collates strings) — pull out the scalar.
        if isinstance(name, (list, tuple)):
            name = name[0]
        name = str(name)

        pil = chosen(image_f)
        preds, _probs = encoder.predict(pil)  # uint8 (H, W)

        # Bucket into the conf matrix, excluding ignore-label pixels.
        valid = label_np != ignore_label
        if valid.any():
            gt_valid = label_np[valid]
            pred_valid = preds[valid]
            # Clip just in case — preds are already in [0..n_classes-1] but defensive.
            gt_valid = np.clip(gt_valid, 0, n_classes - 1)
            pred_valid = np.clip(pred_valid, 0, n_classes - 1)
            bin_idx = gt_valid * n_classes + pred_valid
            counts = np.bincount(bin_idx, minlength=n_classes * n_classes)
            conf += counts.reshape(n_classes, n_classes)

        # Cache pred PNG (uint8 grayscale) — argmax IDs, no palette, so any
        # downstream tool can re-paint with its own palette.
        Image.fromarray(preds).save(preds_dir / f"{name}.png")

        if idx in overlay_indices:
            _save_3up(pil, label_np, preds, n_classes, overlays_dir / f"{name}.png")

    elapsed = time.time() - t0
    iou = _iou_per_class(conf)
    full_miou = np.nanmean(iou)
    fg_miou = np.nanmean(iou[1:])  # excludes the 'unknown' class at contig 0

    # Write a single results.txt that's grep-friendly for the scoreboard updater.
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"# OTAS eval — {out_dir.name}\n")
        f.write(f"modality: {modality}\n")
        f.write(f"n_frames: {n_frames}\n")
        f.write(f"n_classes: {n_classes}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"full_mIoU: {full_miou * 100:.4f}\n")
        f.write(f"fg_only_mIoU: {fg_miou * 100:.4f}\n")
        f.write("\nper_class_IoU:\n")
        for name, val in zip(class_names, iou):
            val_str = "nan" if np.isnan(val) else f"{val * 100:.4f}"
            f.write(f"  {name}: {val_str}\n")
        f.write("\nconfusion_matrix_rows_gt_cols_pred:\n")
        for row in conf:
            f.write("  " + " ".join(str(int(c)) for c in row) + "\n")

    print(f"[{out_dir.name}] full mIoU = {full_miou * 100:.2f}%, "
          f"fg-only mIoU = {fg_miou * 100:.2f}% "
          f"({n_frames} frames, {elapsed:.0f}s)")
    return {"full_mIoU": full_miou, "fg_only_mIoU": fg_miou, "iou_per_class": iou}
