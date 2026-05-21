# RELLIS-3D Dataset (Texas A&M off-road autonomous-driving benchmark) loader for OTAS.
#
# Yields `(image_f (4,H,W), label (H,W), name_safe (str), th_vis (3,H,W))` tuples — the
# same shape the OTAS `own_eval/eval_common.run_eval` loop expects. Self-contained so the
# RELLIS-3D replication can be reproduced from a single OTAS checkout, no sibling repos
# required.
#
# RELLIS-3D ships:
#   - RGB:  pylon_camera_node/<seq>/<stem>.jpg            (1920x1200)
#   - Label-id PNG: pylon_camera_node_label_id/<seq>/<stem>.png  (1920x1200, uint8)
#   - NO thermal channel (sensors are LiDAR + Basler RGB + Nerian stereo + VN-300 INS).
#
# The dataset root also contains `train.lst` / `val.lst` / `test.lst`. Each line is two
# whitespace-separated paths relative to the dataset root: `<rel_image_path> <rel_label_path>`.
# (The legacy HRNet loader has a 1-column branch for test mode, but the released test.lst
# is actually 2-col like train/val — confirmed by inspection.)
#
# Class IDs in the RELLIS ontology are sparse: {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17,
# 18, 19, 23, 27, 31, 33, 34}. We remap to contiguous [0..19] under the unified
# unknown-incl convention: raw 0 (void) -> contig 0 (unknown); raw {1, 3, 4, ..., 34} ->
# contig {1, 2, 3, ..., 19} in the order documented by Rellis-3D/ontology.yaml; any other
# raw ID (gap values 2/11/13/14/16/20-22/24-26/28-30/32/35/36) folds to contig 0
# (unknown) via the 256-entry LUT default. This is byte-identical to what
# `RemapRellisUnknownLabel` does in RADSeg/evaluation/2d/custom_datasets.py — same
# mapping, same class order.
#
# Returned tuple matches GOD_dataset's contract so eval_common.run_eval works unchanged:
#   (image_f (4,H,W) float[0,1] R G B 0, label (H,W) int64 in 0..19, name_safe (str), th_vis (3,H,W) float)

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import PIL
from PIL import Image
from matplotlib import cm


# Order matches Rellis-3D/ontology.yaml + RADSeg's RELLIS_UNKNOWN_CLASSES. Singular `tree`
# is intentional — kept verbatim from the source ontology (vs GOD's plural `trees`).
RELLIS_CLASS_NAMES = [
    "unknown", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle",
    "object", "asphalt", "building", "log", "person", "fence", "bush",
    "concrete", "barrier", "puddle", "mud", "rubble",
]
RELLIS_NUM_CLASSES = len(RELLIS_CLASS_NAMES)  # 20 incl unknown

# Raw RELLIS GT id -> contig id under the unknown-incl convention.
# Raw 0 (void) -> 0 (unknown). 19 fg raw ids -> contig 1..19 in ontology-yaml order.
RELLIS_RAW_TO_CONTIGUOUS = {
    0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    12: 10, 15: 11, 17: 12, 18: 13, 19: 14, 23: 15, 27: 16, 31: 17,
    33: 18, 34: 19,
}


def _build_remap_lut() -> np.ndarray:
    # 256-entry LUT, default 0 (unknown). Gap raw IDs (2, 11, 13, 14, 16, 20-22, 24-26,
    # 28-30, 32, 35, 36) silently fold to unknown — matches RADSeg's LUT semantics.
    lut = np.zeros(256, dtype=np.uint8)
    for raw, mapped in RELLIS_RAW_TO_CONTIGUOUS.items():
        lut[raw] = mapped
    return lut


_REMAP_LUT = _build_remap_lut()


# --- HRNet 19-class learning ontology (the canonical 2D RELLIS benchmark setup) ---------
#
# Source: benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/lib/datasets/rellis.py:label_mapping.
# Raw void + dirt collapse to a single contig 0 "unknown" — siglip2/MaskCLIP can't ground
# bare "void"/"dirt" reliably so the HRNet-trained baseline lumps them, and we follow suit
# for OVSS replication. Raw 32 (out-of-ontology) -> water, raw 29/30 -> grass per the
# HRNet remap; these are dataset-internal special cases that don't appear in the official
# 20-name ontology but are non-zero in the released label PNGs (rare). 19 contig classes
# total (1..18 fg + 0 unknown), the same shape published RELLIS-3D 2D HRNet/GSCNN results
# table is reported on.
#
# Class string ordering follows the remapped-ID order so contig 1 = grass, contig 2 = tree,
# … contig 18 = rubble. Same wording as the 20-class variant (singular `tree`).
RELLIS_HRNET_CLASS_NAMES = [
    "unknown",     # 0 — raw 0 (void) ∪ raw 1 (dirt). bare-name grounded as best-effort.
    "grass",       # 1 — raw 3, 29, 30
    "tree",        # 2 — raw 4
    "pole",        # 3 — raw 5
    "water",       # 4 — raw 6, 32
    "sky",         # 5 — raw 7
    "vehicle",     # 6 — raw 8
    "object",      # 7 — raw 9
    "asphalt",     # 8 — raw 10
    "building",    # 9 — raw 12
    "log",         # 10 — raw 15
    "person",      # 11 — raw 17
    "fence",       # 12 — raw 18
    "bush",        # 13 — raw 19
    "concrete",    # 14 — raw 23
    "barrier",     # 15 — raw 27
    "puddle",      # 16 — raw 31
    "mud",         # 17 — raw 33
    "rubble",      # 18 — raw 34
]
RELLIS_HRNET_NUM_CLASSES = len(RELLIS_HRNET_CLASS_NAMES)  # 19

# Verbatim from benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/lib/datasets/rellis.py.
RELLIS_HRNET_RAW_TO_CONTIGUOUS = {
    0: 0, 1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    12: 9, 15: 10, 17: 11, 18: 12, 19: 13, 23: 14, 27: 15,
    29: 1, 30: 1,  # special-case raw ids the HRNet loader treats as grass
    31: 16, 32: 4,  # raw 32 -> water per HRNet
    33: 17, 34: 18,
}


def _build_hrnet_remap_lut() -> np.ndarray:
    # Default 0 (unknown) so any raw id not explicitly listed lands at 0 — including the
    # ontology-gap ids and any future sensor-noise values. Matches HRNet's behaviour where
    # the default-init mapping is to ignore.
    lut = np.zeros(256, dtype=np.uint8)
    for raw, mapped in RELLIS_HRNET_RAW_TO_CONTIGUOUS.items():
        lut[raw] = mapped
    return lut


_HRNET_REMAP_LUT = _build_hrnet_remap_lut()


class RELLIS_dataset(Dataset):
    # Yields (image_tensor (4,H,W) float[0,1], label_tensor (H,W) int64, name_safe (str),
    # th_vis (3,H,W) float[0,1]) so OTAS's eval_common.run_eval can consume it unchanged.
    #
    # `data_dir` should point at the extracted Rellis-3D root (the directory containing the
    # split lists and the 5 sequence subdirs 00000..00004). The split list is read directly
    # from `<data_dir>/<split>.lst` — no separate `split_dir` is needed; RELLIS ships them
    # at the dataset root.
    #
    # `mask_modality` is accepted for API parity with GOD_dataset (so any test loop that
    # branches on it still works) but only "none" makes sense here — RELLIS has no thermal,
    # so zeroing R/G/B vs the always-zero thermal channel is meaningless.
    def __init__(self, data_dir, split="test", split_dir=None, input_h=480, input_w=640,
                 transform=None, cm_type="jet", mask_modality="none", ontology="raw20"):
        super().__init__()
        self.data_dir = data_dir
        # split_dir is accepted for API parity with GOD_dataset but ignored — RELLIS ships
        # the split lists at the dataset root, not a sibling directory.
        self.split_dir = split_dir or data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform or []
        self.cm_type = cm_type
        if mask_modality not in {"none", "rgb_only", "thermal_only"}:
            raise ValueError(
                f"mask_modality must be one of none|rgb_only|thermal_only, got {mask_modality!r}")
        self.mask_modality = mask_modality
        # ontology controls which LUT + class list the loader uses:
        #   "raw20"  — 20 classes (1 unknown + 19 fg), unknown-incl raw RELLIS ontology
        #              (parallel to RADSeg's RellisUnknownInclDataset).
        #   "hrnet19" — 19 classes (1 unknown + 18 fg) under HRNet's `label_mapping`. Used
        #              for replicating the canonical 2D RELLIS benchmark numbers (likely
        #              what the OTAS paper Table V reports on).
        if ontology not in {"raw20", "hrnet19"}:
            raise ValueError(f"ontology must be raw20|hrnet19, got {ontology!r}")
        self.ontology = ontology
        if ontology == "raw20":
            self._lut = _REMAP_LUT
            self.class_names = RELLIS_CLASS_NAMES
        else:
            self._lut = _HRNET_REMAP_LUT
            self.class_names = RELLIS_HRNET_CLASS_NAMES

        # Parse the split file. Each line has two whitespace-separated relative paths.
        split_path = os.path.join(self.split_dir, split + ".lst")
        self.pairs = []  # list of (img_rel, label_rel)
        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # The released test.lst has 2 cols like train/val (the upstream HRNet
                # loader's 1-col test branch is legacy / unused) — fail loudly if that
                # assumption is ever violated by a future RELLIS release.
                assert len(parts) == 2, (
                    f"RELLIS split line must have 2 paths, got {len(parts)}: {line!r}")
                self.pairs.append((parts[0], parts[1]))
        self.n_data = len(self.pairs)

    def __getitem__(self, index):
        img_rel, label_rel = self.pairs[index]
        # name_safe: replace path separators with '_' so eval_common can use it as a single
        # filename component. Strip the .jpg extension. e.g. "00000_pylon_camera_node_frame000000-1581624652_750"
        stem, _ext = os.path.splitext(img_rel)
        name_safe = stem.replace("/", "_")

        # RGB at 1920x1200 -> bilinear resize to (input_h, input_w)
        rgb_path = os.path.join(self.data_dir, img_rel)
        rgb = np.asarray(PIL.Image.open(rgb_path).convert("RGB"))  # (1200, 1920, 3) uint8
        rgb_resized = np.asarray(
            PIL.Image.fromarray(rgb).resize((self.input_w, self.input_h),
                                            resample=PIL.Image.BILINEAR)
        )  # (input_h, input_w, 3) uint8

        # Synthetic zero thermal channel — RELLIS-3D has no thermal sensor. The 4-channel
        # contract is preserved so eval_common doesn't need a separate code path.
        th_resized = np.zeros((self.input_h, self.input_w), dtype=np.uint8)

        # Modality-ablation hook for API parity with GOD_dataset. On RELLIS rgb_only is the
        # natural state; thermal_only would zero out everything and is meaningless here.
        if self.mask_modality == "rgb_only":
            th_resized = np.zeros_like(th_resized)
        elif self.mask_modality == "thermal_only":
            rgb_resized = np.zeros_like(rgb_resized)

        # (H, W, 4) uint8 R G B 0
        image = np.dstack([rgb_resized, th_resized])

        # Label from pylon_camera_node_label_id (same resolution as RGB).
        # Remap via LUT to contig [0..19]; nearest-neighbor downsample to (input_h, input_w).
        label_path = os.path.join(self.data_dir, label_rel)
        raw_label = np.asarray(PIL.Image.open(label_path))  # (1200, 1920) uint8
        if raw_label.ndim == 3:
            raw_label = raw_label[:, :, 0]
        remapped = self._lut[raw_label]
        label = np.asarray(
            PIL.Image.fromarray(remapped).resize((self.input_w, self.input_h),
                                                 resample=PIL.Image.NEAREST),
            dtype=np.int64,
        )

        for func in self.transform:
            image, label = func(image, label)

        # th_vis: GOD uses a jet colormap on the thermal channel for the qualitative panels.
        # Here the channel is all-zero so th_vis is a constant blue rectangle — harmless,
        # but it keeps the 4-tuple shape that eval_common.run_eval expects.
        cmap = cm.get_cmap(self.cm_type)
        th_vis = (cmap(image[:, :, 3])[:, :, :3] * 255).astype(np.uint8)
        th_vis = np.asarray(
            PIL.Image.fromarray(th_vis).resize((self.input_w, self.input_h)),
            dtype=np.float32,
        ).transpose((2, 0, 1)) / 255

        image_f = np.asarray(
            PIL.Image.fromarray(image).resize((self.input_w, self.input_h)),
            dtype=np.float32,
        ).transpose((2, 0, 1)) / 255

        return torch.tensor(image_f), torch.tensor(label), name_safe, th_vis

    def __len__(self):
        return self.n_data
