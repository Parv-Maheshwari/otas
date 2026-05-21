# OTAS open-vocabulary evaluation on RELLIS-3D (Texas A&M off-road autonomous-driving dataset).
#
# Mirrors own_GOD.py one-for-one, with two differences:
#   1. RGB-only — RELLIS-3D ships no thermal modality (sensors: LiDAR + Basler RGB + Nerian
#      stereo + VN-300 INS). The 4-channel image tensor still gets emitted by RELLIS_dataset
#      for API parity, but channel 3 is all-zero and we never run modality="thermal" here.
#   2. SAM ablation — accepts --enable_mask_refinement to toggle OTAS's SAM2 mask refinement
#      head on top of the bare DINOv2 + MaskCLIP token alignment. Default off (matches the
#      paper's Table V row "OTAS w. DINOv2 ViT-S/14, raw class labels, no mask refinement").
#
# Output dir convention:
#   result/Pred/RELLIS_rgb/      <- SAM off (paper Table V replication target = 48.48 mIoU)
#   result/Pred/RELLIS_rgb_sam/  <- SAM on  (new ablation, not in the paper)
#
# Two invocations are needed to produce both numbers (a single process can't easily flip
# the SAM toggle since OTAS's config is locked at OTASEncoder construction time).

import argparse
import sys
from pathlib import Path

# The RELLIS_dataset.py is vendored next to this script (own_eval/rellis_dataset.py) so
# the RELLIS-3D replication is fully self-contained inside an OTAS checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval_common, otas_segmentor, rellis_dataset

from rellis_dataset import (
    RELLIS_dataset, RELLIS_CLASS_NAMES, RELLIS_HRNET_CLASS_NAMES,
)
from eval_common import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/rellis_3d/Rellis-3D",
                        help="Rellis-3D root dir (contains train.lst/val.lst/test.lst and 00000..00004/).")
    parser.add_argument("--split", default="test",
                        help="One of train|val|test (each .lst file has 2 cols: <img> <label>).")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--input_h", type=int, default=480,
                        help="Resize target H. Default 480 matches every other OTAS row "
                             "and is what the paper presumably used (the OTAS_small.json "
                             "shipped config defaults are 480x640).")
    parser.add_argument("--input_w", type=int, default=640)
    parser.add_argument("--enable_mask_refinement", action="store_true",
                        help="Turn OTAS SAM2 mask refinement on. Paper Table V Row 4 "
                             "(OTAS w. DINOv2 ViT-S/14 = 48.48 mIoU) uses --no-SAM (i.e. "
                             "this flag omitted). Output dir gets '_sam' suffix when set.")
    parser.add_argument("--ontology", default="raw20", choices=["raw20", "hrnet19"],
                        help="Which RELLIS ontology to evaluate. raw20 = 20 classes "
                             "(unknown + 19 raw fg, matches RADSeg's wiring). hrnet19 = "
                             "19 classes (unknown + 18 fg) under HRNet's label_mapping "
                             "(canonical 2D RELLIS benchmark — likely what the OTAS paper "
                             "Table V reports on).")
    parser.add_argument("--paper_config", action="store_true",
                        help="Match the OTAS paper Table V / supplementary §VII.A config "
                             "for RELLIS-3D: input image resized to 1024x1024 (matches "
                             "the paper's 'Input images are resized accordingly to "
                             "1024×1024'), shared_feat_resolution=64, n_components=24, "
                             "dinov2_input_size=224 (DINOv2 internally extracts a 16x16 "
                             "grid, bilinear-interpolated up to d=64). No mask refinement, "
                             "no negative prompt, all 20 ontology.yaml class names as "
                             "positive prompts. Output dir gets '_paper' suffix.")
    parser.add_argument("--out_suffix", default=None,
                        help="Override the output dir suffix. Default: '_sam' if SAM on, "
                             "'_hrnet19' if --ontology=hrnet19, '_paper' if --paper_config, "
                             "concatenated as needed. Set explicitly to override.")
    args = parser.parse_args()

    if args.out_suffix is None:
        suffix = "_hrnet19" if args.ontology == "hrnet19" else ""
        if args.paper_config:
            suffix += "_paper"
        if args.enable_mask_refinement:
            suffix += "_sam"
    else:
        suffix = args.out_suffix
    out_dir = Path(args.out_root) / f"RELLIS_rgb{suffix}"

    # --paper_config also overrides the dataset input H/W to 1024x1024 (per §VII.A).
    # CLI-provided --input_h/--input_w still wins if explicitly set above the default.
    input_h, input_w = args.input_h, args.input_w
    if args.paper_config and (input_h, input_w) == (480, 640):  # defaults — bump them
        input_h, input_w = 1024, 1024
        print(f"[own_RELLIS] --paper_config: bumping dataset input to {input_h}x{input_w}")

    dataset = RELLIS_dataset(
        data_dir=args.data_dir,
        split=args.split,
        input_h=input_h,
        input_w=input_w,
        mask_modality="none",
        ontology=args.ontology,
    )
    class_names = (RELLIS_HRNET_CLASS_NAMES
                   if args.ontology == "hrnet19" else RELLIS_CLASS_NAMES)

    # Smoke-check: make sure the dataset loaded sane label values before paying the
    # 10-minute encoder warmup + ~1672-frame eval cost. Same invariant as RADSeg's
    # verify_rellis.py "(3) GT remap on real frames" check.
    print(f"[own_RELLIS] ontology={args.ontology} ({len(class_names)} classes); "
          f"dataset size: {len(dataset)} frames; sample IDs in 3 frames:")
    import numpy as np
    valid = set(range(len(class_names)))
    seen = set()
    for i in (0, len(dataset) // 2, len(dataset) - 1):
        _, lbl, name, _ = dataset[i]
        ids = set(np.unique(lbl.numpy()).tolist())
        leaks = ids - valid
        assert not leaks, f"frame {i} ({name}) leaked GT values: {sorted(leaks)}"
        seen |= ids
        print(f"  frame {i:>4} ({name}): GT ids = {sorted(ids)}")
    print(f"[own_RELLIS] union of GT ids over 3 frames: {sorted(seen)} (all in 0..{len(class_names)-1})\n")

    config_overrides = {"enable_mask_refinement": bool(args.enable_mask_refinement)}
    if args.paper_config:
        # OTAS paper Table V / §VII.A: 64x64 shared patch grid, 24 PCA components,
        # DINOv2 input 224 to yield a native 16x16 grid (which OTAS bilinearly
        # interpolates up to the 64x64 shared resolution). k=24 is already the
        # OTAS default. No negative prompt, no mask refinement, all 20 ontology.yaml
        # class names as positive prompts (this driver always passes class_names
        # as the positive set and OTASEncoder treats them as a 20-way argmax).
        config_overrides.update({
            "shared_feat_resolution": 64,
            "n_components": 24,
            "dinov2_input_size": 224,
        })
        print(f"[own_RELLIS] --paper_config: overriding "
              f"shared_feat_resolution=64, n_components=24, dinov2_input_size=224")
    run_eval(
        dataset=dataset,
        class_names=class_names,
        out_dir=str(out_dir),
        modality="rgb",
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()
