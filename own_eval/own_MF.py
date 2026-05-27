# OTAS evaluation on MFNet (the dataset OpenRSS was trained on).
#
# 9-class ontology including unified `unknown` at contig 0. MFNet ships 4-channel RGB+T
# PNGs natively; we re-use OpenRSS's MF_dataset adapter unchanged. 393 frames in the
# `test` split.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/code/OpenRSS")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from util.MF_dataset import MF_dataset
from eval_common import run_eval


# OpenRSS's own_MF.py hardcodes these class names in the per-class log lines
# (lines 55 + 174 + 178 of OpenRSS/own_MF.py). Re-create the list here so the
# prompt set is identical. Capitalization + spaces (not lowercase + underscores)
# is intentional and load-bearing on the OpenRSS side: the LoRA was trained
# against these exact strings ("Car", "Person", "Car Stop", "Color Cone", ...).
# Cross-system audit "use OpenRSS as the canonical version on any dataloader
# mismatch" (2026-05-22) brought this in line. RADSeg's matching
# `configs/cls_mfnet_unknown.txt` and `MFNET_CLASSES` tuple use the same form.
MF_CLASS_NAMES = [
    "unknown", "Car", "Person", "Bike", "Curve",
    "Car Stop", "Guardrail", "Color Cone", "Bump",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/mfnet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "thermal"],
                        choices=["rgb", "thermal"])
    # input_h/input_w default to None → MF_dataset returns the 4-channel
    # RGB+T PNG at its native shape with no resize. Pass explicit
    # --input_h --input_w to override.
    parser.add_argument("--input_h", type=int, default=None)
    parser.add_argument("--input_w", type=int, default=None)
    args = parser.parse_args()

    for modality in args.modalities:
        dataset = MF_dataset(
            data_dir=args.data_dir,
            split=args.split,
            input_h=args.input_h,
            input_w=args.input_w,
            mask_modality="none",
        )
        out_dir = Path(args.out_root) / f"MF_{modality}"
        run_eval(
            dataset=dataset,
            class_names=MF_CLASS_NAMES,
            out_dir=str(out_dir),
            modality=modality,
        )


if __name__ == "__main__":
    main()
