# OTAS evaluation on CART (Caltech Aerial RGB-Thermal labeled_rgbt_pairs).
#
# Mirrors OpenRSS's own_CART.py — same 11-class ontology (unknown absorbing raw 0/Unknown
# AND raw 1/Background) and the same `--handheld` flag that restricts the 2282-frame full
# CART set to the 493-frame big_bear_ONR + caltech-coregistered-nature-dataset_ONR subset.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/code/OpenRSS")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from util.CART_dataset import (
    CART_dataset, HandheldCART_dataset, CART_CLASS_NAMES,
)
from eval_common import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/cart")
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "thermal"],
                        choices=["rgb", "thermal"])
    # input_h/input_w default to None → CART_dataset returns frames at native
    # 600×960 with no resize. Pass explicit --input_h --input_w to override.
    parser.add_argument("--input_h", type=int, default=None)
    parser.add_argument("--input_w", type=int, default=None)
    parser.add_argument("--handheld", action="store_true",
                        help="Restrict to handheld CART subset (493 frames).")
    args = parser.parse_args()

    Cls = HandheldCART_dataset if args.handheld else CART_dataset
    tag = "HANDHELD_CART" if args.handheld else "CART"

    for modality in args.modalities:
        dataset = Cls(
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            split=args.split,
            input_h=args.input_h,
            input_w=args.input_w,
            mask_modality="none",
        )
        out_dir = Path(args.out_root) / f"{tag}_{modality}"
        run_eval(
            dataset=dataset,
            class_names=CART_CLASS_NAMES,
            out_dir=str(out_dir),
            modality=modality,
        )


if __name__ == "__main__":
    main()
