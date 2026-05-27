# OTAS evaluation on BASEPROD with the 5-class condensed ontology (Ground/Grass/Rock/
# Vegetation + unified `unknown` at contig 0). Same 950 frames as own_BASEPROD.py but
# the dataset adapter collapses the 4 raw soil subtypes (Compact / Bedrock / Sandy /
# Pebble) into a single "Ground" superclass — matching the RADSeg condensed row.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/code/OpenRSS")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from util.BASEPROD_condensed_dataset import (
    BASEPROD_condensed_dataset, BASEPROD_CONDENSED_CLASS_NAMES,
)
from eval_common import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/baseprod")
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "thermal"],
                        choices=["rgb", "thermal"])
    # input_h/input_w default to None → BASEPROD_condensed_dataset (inherits
    # BASEPROD_dataset) returns frames at native 720×1280 with no resize. Pass
    # explicit --input_h --input_w to override.
    parser.add_argument("--input_h", type=int, default=None)
    parser.add_argument("--input_w", type=int, default=None)
    args = parser.parse_args()

    for modality in args.modalities:
        dataset = BASEPROD_condensed_dataset(
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            split=args.split,
            input_h=args.input_h,
            input_w=args.input_w,
            mask_modality="none",
        )
        out_dir = Path(args.out_root) / f"BASEPROD_condensed_{modality}"
        run_eval(
            dataset=dataset,
            class_names=BASEPROD_CONDENSED_CLASS_NAMES,
            out_dir=str(out_dir),
            modality=modality,
        )


if __name__ == "__main__":
    main()
