# OTAS open-vocabulary evaluation on BASEPROD (planetary rover terrain, 8-class ontology).
#
# Mirrors OpenRSS's own_BASEPROD.py: re-use OpenRSS's BASEPROD_dataset adapter unchanged
# (per-frame normalized thermal CSV + RGB pair, 950 frames) with the canonical 8-class
# ontology and unified `unknown` at contig 0. Runs OTAS over two modalities.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/code/OpenRSS")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from util.BASEPROD_dataset import BASEPROD_dataset, BASEPROD_CLASS_NAMES
from eval_common import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/baseprod")
    parser.add_argument("--split_dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "thermal"],
                        choices=["rgb", "thermal"])
    # input_h/input_w default to None → BASEPROD_dataset returns frames at
    # native 720×1280 with no resize (see BASEPROD_dataset for the None →
    # label-native rule). Pass explicit --input_h --input_w to override.
    parser.add_argument("--input_h", type=int, default=None)
    parser.add_argument("--input_w", type=int, default=None)
    args = parser.parse_args()

    for modality in args.modalities:
        dataset = BASEPROD_dataset(
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            split=args.split,
            input_h=args.input_h,
            input_w=args.input_w,
            mask_modality="none",
        )
        out_dir = Path(args.out_root) / f"BASEPROD_{modality}"
        run_eval(
            dataset=dataset,
            class_names=BASEPROD_CLASS_NAMES,
            out_dir=str(out_dir),
            modality=modality,
        )


if __name__ == "__main__":
    main()
