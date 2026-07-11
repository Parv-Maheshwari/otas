# OTAS open-vocabulary evaluation on Great Outdoors (GOD).
#
# Mirrors OpenRSS's own_GOD.py one-for-one in shape: instantiate the OpenRSS GOD_dataset
# adapter (re-used unchanged from /home/ubuntu/code/OpenRSS/util/GOD_dataset.py) and
# the canonical 22-class GOD ontology with unified `unknown` at contig 0, then run the
# shared OTAS eval loop for two modalities (RGB and thermal-as-RGB replica).
#
# Produces results.txt and cached per-frame predictions under
# /home/ubuntu/code/OTAS/result/Pred/GOD_<modality>/.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/code/OpenRSS")  # GOD_dataset lives here
sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval_common

from util.GOD_dataset import GOD_dataset, GOD_CLASS_NAMES
from eval_common import run_eval


# GOD ontology palette — verbatim copy of RADSeg's GREAT_OUTDOORS_UNKNOWN_PALETTE
# (= [0,0,0] for "unknown" + GREAT_OUTDOORS_PALETTE for the 21 fg classes), itself
# mirroring the colors in GOD_Ontology.png. Used so OTAS overlays, the RADSeg viz
# grid, and the dataset's published key all show the same color for each class
# (dirt=brown, grass=dark green, trees=bright green, sky=blue, vehicle=yellow, ...).
GOD_PALETTE = [
    [0, 0, 0],          # unknown
    [101, 67, 33],      # dirt
    [0, 128, 0],        # grass
    [0, 255, 0],        # trees
    [0, 128, 128],      # pole
    [0, 128, 255],      # water
    [0, 0, 255],        # sky
    [255, 255, 0],      # vehicle
    [255, 0, 128],      # object
    [64, 64, 64],       # asphalt
    [255, 0, 0],        # building
    [128, 0, 0],        # log
    [192, 128, 255],    # person
    [102, 0, 204],      # fence
    [255, 192, 203],    # bush
    [192, 192, 192],    # concrete
    [51, 153, 255],     # barrier
    [128, 255, 255],    # puddle
    [128, 64, 0],       # mud
    [128, 0, 128],      # rubble
    [153, 0, 255],      # mulch
    [128, 128, 128],    # gravel
    [240, 240, 240],    # snow
]
assert len(GOD_PALETTE) == len(GOD_CLASS_NAMES), (
    f"GOD palette/class-name length mismatch: {len(GOD_PALETTE)} vs {len(GOD_CLASS_NAMES)}"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ubuntu/mnt/great-outdoors-dataset")
    parser.add_argument("--split_dir", default="/home/ubuntu/code/OpenRSS/datasets/god")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_root", default="/home/ubuntu/code/OTAS/result/Pred")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "thermal"],
                        choices=["rgb", "thermal"])
    parser.add_argument(
        "--redraw_overlays_only", action="store_true",
        help="Skip OTAS inference; only re-render sampled overlays from cached preds/. "
             "Use after changing GOD_PALETTE to refresh visualizations.",
    )
    # input_h/input_w default to 480/640 to match the OpenRSS-side fix for the
    # SAM-LoRA top-left-crop bug on pylon-native inputs (see
    # /home/ubuntu/code/OpenRSS/own_GOD.py header comment). Holding all three
    # systems (RADSeg, OTAS, OpenRSS) on the same 480×640 GOD scoring grid keeps
    # the cross-system numbers in baseline.md comparing the same denominator.
    # DINOv2 always resizes internally to dinov2_input_size (224); the dataset
    # H/W only controls the GT scoring grid and the bilinear pred-upsample
    # target. Pass explicit --input_h --input_w to override (e.g. for a
    # pylon-native sanity run).
    parser.add_argument("--input_h", type=int, default=480)
    parser.add_argument("--input_w", type=int, default=640)
    args = parser.parse_args()

    for modality in args.modalities:
        # Pick the GOD_dataset mask_modality string that matches OTAS's
        # per-pass modality. This drives two things in the OpenRSS adapter:
        #   1. Channel zeroing — harmless here because eval_common slices the
        #      right 3 channels (R,G,B or T-as-R=G=B) before handing to DINOv2;
        #      the zeroed-out other channel is ignored downstream.
        #   2. Label-camera switch — REQUIRED for correctness on GOD:
        #      "rgb_only" loads pylon_camera_node_label_id (RGB-frame), and
        #      anything else loads lwir_camera_node_label_id (LWIR-frame).
        #      Using LWIR labels for an RGB-camera-frame pass scores
        #      predictions against masks drawn in a different physical
        #      viewpoint and is wrong (the historical bug we are fixing here).
        mask_modality = "rgb_only" if modality == "rgb" else "thermal_only"
        dataset = GOD_dataset(
            data_dir=args.data_dir,
            split=args.split,
            split_dir=args.split_dir,
            input_h=args.input_h,
            input_w=args.input_w,
            mask_modality=mask_modality,
        )
        out_dir = Path(args.out_root) / f"GOD_{modality}"
        run_eval(
            dataset=dataset,
            class_names=GOD_CLASS_NAMES,
            out_dir=str(out_dir),
            modality=modality,
            palette=GOD_PALETTE,
            redraw_overlays_only=args.redraw_overlays_only,
        )


if __name__ == "__main__":
    main()
