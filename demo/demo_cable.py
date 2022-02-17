import argparse
import glob
import os
import time

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.DEVICE = args.device
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument("--input_dir", default="testimgs/ttpla/val")
    # parser.add_argument("--config-file", default="configs/SOLOv2/R50_3x_ttpla.yaml")
    # parser.add_argument("--model", default="models/ttpla_SOLOv2_R50_3x/model_0059999.pth", help="model to use")
    parser.add_argument("--config-file", default="configs/SOLOv2/R50_3x_cable.yaml")
    parser.add_argument("--model", default="models/cable_SOLOv2_R50_3x/model_0044999.pth", help="model to use")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    imgfiles = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg"), recursive=True))
    save_dir = args.input_dir + "_cable_preds"
    os.makedirs(save_dir, exist_ok=True)
    for cur_imgPath in imgfiles:
        # use PIL, to be consistent with evaluation
        img = read_image(cur_imgPath, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                cur_imgPath, len(predictions["instances"]), time.time() - start_time
            )
        )
        save_path = os.path.join(save_dir, os.path.basename(cur_imgPath))
        visualized_output.save(save_path)

