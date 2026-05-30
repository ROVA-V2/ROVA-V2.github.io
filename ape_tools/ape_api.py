# Copyright (c) Facebook, Inc. and its affiliates.
"""
APE detection backend for Robust-TO's detect_objects tool (Table 17).

Runs APE (Aligning and Prompting Everything) and returns, per input image, a
list of (label, [x, y, w, h], score) detections.  The Robust-TO tool wrapper
(rovid_pipeline/tools/perception_tools.py: DetectObjects) discards boxes scoring
below 0.3 and reports the mean of the surviving scores as the tool's intrinsic
confidence c_intrinsic, matching paper Section B.1.

NOTE: paper Section B.1 names GroundingDINO-T as the detector; this release uses
APE.  Swap the backend here to reproduce with GroundingDINO-T -- the returned
(label, box, score) contract is all DetectObjects depends on.
"""
import os
from collections import abc

import tqdm
from detectron2.config import LazyConfig
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo

import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


def setup_cfg():
    config_file = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
    opts = [
        'train.init_checkpoint=/checkpoints/model_final.pth',
        'model.model_language.cache_dir=',
        'model.model_vision.select_box_nums_for_evaluation=500',
        'model.model_vision.text_feature_bank_reset=True',
        'model.model_vision.backbone.net.xattn=False',
    ]
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    # Keep a low service-side threshold; the Robust-TO tool applies the paper's
    # 0.3 discard threshold (Section B.1) on top of whatever is returned.
    confidence_threshold = 0.1

    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    if "train" in cfg.dataloader:
        if isinstance(cfg.dataloader.train, abc.MutableSequence):
            for i in range(len(cfg.dataloader.train)):
                if "output_dir" in cfg.dataloader.train[i].mapper:
                    cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
        else:
            if "output_dir" in cfg.dataloader.train.mapper:
                cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = confidence_threshold
    else:
        cfg.model.test_score_thresh = confidence_threshold

    setup_logger(name="ape")
    setup_logger(name="timm")
    return cfg


def ape_inference(input, text_prompt, demo):
    """
    Returns
    -------
    list (one entry per input image) of list of (label, [x, y, w, h], score).
    Images that fail to load contribute an empty list.
    """
    res_list = []
    for path in tqdm.tqdm(input):
        try:
            img = read_image(path, format="BGR")
        except Exception:
            res_list.append([])
            continue

        predictions, _, _, metadata = demo.run_on_image(
            img, text_prompt=text_prompt,
            with_box=True, with_mask=False, with_sseg=False,
        )

        dets = []
        if "instances" in predictions:
            results = instances_to_coco_json(
                predictions["instances"].to(demo.cpu_device), path
            )
            for r in results:
                label = metadata.thing_classes[r["category_id"]]
                box = [int(v) for v in r["bbox"]]          # [x, y, w, h]
                score = float(r.get("score", 0.0))
                dets.append((label, box, score))
        res_list.append(dets)
    return res_list


if __name__ == "__main__":
    # Minimal manual smoke test (requires checkpoints + a real image path).
    print(ape_inference([], "Apples,Candles,Berries", demo=None))
