# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
# fmt: off
import sys
from networkx.algorithms.clique import enumerate_all_cliques

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import pickle
import re
import json
import cv2
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from collections import defaultdict
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from pathlib import Path
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from predictor import VisualizationDemo
from utils import show_mask, rle_decode_mask
from  calculate_metrics import compute_detection_statistics

# constants
WINDOW_NAME = "mask2former video demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def convert_to_df(data):
    rows = []

    # Iterate over the dictionary and extract information
    for video_name, clips in data.items():
        for clip_id, objects in clips.items():
            for object_id, metrics in objects.items():
                row = {
                    'Video Name': video_name,
                    'Clip ID': clip_id,
                    'Object ID': object_id,
                    'Detections': metrics['detections'] if 'detections' in metrics else 0,
                    'Misdetections': metrics['misdetections'] if 'misdetections' in metrics else 0,
                    'False Alarms': metrics['false_alarms'] if 'false_alarms' in metrics else 0,
                    'OK Images': metrics['ok'] if 'ok' in metrics else 0,
                    'Processed': metrics['processed']
                }
                rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)
    return df


def evaluate(args):
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    inference_statistics = {}
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        vid_frames = []
        masks = []
        sorted_paths = sorted(args.input, key=lambda s: int(re.search(r'(\d+)\.(png|jpg)$', s).group(1)))
        for path in sorted_paths:
            img = read_image(path, format="BGR")
            vid_frames.append(img)
            if args.overlay_masks:
                clip_folder = Path(path).parent.stem
                mask_path = Path(path).parents[2] / 'output_masks' / clip_folder / (Path(path).stem + '.json')
                mask = None
                if os.path.exists(mask_path):
                    with open(mask_path, 'r') as f:
                        rle_data = json.load(f)
                    mask = rle_decode_mask(rle_data)
                masks.append(mask)

        video_name = Path(path).parents[2].name
        clip = int(clip_folder[5:])
        print(f'Processing: {video_name} clip {clip}')

        if args.load_predictions:
            predictions_dir = os.path.join(args.inference_output, 'predictions')
            with open(os.path.join(predictions_dir, f'{video_name}_{clip}.pkl'), 'rb') as f:
                prediction_masks = pickle.load(f)
        else:
            start_time = time.time()
            chunk_size = 30
            predictions_list = []
            visualized_output_list = []
            for i in range(0, len(vid_frames), chunk_size):
                chunk = vid_frames[i:i + chunk_size]
                with autocast():
                    predictions, visualized_output = demo.run_on_video(chunk)
                    predictions_list.append(predictions)
                    visualized_output_list.append(visualized_output)
                logger.info(
                    "detected {} instances per frame in {:.2f}s".format(
                        len(predictions["pred_scores"]), time.time() - start_time))
            prediction_masks = defaultdict(list)
            for prediction in predictions_list:
                for k, pred_label in enumerate(prediction['pred_labels']):
                    pred_list = [i for i in prediction['pred_masks'][k].cpu().detach().numpy()]
                    prediction_masks[pred_label].extend(pred_list)

            # Save predictions
            predictions_dir = os.path.join(args.inference_output, 'predictions')
            os.makedirs(predictions_dir, exist_ok=True)
            with open(os.path.join(predictions_dir, f'{video_name}_{clip}.pkl'), 'wb') as f:
                pickle.dump(prediction_masks, f)

        # Get statistics
        if args.inference_output:
            inference = compute_detection_statistics(masks, prediction_masks)

            inference_statistics[video_name] = {clip: inference}
            new_df = convert_to_df(inference_statistics)
            if os.path.exists(os.path.join(args.inference_output, 'inference.csv')):
                existing_df = pd.read_csv(os.path.join(args.inference_output, 'inference.csv'))
                new_df = pd.concat([new_df, existing_df], ignore_index=True)
            new_df.to_csv(os.path.join(args.inference_output, 'inference.csv'), index=False)

        if args.output:
            visualized_output = [item for sublist in visualized_output_list for item in sublist]
            if args.save_frames:
                for path, _vis_output in zip(args.input, visualized_output):
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, args.video_filename + '.mp4'), fourcc, 10.0, (W, H), True)
            for k, _vis_output in enumerate(visualized_output):
                frame = _vis_output.get_image()[:, :, ::-1]
                if args.overlay_masks:
                    if masks[k] is not None:
                        masked_frame = frame.copy()
                        for obj_id, mask in masks[k].items():
                            frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                out.write(frame)
            cap.release()
            out.release()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)

        vid_frames = []
        while video.isOpened():
            success, frame = video.read()
            if success:
                vid_frames.append(frame)
            else:
                break

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames)
        logger.info(
            "detected {} instances per frame in {:.2f}s".format(
                len(predictions["pred_scores"]), time.time() - start_time))

        if args.output:
            if args.save_frames:
                for idx, _vis_output in enumerate(visualized_output):
                    out_filename = os.path.join(args.output, f"{idx}.jpg")
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()


def get_args():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument("--config-file", default="configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images; "
                                                   "or a single glob pattern such as 'directory/*.jpg' "
                                                   "this will be treated as frames of a video")
    parser.add_argument("--output", help="A file or directory to save output visualizations. If not given, "
                                         "will show output in an OpenCV window.")
    parser.add_argument("--inference_output", help="A directory with inference results")
    parser.add_argument("--load_predictions", default=False, help="Load saved predictions")
    parser.add_argument("--video_filename",help="Name of output video", default="visualization")
    parser.add_argument("--save-frames", default=False, help="Save frame level image outputs.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum score for instance predictions to be shown")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER)
    parser.add_argument("--overlay_masks", type=bool, default=False,
                        help="A file or directory to save output visualizations. If not given, will show output in "
                             "an OpenCV window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
