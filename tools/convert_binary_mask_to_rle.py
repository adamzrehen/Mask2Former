import cv2
import os
import json
import tqdm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from pycocotools import mask as mask_utils


def get_color_map():
    return {
        0: (128, 128, 128),   # Background: Grey
        1: (255, 0, 0),       # Object 1: Red
        2: (0, 255, 0),       # Object 2: Green
        3: (0, 0, 255),       # Object 3: Blue
        4: (255, 255, 0),     # Object 4: Yellow
        5: (0, 255, 255),     # Object 5: Cyan
        6: (255, 0, 255),     # Object 6: Magenta
        7: (255, 165, 0),     # Object 7: Orange
        8: (128, 0, 0),       # Object 8: Maroon
        9: (0, 128, 0),       # Object 9: Dark Green
    }


def extract_masks_from_rgb(rgb_mask, color_map):
    rgb_flat = rgb_mask.reshape(-1, 3)
    mask_dict = {}
    for rgb, label in color_map.items():
        if label == 0:
            continue  # Skip background
        binary_mask = np.all(rgb_flat == rgb, axis=1).reshape(rgb_mask.shape[:2])
        if binary_mask.any():
            mask_dict[label - 1] = binary_mask
    return mask_dict

def rle_encode_mask(mask_dict):
    rle_encoded = {}
    for mask_id, binary_mask in mask_dict.items():
        rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rle_encoded[mask_id] = rle
    return rle_encoded

def process_image(args):
    img_path, color_map = args
    img = cv2.imread(img_path)
    stacked_masks = extract_masks_from_rgb(img, color_map)
    rle_encoded_masks = rle_encode_mask(stacked_masks)
    mask_output_path = Path(img_path).with_suffix('.json')
    with open(mask_output_path, 'w') as f:
        json.dump(rle_encoded_masks, f, indent=4)

def main_parallel(base_dir):
    color_map = {v: k for k, v in get_color_map().items()}
    tasks = []

    for video_dir in os.listdir(base_dir):
        video_path = os.path.join(base_dir, video_dir)

        masks_path = os.path.join(video_path, 'output_masks')
        if os.path.exists(masks_path):
            for clip_dir in os.listdir(masks_path):
                clip_path = os.path.join(masks_path, clip_dir)

                for file in os.listdir(clip_path):
                    file_path = os.path.join(clip_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith('.png'):
                        tasks.append((file_path, color_map))

    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))



if __name__ == "__main__":
    base_dir = '/home/adam/mnt/qnap/annotation_data/data/sam2'
    main_parallel(base_dir)
