import cv2
import os
import json
import tqdm
import numpy as np
from pathlib import Path
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
        10: (0, 0, 128),      # Object 10: Navy Blue
        11: (255, 69, 0),     # Object 11: Red-Orange
        12: (0, 128, 128),    # Object 12: Teal
        13: (128, 0, 128),    # Object 13: Purple
        14: (255, 192, 203),  # Object 14: Pink
        15: (255, 215, 0),    # Object 15: Gold
        16: (0, 255, 127),    # Object 16: Spring Green
        17: (70, 130, 180),   # Object 17: Steel Blue
        18: (255, 240, 245),  # Object 18: Lavender Blush
        19: (255, 20, 147),   # Object 19: Deep Pink
    }


def extract_masks_from_rgb(rgb_mask):
    color_map = {v: k for k, v in get_color_map().items()}

    stacked_masks = {}
    for rgb, label in color_map.items():
        # Create a binary mask for each label
        binary_mask = np.all(rgb_mask == rgb, axis=-1)
        if label != 0 and np.sum(binary_mask):  # Exclude background from the dictionary
            stacked_masks[label - 1] = np.expand_dims(binary_mask, axis=0)
    return stacked_masks


def rle_encode_mask(mask_dict):
    rle_encoded_masks = {}

    for mask_id, binary_mask in mask_dict.items():
        rle = mask_utils.encode(np.asfortranarray(binary_mask.squeeze().astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rle_encoded_masks[mask_id] = rle
    return rle_encoded_masks


def main(base_dir):
    for video_folder in tqdm.tqdm(os.listdir(base_dir)):
        print(f'Video: {video_folder}')
        video_path = os.path.join(base_dir, video_folder)
        if not os.path.isdir(video_path):
            continue  # Skip files, we are only interested in directories

        output_masks_path = os.path.join(video_path, "output_masks")
        if not os.path.exists(output_masks_path):
            continue  # Skip if output_masks folder doesn't exist

        for clip_folder in os.listdir(output_masks_path):
            clip_path = os.path.join(output_masks_path, clip_folder)
            if not os.path.isdir(clip_path):
                continue  # Skip files, we are only interested in directories
            # if any([f for f in os.listdir(clip_path) if f.endswith('.json')]):
            #     continue

            # Collect all image files in the clip folder
            images = [
                os.path.join(clip_path, img)
                for img in os.listdir(clip_path)
                if img.lower().endswith('.png')
            ]
            # Add the json file
            for img_path in images:
                img = cv2.imread(os.path.join(img_path))
                stacked_masks = extract_masks_from_rgb(img)
                rle_encoded_masks = rle_encode_mask(stacked_masks)
                mask_output_path = Path(img_path).with_suffix('.json')
                with open(mask_output_path, 'w') as f:
                    json.dump(rle_encoded_masks, f, indent=4)  # takes 0.009 ms



if __name__ == "__main__":
    base_dir = '/home/adam/mnt/qnap/annotation_data/data/sam2'
    main(base_dir)
