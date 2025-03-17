import numpy as np
import os
import cv2
import tempfile
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
    }


def check_overlap(mask, prediction, min_overlap=50):
    mask = mask.astype(bool)
    prediction = prediction.astype(bool)
    overlap = np.sum(mask & prediction)
    if overlap >= min_overlap:
        return True
    return False


def show_mask(mask, image=None, obj_id=None):
    color_map = get_color_map()

    color = np.array(color_map[obj_id + 1]) / 255.0  # Normalize to [0, 1] range
    color = np.append(color, 0.6)  # Add alpha channel

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[..., c] = mask_image[..., c]
        alpha_mask = mask_image[..., 3] / 255.0
        for c in range(3):
            image[..., c] = np.where(alpha_mask > 0,
                                     (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c],
                                     image[..., c])
        return image
    return mask_image


def rle_decode_mask(rle_dict):
    stacked_masks = {}
    for key, val in rle_dict.items():
        mask = mask_utils.decode(val)
        stacked_masks[int(key)] = mask
    return stacked_masks


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False