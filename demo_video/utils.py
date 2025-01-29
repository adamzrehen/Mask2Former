import numpy as np


def get_color_map():
    # Note these colors are RGB. Should be inverted for BGR
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


def show_mask(mask, image=None, obj_id=None):
    color_map = get_color_map()
    if obj_id is None or obj_id not in color_map:
        raise ValueError(f"Invalid or missing object ID: {obj_id}. Valid IDs are {list(color_map.keys())}")

    # Get the color for the given object ID
    color = np.array(color_map[obj_id + 1][::-1]) / 255.0  # Normalize to [0, 1] range
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