import numpy as np
import matplotlib.pyplot as plt


def get_color_map():
    cmap = plt.get_cmap("tab10")
    return {
        0: (128, 128, 128),  # Background: Grey
        1: (31, 119, 180),  # Blue
        2: (255, 127, 14),  # Orange
        **{i: tuple(int(c * 255) for c in cmap(i - 1)[:3]) for i in range(3, 20)}
    }


def show_mask(mask, image=None, obj_id=None):
    color_map = get_color_map()

    color = np.array(color_map[1][::-1]) / 255.0  # Normalize to [0, 1] range
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