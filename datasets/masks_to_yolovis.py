import numpy as np
import glob
import os
import json
import tqdm
import pandas as pd
from PIL import Image


palette = {
    (128, 128, 128): 0,
    (0, 255, 0): 1,
    (255, 0, 0): 2,
    (0, 0, 255): 3,
    (0, 0, 0): 4,
    (255, 255, 0): 5,
    (0, 255, 255): 6,
}


def decode_mask(mask_image, palette):
    """Decode a mask image into a segmentation mask using the provided palette."""
    mask_array = np.array(mask_image)
    segmentation_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)

    for color, class_id in palette.items():
        segmentation_mask[(mask_array == color).all(axis=-1)] = class_id

    return segmentation_mask


def binary_mask_to_rle_np(binary_mask):
    """Convert a binary mask to RLE format."""
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle


def process_sequence(group, sequence_id, category_id, base_dir):
    """
    Process a DataFrame group as a sequence.

    Args:
        group (pd.DataFrame): Grouped DataFrame containing mask file information.
        sequence_id (int): ID of the sequence.
        category_id (int): Category ID for all objects.

    Returns:
        dict: Processed sequence data including segmentation and bounding box information.
    """
    segmentations = []
    bboxes = []
    areas = []

    for _, row in group.iterrows():
        mask_path = os.path.join(base_dir, row['Mask Path']) if not pd.isna(row['Mask Path']) else None
        frame_path = os.path.join(base_dir, row['Frame Path']) if 'Frame Path' in row else None

        # Load mask image or create an empty one
        if mask_path and os.path.exists(mask_path):
            mask_image = Image.open(mask_path).convert("RGB")
        elif frame_path and os.path.exists(frame_path):
            # Create an empty mask with the same dimensions as the frame
            frame_image = Image.open(frame_path)
            mask_image = Image.new("RGB", frame_image.size, (0, 0, 0))

        width, height = mask_image.size

        # Decode the mask
        segmentation_mask = decode_mask(mask_image, palette)

        # Generate bounding box from segmentation
        positions = np.where(segmentation_mask > 0)
        if positions[0].size > 0 and positions[1].size > 0:
            ymin, ymax = positions[0].min(), positions[0].max()
            xmin, xmax = positions[1].min(), positions[1].max()
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = (xmax - xmin) * (ymax - ymin)
            rle = binary_mask_to_rle_np(segmentation_mask)
        else:
            bbox = None
            area = 0
            rle = None

        segmentations.append({
            "counts": rle["counts"] if rle else [],
            "size": [height, width]
        })
        bboxes.append(bbox)
        areas.append(area)

    return {
        "id": sequence_id,
        "file_names": group['Mask Path'].tolist(),
        "height": height,
        "width": width,
        "length": len(group),
        "annotations": [{
            "id": sequence_id,
            "video_id": sequence_id,
            "category_id": category_id,
            "segmentations": segmentations,
            "bboxes": bboxes,
            "areas": areas,
            "iscrowd": 0
        }]
    }


def main(base_dir, csv_path, output_json, category_id=1):
    """
    Main function to process all sequences in a DataFrame and generate JSON.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing mask file information.
        output_json (str): Path to save the generated JSON file.
        category_id (int): Category ID for all objects (default: 1).
    """
    video_annotations = []
    videos = []
    categories = [{"id": category_id, "name": "object", "supercategory": "generic"}]
    sequence_id = 1
    dataframe = pd.read_csv(csv_path)

    # Group by 'Video' and 'Clip ID'
    grouped = dataframe.groupby(['Video', 'Clip ID'])

    for (video_name, clip_id), group in tqdm.tqdm(grouped):
        # Process the group as a sequence
        sequence_data = process_sequence(group, sequence_id, category_id, base_dir)

        videos.append({
            "id": int(sequence_data["id"]),  # Ensure conversion to Python int
            "file_names": group['Frame Path'].tolist(),
            "height": int(sequence_data["height"]),
            "width": int(sequence_data["width"]),
            "length": int(len(group))
        })
        video_annotations.extend(sequence_data["annotations"])
        sequence_id += 1

    # Combine into the final dataset structure
    dataset = {
        "info": {
            "description": "Converted YouTube-VIS dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "Script",
            "date_created": "2024-11-20"
        },
        "videos": videos,
        "annotations": video_annotations,
        "categories": categories
    }

    # Save to JSON with safe conversion
    def convert_np(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy scalars to Python types
        raise TypeError("Object of type {0} is not JSON serializable".format(type(obj)))

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4, default=convert_np)


# Example usage
if __name__ == "__main__":
    base_dir = "/home/adam/mnt/qnap/annotation_data/data/sam2/"  # Root directory containing sequence folders
    csv_path = '/home/adam/Downloads/filtered_data.csv'
    output_json = "/home/adam/Downloads/filtered_data.json"
    main(base_dir, csv_path, output_json)