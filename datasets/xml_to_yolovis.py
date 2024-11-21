import os
import json
import glob
import cv2
import numpy as np
from xml.etree import ElementTree as ET
from pycocotools import mask as coco_mask
from itertools import groupby


def parse_bndbox(xml_file):
    """Parse bounding box from an XML file. Return None if bndbox is missing."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    if bndbox is None:
        return None, None, None, None
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    return xmin, ymin, xmax, ymax


def parse_image_size(xml_file):
    """Extract image dimensions from an XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find(".//size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    return width, height


def generate_segmentation_mask(xmin, ymin, xmax, ymax, width, height):
    """Generate a binary segmentation mask for the bounding box."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 1
    return mask


import numpy as np


import numpy as np

def binary_mask_to_rle_np(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle


def process_sequence(sequence_folder, sequence_id, category_id):
    """Process all frames in a sequence folder, handling cases with missing bndbox."""
    annotations = []
    # Sort frame files numerically
    frame_files = sorted(glob.glob(os.path.join(sequence_folder, "*.xml")),
                         key=lambda x: int(os.path.basename(x).split('.')[0]))

    segmentations = []
    bboxes = []
    areas = []

    for frame_index, xml_file in enumerate(frame_files):
        # Parse image dimensions
        width, height = parse_image_size(xml_file)

        # Check for bounding box
        xmin, ymin, xmax, ymax = parse_bndbox(xml_file)

        if xmin is None:
            # No bounding box: assign Null values
            segmentations.append(None)
            bboxes.append(None)
            areas.append(None)
        else:
            # Bounding box exists: create mask and RLE
            mask = generate_segmentation_mask(xmin, ymin, xmax, ymax, width, height)
            rle = binary_mask_to_rle_np(mask)

            segmentations.append({
                "counts": rle["counts"],
                "size": [height, width]
            })
            bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            areas.append((xmax - xmin) * (ymax - ymin))

    # Return video and annotations structure
    return {
        "id": sequence_id,
        "file_names": [os.path.basename(f).replace(".xml", ".jpg") for f in frame_files],
        "height": height,
        "width": width,
        "length": len(frame_files),
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


def main(data_dir, output_json, category_id=1):
    """Main function to process all sequences and generate JSON."""
    video_annotations = []
    videos = []
    categories = [{"id": category_id, "name": "object", "supercategory": "generic"}]
    sequence_id = 1

    for sequence_folder in sorted(os.listdir(data_dir)):
        sequence_path = os.path.join(data_dir, sequence_folder)
        if os.path.isdir(sequence_path):
            video_data = process_sequence(sequence_path, sequence_id, category_id)
            videos.append({
                "id": video_data["id"],
                "file_names": [os.path.join(sequence_folder, _) for _ in video_data["file_names"]],
                "height": video_data["height"],
                "width": video_data["width"],
                "length": video_data["length"]
            })
            video_annotations.extend(video_data["annotations"])
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

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4)


# Example usage
if __name__ == "__main__":
    DATA_DIR = "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid/Annotations"  # Root directory containing sequence folders
    OUTPUT_JSON = "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid.json"
    main(DATA_DIR, OUTPUT_JSON)
