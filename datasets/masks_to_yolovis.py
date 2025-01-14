import numpy as np
import os
import json
import tqdm
import pandas as pd
from pycocotools import mask as mask_utils


def process_sequence(group, sequence_id, base_dir):
    annotations = {}
    for k, (_, row) in enumerate(group.iterrows()):
        mask_path = os.path.join(base_dir, row['Mask Path']) if not pd.isna(row['Mask Path']) else None
        category_id = 1  # Replace with actual category ID if available

        # Skip if the mask_path doesn't exist
        if mask_path is None or not os.path.exists(mask_path):
            continue

        # Read RLE mask from JSON file
        with open(mask_path, 'r') as f:
            rle_data = json.load(f)

        for key, val in rle_data.items():
            # Decode RLE mask
            segmentation_mask = mask_utils.decode(val)

            # Generate bounding box from segmentation
            positions = np.where(segmentation_mask > 0)
            if positions[0].size > 0 and positions[1].size > 0:
                ymin, ymax = positions[0].min(), positions[0].max()
                xmin, xmax = positions[1].min(), positions[1].max()
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)
            else:
                bbox = None
                area = 0

            # Add data to annotations
            if key not in annotations:
                annotations[key] = {
                    "segmentations": [None for _ in range(len(group))],
                    "bboxes": [None for _ in range(len(group))],
                    "height": rle_data[key]["size"][0],
                    "width": rle_data[key]["size"][1],
                    "areas": [None for _ in range(len(group))],
                    "id": sequence_id,
                    "video_id": sequence_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "length": 1
                }

            annotations[key]['segmentations'][k] = {
                "counts": rle_data[key]["counts"],
                "size": [rle_data[key]["size"][0], rle_data[key]["size"][1]]
            }
            annotations[key]['bboxes'][k] = bbox
            annotations[key]['areas'][k] = area

    return list(annotations.values())


def main(base_dir, csv_path, output_json):
    video_annotations = []
    videos = []
    categories = [{"id": 1, "name": "Adenoma", "supercategory": "generic"},
                  {"id": 2, "name": "SCC", "supercategory": "generic"},
                  {"id": 3, "name": "Adenocarcinoma", "supercategory": "generic"},
                  {"id": 4, "name": "Normal Tissue", "supercategory": "generic"},
                  {"id": 5, "name": "Other", "supercategory": "generic"}]

    sequence_id = 1
    dataframe = pd.read_csv(csv_path)

    # Group by 'Video' and 'Clip ID'
    grouped = dataframe.groupby(['Video', 'Clip ID'])

    for (_, _), group in tqdm.tqdm(grouped):
        # Process the group as a sequence
        sequence_data = process_sequence(group, sequence_id, base_dir)

        videos.append({
            "id": int(sequence_data[0]["id"]),  # Ensure conversion to Python int
            "file_names": group['Frame Path'].tolist(),
            "height": int(sequence_data[0]["height"]),
            "width": int(sequence_data[0]["width"]),
            "length": int(len(group))
        })
        video_annotations.extend(sequence_data)
        sequence_id += 1

        if sequence_id == 10:
            break

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