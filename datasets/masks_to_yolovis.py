import numpy as np
import os
import json
import tqdm
import pandas as pd
from pycocotools import mask as mask_utils
from PIL import Image


categories = [{"id": 1, "name": "adenoma", "supercategory": "generic"},
              {"id": 2, "name": "scc", "supercategory": "generic"},
              {"id": 3, "name": "adenocarcinoma", "supercategory": "generic"},
              {"id": 4, "name": "normal_tissue", "supercategory": "generic"},
              {"id": 5, "name": "other", "supercategory": "generic"}]


# Create a dictionary mapping category names to IDs
category_dict = {cat["name"]: cat["id"] for cat in categories}


# Function to get the ID given a category name
def get_category_id(category_name):
    return category_dict.get(category_name, None)  # Returns None if the name is not found


def split_groups(grouped, n_splits):
    # Initialize an empty list to hold the split groups
    split_groups = []

    # Iterate through each group
    for (_, _), group in grouped:
        # Use np.array_split to split the group into 5 parts
        splits = np.array_split(group, n_splits)

        # Assign a split ID to each part
        for i, split in enumerate(splits):
            split['Split ID'] = i + 1
            split_groups.append(split)
    return split_groups


def rle_encode_mask(mask_dict):
    rle_encoded_masks = {}

    for mask_id, binary_mask in mask_dict.items():
        rle = mask_utils.encode(np.asfortranarray(binary_mask.squeeze().astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rle_encoded_masks[mask_id] = rle
    return rle_encoded_masks


def process_sequence(group, sequence_id, base_dir, misalignment_dict):
    annotations = {}
    no_objects = 0
    for k, (_, row) in enumerate(group.iterrows()):
        mask_path = os.path.join(base_dir, row['relative_mask_path']) if not pd.isna(row['relative_mask_path']) else None
        frame_path = os.path.join(base_dir, row['relative_image_path']) if not pd.isna(row['relative_image_path']) else None
        category_types = eval(row['object_type'])

        if k == 0 and frame_path is not None:
            image = Image.open(frame_path)
            width, height = image.size

        # Skip if the mask_path doesn't exist
        if mask_path is None or len(mask_path) == 0 or not os.path.exists(mask_path):
            img = np.zeros((height, width))
            rle_data = rle_encode_mask({-1: img})
            no_objects += 1
        else:
            # Read RLE mask from JSON file
            try:
                with open(mask_path, 'r') as f:
                    rle_data = json.load(f)
            except:
                rle_data = None
            if not rle_data:
                img = np.zeros((height, width))
                rle_data = rle_encode_mask({-1: img})
                no_objects += 1

        for key, val in rle_data.items():
            # Decode RLE mask
            segmentation_mask = mask_utils.decode(val)
            if len(category_types) - 1 >= int(key):
                category_id = get_category_id(category_types[int(key)])
            else:
                if not row['video_name'] in misalignment_dict:
                    misalignment_dict[row['video_name']] = {}
                if not row['clip_id'] in misalignment_dict[row['video_name']]:
                    misalignment_dict[row['video_name']][row['clip_id']] = 0
                misalignment_dict[row['video_name']][row['clip_id']] += 1
                continue

            # Generate bounding box from segmentation
            positions = np.where(segmentation_mask > 0)
            if positions[0].size > 0 and positions[1].size > 0:
                ymin, ymax = positions[0].min(), positions[0].max()
                xmin, xmax = positions[1].min(), positions[1].max()
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)
            else:
                bbox = [0, 0, width, height]
                area = 0

            # Add data to annotations
            if int(key) not in annotations:
                annotations[int(key)] = {
                    "segmentations": [None for _ in range(len(group))],
                    "bboxes": [None for _ in range(len(group))],
                    "height": height,
                    "width": width,
                    "areas": [None for _ in range(len(group))],
                    "id": sequence_id,  # according to ytvoseval this should be video_id
                    "video_id": sequence_id,
                    "category_id": 1,  # category_id
                    "iscrowd": 0,
                    "length": 1
                }

            annotations[int(key)]['segmentations'][k] = {
                "counts": rle_data[key]["counts"],
                "size": [rle_data[key]["size"][0], rle_data[key]["size"][1]]
            }
            annotations[int(key)]['bboxes'][k] = bbox
            annotations[int(key)]['areas'][k] = area

    if annotations.keys() == {-1}:
        annotations = []
    else:
        annotations = list(annotations.values())

    return annotations, height, width, no_objects, misalignment_dict


def main(base_dir, csv_path, output_json, test=False):
    video_annotations = []
    videos = []

    sequence_id = 1
    dataframe = pd.read_csv(csv_path)

    # Group by 'Video' and 'Clip ID'
    grouped = dataframe.groupby(['unique_name'])

    if test:
        grouped = split_groups(grouped, n_splits=5)
    else:
        grouped = [group for _, group in grouped]

    grouped = [_ for _ in grouped if len(_) > 1]
    total_no_objects = 0
    total_frames = 0
    misalignment_dict = {}
    for group in tqdm.tqdm(grouped):
        # Process the group as a sequence
        sequence_data, height, width, no_objects, misalignment_dict = process_sequence(group, sequence_id, base_dir,
                                                                                       misalignment_dict)
        total_no_objects += no_objects
        total_frames += len(group)

        videos.append({
            "id": sequence_id,  # Ensure conversion to Python int
            "file_names": group['relative_image_path'].tolist(),
            "height": height,
            "width": width,
            "length": int(len(group))
        })
        video_annotations.extend(sequence_data)
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
    print(f'Total number of frames: {total_frames} ')
    print(f'Total number of frames without objects: {total_no_objects} ')
    print(f'Misalignment between rle data and metadata in videos / clips:')
    for video_name, val in misalignment_dict.items():
        print(video_name, end=": ")  # Print video name followed by a colon
        print(", ".join(str(clip_id) for clip_id in val.keys()))

    # Save to JSON with safe conversion
    def convert_np(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy scalars to Python types
        raise TypeError("Object of type {0} is not JSON serializable".format(type(obj)))

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4, default=convert_np)


# Example usage
if __name__ == "__main__":
    base_dir = "/home/cortica/mnt/qnap/annotation_data/data/sam2/"  # Root directory containing sequence folders
    csv_path = '/home/cortica/Documents/Adam/Experiments/Mask2Former/Experiment 20/test.csv'
    output_json = "/home/cortica/Documents/Adam/Experiments/Mask2Former/Experiment 20/test.json"
    main(base_dir, csv_path, output_json, test=False)