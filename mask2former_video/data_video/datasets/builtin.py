# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

from .kumc import (
    register_kumc_instances,
    _get_kumc_instances_meta
)

from .ichilov import (
    register_ichilov_instances,
    _get_ichilov_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("/home/adam/Documents/Data/ytvis_2021/train/JPEGImages",
                         "/home/adam/Documents/Data/ytvis_2021/train.json"),
    "ytvis_2021_val": ("/home/adam/Documents/Data/ytvis_2021/valid/JPEGImages",
                       "/home/adam/Documents/Data/ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

_PREDEFINED_SPLITS_KUMC = {
    "kumc_train": ("/home/adam/Documents/Data/KUMC Dataset/ytvis_format/train/JPEGImages",
                         "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/train.json"),
    "kumc_val": ("/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid/JPEGImages",
                       "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid.json"),
    "kumc_test": ("/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid/JPEGImages",
                        "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid.json"),
}

_PREDEFINED_SPLITS_ICHILOV = {
    "ichilov_train": ("/home/adam/mnt/qnap/annotation_data/data/sam2",
                         "/home/adam/Documents/Experiments/Mask2Former/train.json"),
    "ichilov_val": ("/home/adam/mnt/qnap/annotation_data/data/sam2",
                       "/home/adam/Documents/Experiments/Mask2Former/test.json"),
    "ichilov_test": ("/home/adam/mnt/qnap/annotation_data/data/sam2",
                        "/home/adam/Documents/Experiments/Mask2Former/test.json"),
}



def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_kumc(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_KUMC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_kumc_instances(
            key,
            _get_kumc_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ichilov(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ICHILOV.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ichilov_instances(
            key,
            _get_ichilov_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_kumc(_root)
    register_all_ichilov(_root)
