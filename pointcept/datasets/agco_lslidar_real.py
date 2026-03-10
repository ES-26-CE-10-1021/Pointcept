import os
import glob
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class LsLidarRealDataset(DefaultDataset):
    """
    Splits:
    - train: all from static_2_lslidar/split_2
    - val:   first half from static_1_lslidar/split_1
    - test:  second half from static_1_lslidar/split_1
    """
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        # Accepts "train" | "val" | "test" or a list of them
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, (list, tuple)):
            split_list = list(self.split)
        else:
            raise NotImplementedError(f"Unsupported split type: {type(self.split)}")

        data_list = []
        for split in split_list:
            if split == "train":
                split_dir = os.path.join(self.data_root, "static_2_lslidar_paired", "split_2")
                scenes = sorted(d for d in glob.glob(os.path.join(split_dir, "scene_*")) if os.path.isdir(d))
                data_list.extend(scenes)
            elif split in ["val", "test"]:
                split_dir = os.path.join(self.data_root, "static_1_lslidar_paired", "split_1")
                scenes = sorted(d for d in glob.glob(os.path.join(split_dir, "scene_*")) if os.path.isdir(d))
                mid = len(scenes) // 2
                data_list.extend(scenes[:mid] if split == "val" else scenes[mid:])
            else:
                raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")
        return data_list

    def get_data(self, idx):
        # Load coord/segment/etc. from DefaultDataset class
        data_dict = super().get_data(idx)  # coord float32, segment int32 handled here.
        seg = data_dict["segment"].reshape(-1).astype(np.int32)

        # Apply learning map like in agco_all_real_wml (vectorized dict lookup).
        seg = np.vectorize(self.learning_map.__getitem__)(seg).astype(np.int32)
        data_dict["segment"] = seg
        return data_dict

    @staticmethod
    def get_learning_map(ignore_index):
        # Identity over your 4 classes; add extra source IDs here if they appear later.
        # Include ignore_index to avoid KeyError if it leaks in.
        return {
            ignore_index: ignore_index,
            0: 0,  # tractor
            1: 1,  # harvester
            2: 2,  # trailer
            3: 3,  # background
        }

    @staticmethod
    def get_learning_map_inv(ignore_index):
        # Inverse (used if you ever need to export predictions back to source IDs)
        return {
            ignore_index: ignore_index,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }
