import os
import glob
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class AgcoRealDataset(DefaultDataset):
    """
    Splits:
    - train: 
    - val:   
    - test:  
    """
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self.lidar_topic = "_ouster_points"
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.get_data_list()


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
            split_dir = os.path.join(self.data_root, split)
            sequnences = sorted(os.listdir(split_dir))
            for sequnence in sequnences:
                assert os.path.exists(os.path.join(split_dir, sequnence, self.lidar_topic)), "the path dont exist"
                sequnence_ids = os.listdir(os.path.join(split_dir, sequnence, self.lidar_topic, 'coord'))
                data_list.extend(
                    [
                        os.path.join(split_dir, sequnence, self.lidar_topic, 'coord', cloud_id)
                        for cloud_id in sequnence_ids
                    ]
)
        return data_list

    def get_data(self, idx):
        # Load coord/segment/etc. from DefaultDataset class
        data_dict = {}
        coord_path:str = self.data_list[idx % len(self.data_list)] 
        dir_path, filename = os.path.split(coord_path)
        # Replace 'coord' with 'segment' safely
        seg_dir = os.path.join(os.path.dirname(dir_path), "segment")
        seg_path = os.path.join(seg_dir, filename)

        coord = np.load(coord_path).astype(np.float32)
        seg = np.load(seg_path).reshape(-1).astype(np.int32)

        seg = data_dict["segment"] = seg

        # Apply learning map like in agco_all_real_wml (vectorized dict lookup).
        seg = np.vectorize(self.learning_map.__getitem__)(seg).astype(np.int32)
        data_dict["coord"] = coord
        data_dict["segment"] = seg
        return data_dict

    @staticmethod
    def get_learning_map(ignore_index):
        # Identity over your 4 classes; add extra source IDs here if they appear later.
        # Include ignore_index to avoid KeyError if it leaks in.
        return {
            ignore_index: ignore_index,
            0: 0,  # ground
            1: 1,  # tractor
            2: 2,  # combine 
            3: 3,  # trailer
            4: 4,  # truck
            5: 5,  # truck trailer
            6: 6,  # building
            7: 7,  # trees
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
            4: 4,
            5: 5,
            6: 6,
            7: 7,
        }
