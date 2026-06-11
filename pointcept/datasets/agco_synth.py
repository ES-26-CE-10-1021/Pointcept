import os
import glob
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


def _list_ids(sensor_root):
    paths = sorted(glob.glob(os.path.join(sensor_root, "coord", "*.npy")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if not ids:
        raise RuntimeError(
            f"No .npy files found under: {os.path.join(sensor_root, 'coord')}"
        )
    return ids


class _BaseSynthDataset(DefaultDataset):
    """
    Synthetic lidar dataset with flat folder layout and 80/10/10 split.

    data_root/
      lslidar/
        coord/<id>.npy
        segment/<id>.npy   (optional; if missing -> -1)
        normal/<id>.npy    (optional)
      ouster/
        coord/<id>.npy
        segment/<id>.npy
        normal/<id>.npy
    """

    SENSOR = ""  # override in subclass: "lslidar" or "ouster"

    def __init__(self, ignore_index=-1, split_seed=42, **kwargs):
        """
        Args:
            split: "train" | "val" | "test" | list/tuple of them (handled by DefaultDataset.cfg)
            split_seed: deterministic seed for 80/10/10 partition
        """
        self.ignore_index = ignore_index
        self.split_seed = int(split_seed)
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self._ids = None
        self._split_ids = None
        super().__init__(ignore_index=ignore_index, **kwargs)

    # ---------------- listing & splitting ----------------
    def _ensure_splits(self):
        if self._split_ids is not None:
            return
        sensor_root = os.path.join(self.data_root, self.SENSOR)
        ids = _list_ids(sensor_root)
        # deterministic permutation
        rng = np.random.RandomState(self.split_seed)
        perm = np.array(ids)[rng.permutation(len(ids))]
        # 10-way split => 80/10/10
        parts = np.array_split(perm, 10)
        train_ids = np.concatenate(parts[:8]) if len(parts) >= 8 else perm
        val_ids = parts[8] if len(parts) >= 9 else np.array([], dtype=perm.dtype)
        test_ids = parts[9] if len(parts) >= 10 else np.array([], dtype=perm.dtype)
        self._ids = ids
        self._split_ids = {
            "train": train_ids.tolist(),
            "val": val_ids.tolist(),
            "test": test_ids.tolist(),
        }

    def get_data_list(self):
        # Normalize requested split(s)
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, (list, tuple)):
            split_list = list(self.split)
        else:
            raise NotImplementedError(f"Unsupported split type: {type(self.split)}")

        self._ensure_splits()
        sensor_root = os.path.join(self.data_root, self.SENSOR)

        chosen_ids = []
        for s in split_list:
            if s not in ("train", "val", "test"):
                raise ValueError(f"Unknown split '{s}'. Expected 'train'|'val'|'test'.")
            chosen_ids.extend(self._split_ids[s])

        # store for name lookup
        self._chosen_ids = chosen_ids
        # Return coord paths (nice for logs)
        return [os.path.join(sensor_root, "coord", f"{sid}.npy") for sid in chosen_ids]

    def get_data_name(self, idx):
        return self._chosen_ids[idx % len(self._chosen_ids)]

    def get_split_name(self, idx):
        # Report the configured split if a single split was requested; otherwise "mixed"
        if isinstance(self.split, str):
            return self.split
        return "mixed"

    # ---------------- loading ----------------
    def get_data(self, idx):
        sid = self._chosen_ids[idx % len(self._chosen_ids)]
        base = os.path.join(self.data_root, self.SENSOR)
        coord_path = os.path.join(base, "coord", f"{sid}.npy")
        segment_path = os.path.join(base, "segment", f"{sid}.npy")
        normal_path = os.path.join(base, "normal", f"{sid}.npy")

        coord = np.load(coord_path).astype(np.float32)

        if os.path.isfile(segment_path):
            segment = np.load(segment_path).reshape([-1]).astype(np.int32)
        else:
            segment = np.full(coord.shape[0], -1, dtype=np.int32)

        # optional normal
        normal = (
            np.load(normal_path).astype(np.float32)
            if os.path.isfile(normal_path)
            else None
        )

        # learning map (same pattern as your real datasets)
        segment = np.vectorize(self.learning_map.get, otypes=[np.int32])(
            segment
        ).astype(np.int32)

        data = {
            "coord": coord,
            "segment": segment,
            "instance": np.full(coord.shape[0], -1, dtype=np.int32),
            "name": sid,
            "split": self.get_split_name(idx),
        }
        if normal is not None:
            data["normal"] = normal

        return data

    # ---------------- learning maps ----------------
    @staticmethod
    def get_learning_map(ignore_index):
        return {
            ignore_index: ignore_index,
            0: 0,  # tractor
            1: 1,  # harvester
            2: 2,  # trailer
            3: 3,  # background
        }

    @staticmethod
    def get_learning_map_inv(ignore_index):
        return {
            ignore_index: ignore_index,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }


@DATASETS.register_module()
class LsLidarSynthDataset(_BaseSynthDataset):
    SENSOR = "lslidar"


@DATASETS.register_module()
class OusterSynthDataset(_BaseSynthDataset):
    SENSOR = "ouster"


@DATASETS.register_module()
class s512SynthDataset(_BaseSynthDataset):
    SENSOR = "512"


@DATASETS.register_module()
class s1024SynthDataset(_BaseSynthDataset):
    SENSOR = "1024"


@DATASETS.register_module()
class s2048SynthDataset(_BaseSynthDataset):
    SENSOR = "2048"


@DATASETS.register_module()
class s5096SynthDataset(_BaseSynthDataset):
    SENSOR = "5096"
