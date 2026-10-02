from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


MFEAT_DIMS = [216, 76, 64, 6, 240, 47]
MFEAT_VIEWS = 6
MFEAT_SIZE = 2000
MFEAT_CLASSES = 10


class MfeatDataset(Dataset):
    def __init__(self, mat_path):
        mat_path = Path(mat_path)
        if not mat_path.is_file():
            raise FileNotFoundError(
                f"Mfeat data file not found: {mat_path}. "
                "Place mfeat.mat in the data directory."
            )
        content = scipy.io.loadmat(mat_path)
        missing = [f"X{index}" for index in range(1, 7) if f"X{index}" not in content]
        if missing:
            raise KeyError(f"{mat_path} is missing arrays: {missing}")

        raw_views = [content[f"X{index}"].astype(np.float32) for index in range(1, 7)]
        if [view.shape[1] for view in raw_views] != MFEAT_DIMS:
            raise ValueError(
                f"Unexpected Mfeat view dimensions: {[view.shape for view in raw_views]}"
            )
        if any(view.shape[0] != MFEAT_SIZE for view in raw_views):
            raise ValueError(f"Expected {MFEAT_SIZE} samples in every Mfeat view")

        self.views = [
            (view - np.mean(view, axis=0)) / np.std(view, axis=0)
            for view in raw_views
        ]
        if any(not np.isfinite(view).all() for view in self.views):
            raise ValueError("Mfeat standardization produced NaN/Inf values")

        self.labels = np.repeat(np.arange(MFEAT_CLASSES), 200)

    def __len__(self):
        return MFEAT_SIZE

    def __getitem__(self, index):
        return (
            [view[index] for view in self.views],
            self.labels[index],
            torch.tensor(index, dtype=torch.long),
        )


def load_mfeat(data_root):
    dataset = MfeatDataset(Path(data_root) / "mfeat.mat")
    return dataset, MFEAT_DIMS, MFEAT_VIEWS, MFEAT_SIZE, MFEAT_CLASSES
