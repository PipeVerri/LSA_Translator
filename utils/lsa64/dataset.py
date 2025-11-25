import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class LSA64Dataset(Dataset):
    def __init__(self, root, hand_label=False):
        self.root = root
        self.hand_label = hand_label
        self.files = [f for f in os.listdir(root) if f.endswith(".npy")]
        if hand_label:
            self.meta = pd.read_csv(os.path.join(root, "../meta.csv"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        label = int(filename.split("_")[0]) - 1  # -1 para que los labels empiezen en 0
        hand_label = self.meta.iloc[label]["H"]
        hand_label = 0 if hand_label == "R" else 1

        arr = np.load(os.path.join(self.root, filename))

        if arr.shape[0] == 0:
            raise ValueError(f"Empty array at {filename}")
        if arr.shape[1] != 144:
            raise ValueError(f"Wrong array shape at {filename}: {arr.shape}")

        tensor = torch.from_numpy(arr).float() # Numpy trabaja con float64 pero torch con float32
        if self.hand_label:
            return tensor, (label, hand_label)
        else:
            return tensor, label