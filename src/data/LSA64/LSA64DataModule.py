import lightning as L
from pathlib import Path
from .dataset import LSA64Dataset
from src.data.utils import train_test_split, generate_collated_dataloader

class LSA64DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_size=0.8):
        super().__init__()
        root_dir = Path(__file__).parent.parent.parent
        self.path = root_dir / data_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        dataset = LSA64Dataset(self.path)
        train_ds, test_ds = train_test_split(dataset, train_size=self.train_size)
        self.train_ds = train_ds
        self.val_ds = test_ds

    def train_dataloader(self):
        return generate_collated_dataloader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return generate_collated_dataloader(self.val_ds, batch_size=self.batch_size)