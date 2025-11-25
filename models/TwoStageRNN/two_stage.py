import os.path
from torch.utils.checkpoint import checkpoint
from utils.inference import train_test_split_dataset
from utils.lsa64.dataloader import generate_collated_dataloader
from utils.lsa64.dataset import LSA64Dataset
import torch
from torchmetrics.classification import BinaryAccuracy
from pathlib import Path
import lightning as L
from models import SimpleRNN
import torch.nn as nn
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint

root_path = Path(__file__).parent.parent.parent
dataset = LSA64Dataset(root_path/"data"/"LSA64"/"landmarks", hand_label=True)
train_ds, test_ds = train_test_split_dataset(dataset)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using devic: {device}")

train_loader = generate_collated_dataloader(train_ds, 1)
test_loader = generate_collated_dataloader(test_ds, 1, shuffle=False)

class LitHandDetector(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleRNN(input_dim=144, hidden_dim=144, hidden_layers=3, output_dim=1)
        self.train_accuracy = BinaryAccuracy()
        self.validation_accuracy = BinaryAccuracy()

    def training_step(self, batch, batch_idx):
        x, lengths, y = batch
        y_pred = self.model(x, lengths)
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # Compute accuracy
        self.train_accuracy.update(y_pred, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, y = batch
        y_pred = self.model(x, lengths)
        self.validation_accuracy.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.validation_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.validation_accuracy.reset()

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    dirpath=Path(__file__).parent,
    filename="hand_detector",
    save_last=False # No guardar el ultimo modelo
)

trainer = L.Trainer(max_epochs=25, callbacks=[RichProgressBar(), checkpoint_callback], logger=False)
model = LitHandDetector()
trainer.fit(model, train_loader, val_dataloaders=test_loader)