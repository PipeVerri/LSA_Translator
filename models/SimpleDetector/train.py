import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from wandb.integration.lightning.fabric import WandbLogger

from utils.inference import train_test_split_dataset
from utils.lsa64 import LSA64Dataset, generate_collated_dataloader
from pathlib import Path
from .network import LitSimpleSignDetector
import wandb

root_dir = Path(__file__).parent.parent.parent
dataset = LSA64Dataset(root_dir/"data"/"LSA64"/"landmarks")
train_dataset, test_dataset = train_test_split_dataset(dataset)

train_loader = generate_collated_dataloader(train_dataset, batch_size=128)
test_loader = generate_collated_dataloader(test_dataset, shuffle=False)

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    dirpath=Path(__file__).parent,
    filename="hand-detector-{val_acc:.2f}",
    save_last=False # No guardar el modelo del ultimo epoch
)

wandb.init(
    project="SignTranslator",
    group="v2",
)
wandb_logger = WandbLogger(
    save_dir=Path(__file__).parent/"wandb"
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = L.Trainer(
    max_epochs=400,
    callbacks=[RichProgressBar(), checkpoint_callback, lr_monitor],
    logger=wandb_logger
)
model = LitSimpleSignDetector(hidden_width=300)
trainer.fit(model, train_loader, val_dataloaders=test_loader)