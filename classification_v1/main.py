# main.py
import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from data import DInterface
from model import MInterface

seed_everything(42)

dm = DInterface(
    num_workers=4,
    dataset='mpidb_dataset',
    batch_size=32,
    root='data/MPIDB',
    classes=['falciparum', 'vivax', 'ovale'],
    img_size=100,
    aug=True,
)

model = MInterface(
    model="standard_net",
    in_ch=7,
    num_classes=3,
    lr=1e-3,
    weight_decay=1e-4,
    resnet_name="resnet18",
    pretrained=False,
    freeze=False,
)

ckpt = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="best"
)
es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=8
)

trainer = Trainer(
    max_epochs=30,
    precision="16-mixed",
    accelerator="auto",
    callbacks=[ckpt, es],
)

trainer.fit(model, datamodule=dm)


