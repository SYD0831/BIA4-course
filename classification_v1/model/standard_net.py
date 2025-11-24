# model/standard_net.py
# 7-channel CNN for MP-IDB (falciparum / vivax / ovale)
# 架构：4 个卷积块 + 2 个全连接层，配合 7 通道输入，用于 3 分类。

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy


class MPIDBCNN(nn.Module):
    """
    7 通道 CNN:

    输入: (B, 7, 100, 100)
      7 通道 = [R, G, B, L(Lab), S(HSV), Laplacian, TopHat]
      -> 通过 data/mpidb_dataset.py 中的 to7ch() 构造

    结构:
      Block1: Conv(7->32) + BN + LeakyReLU + MaxPool + Dropout
      Block2: Conv(32->64)  + BN + LeakyReLU + MaxPool + Dropout
      Block3: Conv(64->128) + BN + LeakyReLU + MaxPool + Dropout
      Block4: Conv(128->256)+ BN + LeakyReLU + MaxPool + Dropout
      展平 -> FC1(256*6*6 -> 256) + LeakyReLU
           -> FC2(256 -> num_classes)
    """
    def __init__(self, in_ch: int = 7, num_classes: int = 3, p: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.b1 = nn.Sequential(
            block(in_ch, 32),
            nn.MaxPool2d(2),
            nn.Dropout(p),
        )
        self.b2 = nn.Sequential(
            block(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout(p),
        )
        self.b3 = nn.Sequential(
            block(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout(p),
        )
        self.b4 = nn.Sequential(
            block(128, 256),
            nn.MaxPool2d(2),
            nn.Dropout(p),
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = x.flatten(1)
        x = F.leaky_relu(self.fc1(x), 0.1, inplace=True)
        x = self.fc2(x)
        return x


class MPIDBCNNLightning(pl.LightningModule):

    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MPIDBCNN(
            in_ch=in_channels,
            num_classes=num_classes,
            p=dropout,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        if stage == "train":
            acc = self.train_acc(preds, y)
            self.log("train_loss", loss,
                     on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)
            self.log("train_acc", acc,
                     on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)

        elif stage == "val":
            acc = self.val_acc(preds, y)
            self.log("val_loss", loss,
                     on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
            self.log("val_acc", acc,
                     on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)

        elif stage == "test":
            acc = self.test_acc(preds, y)
            self.log("test_loss", loss,
                     on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
            self.log("test_acc", acc,
                     on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
            },
        }


StandardNet = MPIDBCNN
StandardNetLightning = MPIDBCNNLightning
