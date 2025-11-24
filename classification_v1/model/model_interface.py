# model/model_interface.py
import importlib
import inspect
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MInterface(pl.LightningModule):
    def __init__(self,
                 model: str = "standard_net",
                 in_ch: int = 7, 
                 num_classes: int = 3, 
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 **kwargs):  
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = dict(kwargs)

        self._build_model() 
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=False)

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def _build_model(self):
        camel = "".join(s.capitalize() for s in self.model_name.split("_"))
        try:
            mod = importlib.import_module("." + self.model_name, package=__package__)
            ModelCls = getattr(mod, camel)
        except Exception as e:
            raise ValueError(
                f"Invalid model '{self.model_name}'. Expect model/{self.model_name}.py and class {camel}."
            ) from e

        sig = inspect.signature(ModelCls.__init__)
        valid = {p.name for p in sig.parameters.values() if p.name != "self"}
        args = {k: v for k, v in self.kwargs.items() if k in valid}

        if "in_channel" in valid:
            args.setdefault("in_channel", self.in_ch)
        elif "in_ch" in valid:
            args.setdefault("in_ch", self.in_ch)

        if "out_channel" in valid:
            args.setdefault("out_channel", self.num_classes)
        elif "num_classes" in valid:
            args.setdefault("num_classes", self.num_classes)

        self.net = ModelCls(**args)
