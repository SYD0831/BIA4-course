import importlib
import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DInterface(pl.LightningDataModule):
    """
    通用 DataModule：根据 dataset 名（文件名 snake_case）动态加载对应类（CamelCase），
    并用 split=train/val/test 实例化三套 Dataset。
    """
    def __init__(
        self,
        num_workers: int = 8,
        dataset: str = "",          # "mpidb_dataset"
        batch_size: int = 64,
        pin_memory: bool = True,
        **kwargs                   
    ):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = dict(kwargs)
        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self._load_data_module()

        self.save_hyperparameters(ignore=["kwargs"])
        self.trainset = None
        self.valset = None
        self.testset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.trainset = self._instancialize(split="train")
            self.valset   = self._instancialize(split="val")
        if stage == "test" or stage is None:
            self.testset  = self._instancialize(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )

    def _load_data_module(self):

        if not self.dataset:
            raise ValueError("Please set dataset='mpidb_dataset' (or your dataset file name).")
        camel_name = "".join([s.capitalize() for s in self.dataset.split("_")])
        try:
            module = importlib.import_module("." + self.dataset, package=__package__)
            self.data_cls = getattr(module, camel_name)
        except Exception as e:
            raise ValueError(
                f"Invalid dataset '{self.dataset}'. Expect file data/{self.dataset}.py "
                f"and class {camel_name} in it."
            ) from e

    def _instancialize(self, **override_args):

        sig = inspect.signature(self.data_cls.__init__)
        valid_params = set(p.name for p in sig.parameters.values() if p.name != "self")
        args = {k: v for k, v in self.kwargs.items() if k in valid_params}
        args.update(override_args)
        return self.data_cls(**args)
