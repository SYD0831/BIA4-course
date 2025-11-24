# test_mpidb_dataloader.py
import torch
from data import DInterface

def main():
    dm = DInterface(
        num_workers=2,                 # 这里你可以保持 2 或 4，都行
        dataset="mpidb_dataset",       # 对应 data/mpidb_dataset.py 里的 MpidbDataset
        batch_size=8,
        root="data/MPIDB",             # 你的 train/val/test 根目录
        classes=["falciparum", "vivax", "ovale"],
        img_size=100,
        aug=True,                      # 训练集会用增强，val/test 自动关闭增强
    )

    print("=== SETUP(FIT) ===")
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    xb, yb = next(iter(train_loader))
    print("train batch x:", xb.shape, xb.dtype, xb.device)
    print("train batch y:", yb.shape, yb.dtype)
    print("train labels sample:", yb[:10])

    xbv, ybv = next(iter(val_loader))
    print("val batch x:", xbv.shape, xbv.dtype, xbv.device)
    print("val batch y:", ybv.shape, ybv.dtype)
    print("val labels sample:", ybv[:10])

    print("\n=== SETUP(TEST) ===")
    dm.setup("test")
    test_loader = dm.test_dataloader()

    xbt, ybt = next(iter(test_loader))
    print("test batch x:", xbt.shape, xbt.dtype, xbt.device)
    print("test batch y:", ybt.shape, ybt.dtype)
    print("test labels sample:", ybt[:10])

if __name__ == "__main__":
    main()
