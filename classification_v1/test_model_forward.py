import torch
from model import MInterface

# 只用 standard_net
net = MInterface(
    model="standard_net",   # 关键：只写 standard_net
    in_ch=7,
    num_classes=3,
    resnet_name="resnet18",
    pretrained=False,       # 如需 ImageNet 预训练，改 True（首层仍会替换为7通道）
    freeze=False
)

x = torch.randn(2, 7, 100, 100)
y = net(x)
print("logits shape:", y.shape)  # 期望 [2, 3]
