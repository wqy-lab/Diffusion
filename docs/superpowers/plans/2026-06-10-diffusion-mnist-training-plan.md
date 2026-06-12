# Diffusion MNIST 训练脚本实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写一个完整的 MNIST 训练脚本，包含数据加载、训练循环、checkpoint 保存。

**Architecture:** 单文件 `train.py`，通过 import 引用 `models.diffusion.Diffusion` 和 `models.unet_time.UNet`。训练循环包含 tqdm 进度条、loss 打印、每 10 epoch 保存 checkpoint。

**Tech Stack:** PyTorch, torchvision, tqdm

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `train.py` | 训练入口：数据加载 + 训练循环 + checkpoint |
| `models/diffusion.py` | Diffusion 调度（已有） |
| `models/unet_time.py` | UNet 模型（已有） |
| `config.py` | 超参数（已有） |

---

## 实现步骤

### Task 1: 完成 train.py 框架

**Files:**
- Modify: `train.py:1-3`

- [ ] **Step 1: 编写完整的 train.py**

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.unet_time import UNet
from models.diffusion import Diffusion
import config


def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 归一化到 [0, 1]
    ])
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return loader


def train():
    device = config.device

    # 数据
    train_loader = get_data_loader()

    # 模型
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_groups=config.num_groups,
        time_dim=config.time_dim
    ).to(device)

    diffusion = Diffusion(T=config.T, beta_start=config.beta_start, beta_end=config.beta_end, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for step, (images, _) in enumerate(pbar):
            images = images.to(device)

            # 采样 t
            t = torch.randint(0, config.T, (images.size(0),), device=device)

            # 加噪
            noisy_images, noise = diffusion.q_sample(images, t, torch.randn_like(images))

            # 预测噪声
            pred_noise = model(noisy_images, t)

            # loss
            loss = nn.MSELoss()(pred_noise, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # 每个 epoch 结束后打印
        avg_loss = loss.item()
        print(f'[Epoch {epoch+1}/{config.epochs}] Step {step+1}: loss={avg_loss:.6f}')

        # Checkpoint 保存（每 10 epoch + 最佳）
        if (epoch + 1) % 10 == 0:
            ckpt_path = f'checkpoints/model_epoch_{epoch+1}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, ckpt_path)
            print(f'Checkpoint saved: {ckpt_path}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }, 'checkpoints/best_model.pt')
            print(f'Best model updated: loss={best_loss:.6f}')

    print('Training complete!')


if __name__ == '__main__':
    train()
```

- [ ] **Step 2: 运行测试**

命令：`python train.py`
预期：能启动训练，显示 tqdm 进度条和 loss 值。无报错。

---

## 验证清单

- [ ] MNIST 数据集能正常下载/加载
- [ ] tqdm 进度条显示 epoch、loss
- [ ] 每个 epoch 打印 loss
- [ ] 每 10 epoch 保存 checkpoint 到 `checkpoints/`
- [ ] 最佳模型保存到 `checkpoints/best_model.pt`
- [ ] 训练能在 CPU 或 CUDA 上正常运行