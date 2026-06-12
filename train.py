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
        transforms.Resize(32),  # 调整到 32x32，和 UNet 下采样对称
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