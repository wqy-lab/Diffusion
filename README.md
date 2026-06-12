# Diffusion MNIST

## 环境配置

```bash
pip install -r requirements.txt
```

## 运行训练

```bash
python train.py
```

## 文件结构

```
├── train.py              # 训练入口
├── config.py             # 超参数
├── models/
│   ├── diffusion.py     # Diffusion 调度
│   └── unet_time.py      # 带 time embedding 的 UNet
├── checkpoints/          # 模型保存目录
└── requirements.txt
```