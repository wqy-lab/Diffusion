# Diffusion 训练脚本设计

## 目标

为 MNIST 数据集编写一个完整的训练脚本，用于调试和练习 diffusion 模型。

## 文件结构

单文件 `train.py`，模块通过 import 引用：

```
train.py       — 训练入口（数据加载 + 训练循环）
models/
  diffusion.py — Diffusion 调度（q_sample, p_sample）
  unet_time.py — 带 time embedding 的 UNet（噪声预测）
config.py      — 超参数
```

## 数据加载

- 数据集：MNIST（`torchvision.datasets.MNIST`）
- 图像 resize 到 28×28（已是 28×28，无需 resize）
- 归一化到 `[0, 1]` 范围
- DataLoader：`batch_size=64`，`shuffle=True`

## 模型

- `UNet` from `models.unet_time`
- 输入：噪声图像 + time step
- 输出：预测噪声

## 训练配置（从 config.py 读取）

| 参数 | 值 |
|------|-----|
| T | 200 |
| batch_size | 64 |
| lr | 1e-3 |
| epochs | 50 |
| device | cuda / cpu |
| time_dim | 256 |
| base_channels | 64 |
| channel_mults | [1, 2, 4, 8] |

## 训练循环

### 进度条
- `tqdm` 包装每个 epoch 的 batch 迭代，显示 epoch、step、loss

### Loss 计算
- MSE loss：`loss = MSELoss(pred_noise, true_noise)`
- 每个 step 打印：`[Epoch X/Y] Step Z: loss=W`

### Checkpoint 保存
- 每 10 个 epoch 保存一次
- 保存内容：`{'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e, 'loss': loss}`
- 路径：`checkpoints/model_epoch_{epoch}.pt`
- 仅保留最新 + 最佳（覆盖式保存，不堆积）

## 损失记录

- 仅 print 输出，格式：`[Epoch {e}/{epochs}] Step {step}: loss={loss:.6f}`
- 不使用 tensorboard / wandb（后续可扩展）

## 实现步骤

1. 在 `train.py` 中 import 所需模块
2. 从 `config.py` 读取超参数（或内联）
3. 编写 `get_data_loader()` 函数
4. 编写 `train()` 函数，包含训练循环
5. 添加 checkpoint 保存逻辑
6. 运行测试