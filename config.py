import torch

dataset = 'cifar10'  # 'mnist' or 'cifar10'

T = 200
beta_start = 1e-4
beta_end = 0.02
batch_size = 128
lr = 1e-3
epochs = 200
early_stopping_patience = 10
early_stopping_min_delta = 1e-5
base_channels = 64
channel_mults = [1, 2, 4, 8]
time_dim = 256
num_groups = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PRESETS = {
    'mnist': {'image_size': 32, 'in_channels': 1, 'out_channels': 1},
    'cifar10': {'image_size': 32, 'in_channels': 3, 'out_channels': 3},
}

_preset = DATASET_PRESETS[dataset]
image_size = _preset['image_size']
in_channels = _preset['in_channels']
out_channels = _preset['out_channels']
checkpoint_base = f'checkpoints/{dataset}'
init_checkpoint = None  # 初始权重路径，None 表示 PyTorch 默认随机初始化