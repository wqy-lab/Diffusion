import torch

T = 200
beta_start = 1e-4
beta_end = 0.02
batch_size = 64
lr = 1e-3
epochs = 50
image_size = 32
in_channels = 1
out_channels = 1
base_channels = 64
channel_mults = [1, 2, 4, 8]
time_dim = 256
num_groups = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'