import argparse
import os

import torch
from torchvision.utils import save_image

from models.diffusion import Diffusion
from models.unet_time import UNet
from utils.run_dir import resolve_run_dir
import config


def load_model(ckpt_path, device):
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_groups=config.num_groups,
        time_dim=config.time_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def sample(num_images=16, ckpt_path=None, output_path=None, run_name=None):
    device = config.device
    print(f'Using device: {device}')

    if ckpt_path is None:
        run_dir = resolve_run_dir(run_name)
        ckpt_path = os.path.join(run_dir, 'best_model.pt')
        if output_path is None:
            output_path = os.path.join(run_dir, 'samples.png')
        print(f'Using run: {run_dir}')
    elif output_path is None:
        output_path = f'samples_{config.dataset}.png'

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model, ckpt = load_model(ckpt_path, device)
    diffusion = Diffusion(
        T=config.T,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )

    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}, loss={ckpt.get("loss", "?")}')
    print(f'Generating {num_images} images...')

    images = diffusion.sample(
        model,
        batch_size=num_images,
        image_size=config.image_size,
        channels=config.in_channels,
    )
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_image(images, output_path, nrow=int(num_images ** 0.5))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample images from a trained diffusion model')
    parser.add_argument('-n', '--num-images', type=int, default=16, help='number of images to generate')
    parser.add_argument('-r', '--run', type=str, default='latest', help='training run folder name or latest')
    parser.add_argument('-c', '--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('-o', '--output', type=str, default=None, help='output image path')
    args = parser.parse_args()
    sample(
        num_images=args.num_images,
        ckpt_path=args.ckpt,
        output_path=args.output,
        run_name=args.run,
    )
