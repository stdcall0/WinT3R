"""
WinT3R Inference Script with FP16/BF16 Support for SageAttention

This script demonstrates the BEST PRACTICE for using SageAttention:
- Convert the entire model to FP16/BF16 at initialization
- No runtime dtype conversion overhead
- Maximum performance and minimal memory usage
"""

import torch
from PIL import Image
from dust3r.utils.image import depth_edge
import math
import os
from dust3r.utils.misc import move_to_device
import argparse
import time

import numpy as np
from dust3r.utils.vis_utils import write_ply

from dust3r.utils.image import load_images_for_eval as load_images
from layers.pose_enc import pose_encoding_to_extri
from typing import Union

def save_image_grid(images: np.ndarray, grid_shape: tuple, save_path: str):
    """
    images: numpy array of shape (N, H, W, 3)
    grid_shape: (rows, cols)
    """
    H, W = images.shape[1], images.shape[2]
    grid = np.zeros((grid_shape[0]*H, grid_shape[1]*W, 3), dtype=np.uint8)
    
    for i in range(min(len(images), grid_shape[0]*grid_shape[1])):
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        grid[row*H:(row+1)*H, col*W:(col+1)*W] = images[i]
    
    Image.fromarray(grid).save(save_path)

def save_image_grid_auto(images: Union[np.ndarray, torch.Tensor], save_path: str):
    num_images = images.shape[0]
    """
    images: np.ndarray of shape (N, H, W, 3) in [0, 255] or torch.Tensor of shape (N, 3, H, W) in range [0, 1]
    """
    if isinstance(images, torch.Tensor):
        assert images.ndim == 4 and (images.shape[1] == 3 or images.shape[-1] == 3), f"images must be a 4D torch tensor with shape (N, 3, H, W) or (N, H, W, 3)"
        if images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
        images = (images.detach().cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(images, np.ndarray):
        assert images.ndim == 4 and images.shape[3] == 3, f"images must be a 4D numpy array with shape (N, H, W, 3)"
    else:
        raise ValueError(f"images must be a numpy array or a torch tensor, but got {type(images)}")

    cols = math.floor(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    save_image_grid(images, (rows, cols), save_path)

def recover_image(normalized_tensor):
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_tensor = normalized_tensor * std + mean
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    return unnormalized_tensor


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference with WinT3R in FP16/BF16 mode")
    
    parser.add_argument("--data_path", type=str, default='examples/001',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="Path to save the output .ply file.")
    parser.add_argument("--save_name", type=str, default='recon.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--inference_mode", type=str, default='online',
                        help="WinT3R inference mode. online or offline")
    parser.add_argument("--interval", type=int, default=10,
                        help="Interval to sample video. Default: 10 for video")
    parser.add_argument("--ckpt", type=str, default='/home/featurize/work/data/pretrained_models/wint3r/pytorch_model.bin',
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--dtype", type=str, default='auto', choices=['fp32', 'fp16', 'bf16', 'auto'],
                        help="Model precision. 'auto' will use bf16 if supported, otherwise fp16. Default: 'auto'")
    args = parser.parse_args()

    # Determine the dtype to use
    if args.dtype == 'auto':
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
            dtype_name = "BFloat16"
        else:
            dtype = torch.float16
            dtype_name = "Float16"
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
        dtype_name = "BFloat16"
    elif args.dtype == 'fp16':
        dtype = torch.float16
        dtype_name = "Float16"
    else:  # fp32
        dtype = torch.float32
        dtype_name = "Float32"
    
    print(f"\n{'='*60}")
    print(f"Running WinT3R inference in {dtype_name} precision")
    print(f"{'='*60}\n")

    # Load dataset
    print(f"Loading images from: {args.data_path}")
    dataset = load_images(args.data_path, size=512, verbose=True, crop=True, interval=args.interval)

    # Initialize model
    print(f"Initializing WinT3R model...")
    from dust3r.wint3r import WinT3R
    model = WinT3R(
            state_size=1024,
            state_pe="2d",
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=[512, 512],
            head_type="conv",
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            landscape_only=False,
            global_merging=1
        ).cuda()

    # Load weights
    print(f"Loading checkpoint from: {args.ckpt}")
    weights = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(weights, strict=False)
    
    # ====================================================================
    # BEST PRACTICE: Convert model to FP16/BF16 ONCE at initialization
    # This eliminates runtime conversion overhead and maximizes performance
    # ====================================================================
    if dtype != torch.float32:
        print(f"Converting model to {dtype_name}...")
        model = model.to(dtype=dtype)
        print(f"✓ Model converted to {dtype_name}")
    
    model.eval()

    # Prepare batch
    batch = move_to_device(dataset, args.device)
    imgs = torch.stack([recover_image(view['img'].detach().cpu()) for view in batch], dim=1)
    
    # Convert batch to the same dtype as model
    if dtype != torch.float32:
        for view in batch:
            view['img'] = view['img'].to(dtype=dtype)

    # Run inference
    print(f"\nRunning inference...")
    torch.cuda.reset_peak_memory_stats(args.device)
    start_time = time.time()
    
    with torch.no_grad():
        pred = model(batch, ret_first_pred=False, mode=args.inference_mode)
    
    torch.cuda.synchronize()
    end_time = time.time()
    peak_vram_gb = torch.cuda.max_memory_allocated(args.device) / (1024**3)
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print(f"Peak GPU VRAM usage: {peak_vram_gb:.4f} GB")
    print(f"{'='*60}\n")

    # Post-processing
    colors = imgs.permute(0, 1, 3, 4, 2)
    B, N, H, W, _ = colors.shape 

    extrinsics = pose_encoding_to_extri(pred["camera_pos_enc"][-1])
    R_cam_to_world = extrinsics[:, :, :3, :3]     #  B, S, 3, 3
    t_cam_to_world = extrinsics[:, :, :3, 3]      #  B, S, 3
    world_coords_points = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, pred['pts_local']) + t_cam_to_world[:, :, None, None]   #B, S, H, W, 3
    pred_pts = world_coords_points
    pred_conf = pred['conf']
    pred_depth = pred['pts_local'][..., 2:]

    pred = {"conf": pred_conf, "points": pred_pts, "depth":pred_depth}

    masks_depth = pred['depth'][..., 0]  < 500.0
    masks_edge = ~depth_edge(pred_depth.squeeze(-1), rtol=0.05)
    masks = masks_depth*masks_edge

    # Save results
    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(args.save_dir, exist_ok=True)
    write_ply(pred['points'][masks].detach().cpu().numpy().reshape(-1, 3), colors[masks.detach().cpu()].reshape(-1, 3).detach().cpu().numpy(), save_path)
    save_image_grid_auto(colors[0].permute(0,3,1,2), save_path.replace('ply', 'png'))
    print(f'Results saved to: {save_path}')
    print(f'Image grid saved to: {save_path.replace("ply", "png")}')
