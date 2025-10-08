import torch
from PIL import Image
from dust3r.utils.image import depth_edge
import math
import os
from dust3r.utils.misc import move_to_device
import argparse

import numpy as np
from dust3r.utils.vis_utils import write_ply

from dust3r.utils.image import load_images_for_eval as load_images
from layers.pose_enc import pose_encoding_to_extri
from typing import Union

import time
import cv2
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

import trimesh
import matplotlib
from scipy.spatial.transform import Rotation

def predictions_to_glb(
    predictions,
    conf_thres=50.0
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - points: 3D point coordinates (S, H, W, 3)
            - conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        # conf_threshold = np.percentile(conf, conf_thres)
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()            # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()           # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d

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

def postprocess(points, colors):
    clean_mask1 = ~depth_edge(points[::2, 0, ..., 2], rtol=0.03)
    clean_mask2 = ~depth_edge(points[1::2, 0, ..., 2], rtol=0.03)

    clean_points = torch.cat([points[::2, 0, clean_mask1[0]], points[::2, 1, clean_mask2[0]]], dim=1).reshape(-1, 3)
    clean_colors = torch.cat([colors[::2, 0, clean_mask1[0]], colors[::2, 1, clean_mask2[0]]], dim=1).reshape(-1, 3)

    return clean_points.detach().cpu().numpy(), clean_colors.cpu().numpy()

def recover_image(normalized_tensor):
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_tensor = normalized_tensor * std + mean
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    return unnormalized_tensor
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference with WinT3R ")
    
    parser.add_argument("--data_path", type=str, default='examples/001',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='out.glb',
                        help="Path to save the output .glb file.")
    parser.add_argument("--inference_mode", type=str, default='online',
                        help="WinT3R inference mode. online or offline")
    parser.add_argument("--interval", type=int, default=10,
                        help="Interval to sample video. Default: 10 for video")
    parser.add_argument("--ckpt", type=str, default='checkpoints/pytorch_model.bin',
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--conf", type=float, default=50.0,
                        help="GLB confidence (1~100)")
    args = parser.parse_args()

    file = "examples/001"
    dataset = load_images(args.data_path, size=512, verbose=True, crop=True, interval=args.interval)
    dataset = dataset + dataset

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
        ).cuda()

    weights = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(weights, strict=False)

    batch = move_to_device(dataset, args.device)
    imgs = torch.stack([recover_image(view['img'].detach().cpu()) for view in batch], dim=1)

    model.eval()

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            torch.cuda.reset_peak_memory_stats(args.device)
            start_time = time.time()
            pred = model(batch, ret_first_pred=False, mode=args.inference_mode)
            end_time = time.time()
            peak_vram_gb = torch.cuda.max_memory_allocated(args.device) / (1024**3)

    print(f"Model inference finished.")
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print(f"Peak GPU VRAM usage: {peak_vram_gb:.4f} GB")
    colors = imgs.permute(0, 1, 3, 4, 2)
    B, N, H, W, _ = colors.shape 

    extrinsics = pose_encoding_to_extri(pred["camera_pos_enc"][-1])
    R_cam_to_world = extrinsics[:, :, :3, :3]     #  B, S, 3, 3
    t_cam_to_world = extrinsics[:, :, :3, 3]      #  B, S, 3
    world_coords_points = torch.einsum("bsij,bshwj->bshwi", R_cam_to_world, pred['pts_local']) + t_cam_to_world[:, :, None, None]   #B, S, H, W, 3
    
    # 只使用后一半的点云数据
    half_idx = world_coords_points.shape[1] // 2
    pred_pts = world_coords_points[:, half_idx:]
    pred_conf = torch.sigmoid(pred['conf'])[:, half_idx:]
    pred_depth = pred['pts_local'][..., 2:][:, half_idx:]
    colors_half = colors[:, half_idx:]
    
    pred = {"conf": pred_conf, "points": pred_pts, "images": colors_half}
    edge = depth_edge(pred_depth.squeeze(-1), rtol=0.05)
    pred['conf'][edge] = 0.0
    masks_depth = pred_depth[..., 0] >= 500.0
    pred['conf'][masks_depth] = 0.0

    for key in pred.keys():
        if isinstance(pred[key], torch.Tensor):
            pred[key] = pred[key].cpu().numpy().squeeze(0)  # remove batch dimension

    torch.cuda.empty_cache()
    
    print("Building GLB scene...")
    glbscene = predictions_to_glb(
        pred,
        conf_thres=args.conf
    )
    print(f"Saving GLB to: {args.save_path}")
    glbscene.export(file_obj=args.save_path)
