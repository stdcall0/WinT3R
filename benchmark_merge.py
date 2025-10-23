"""
Visual comparison of Token Merging impact
Compares baseline vs merged reconstruction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dust3r.wint3r import WinT3R
from dust3r.utils.image import load_images_for_eval as load_images
from dust3r.utils.misc import move_to_device
from layers.pose_enc import pose_encoding_to_extri
import argparse
import time

def run_inference(model, batch, device, inference_mode):
    """Run inference and return timing + predictions"""
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            torch.cuda.reset_peak_memory_stats(device)
            start_time = time.time()

            pred = model(batch, ret_first_pred=False, mode=inference_mode)

            torch.cuda.synchronize()
            end_time = time.time()
            
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
    
    return pred, end_time - start_time, peak_vram

def compare_models(data_path, ckpt_path, merge_ratio=0.6, device='cuda', inference_mode='online'):
    """Compare baseline and merged models"""
    
    print("Loading images...")
    dataset = load_images(data_path, size=512, verbose=False, crop=True)
    batch = move_to_device(dataset, device)
    
    # Create baseline model
    print("\n" + "="*60)
    print("BASELINE MODEL (no token merging)")
    print("="*60)
    model_baseline = WinT3R(
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
        #enable_token_merge=False,  # Baseline
    ).to(device)
    
    weights = torch.load(ckpt_path, weights_only=False, map_location=device)
    model_baseline.load_state_dict(weights, strict=False)
    model_baseline.eval()
    
    pred_baseline, time_baseline, vram_baseline = run_inference(model_baseline, batch, device, inference_mode)
    print(f"Time: {time_baseline:.3f}s")
    print(f"VRAM: {vram_baseline:.3f}GB")
    
    del model_baseline
    torch.cuda.empty_cache()
    
    # Create merged model
    print("\n" + "="*60)
    print(f"MERGED MODEL (token merging at {merge_ratio})")
    print("="*60)
    model_merged = WinT3R(
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
        merge_tokens=True,   # Enable merging
        merging_ratio=merge_ratio,
    ).to(device)
    
    model_merged.load_state_dict(weights, strict=False)
    model_merged.eval()  # Use eval mode like baseline
    
    pred_merged, time_merged, vram_merged = run_inference(model_merged, batch, device, inference_mode)
    print(f"Time: {time_merged:.3f}s")
    print(f"VRAM: {vram_merged:.3f}GB")
    
    # Calculate improvements
    speedup = time_baseline / time_merged
    vram_reduction = (1 - vram_merged / vram_baseline) * 100
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<15} {'Merged':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Time (s)':<20} {time_baseline:<15.3f} {time_merged:<15.3f} {speedup:.2f}x")
    print(f"{'VRAM (GB)':<20} {vram_baseline:<15.3f} {vram_merged:<15.3f} {vram_reduction:.1f}% ↓")
    
    # Extract point clouds
    extrinsics_baseline = pose_encoding_to_extri(pred_baseline["camera_pos_enc"][-1])
    R_baseline = extrinsics_baseline[:, :, :3, :3]
    t_baseline = extrinsics_baseline[:, :, :3, 3]
    pts_baseline = torch.einsum("bsij,bshwj->bshwi", R_baseline, pred_baseline['pts_local']) + t_baseline[:, :, None, None]
    
    extrinsics_merged = pose_encoding_to_extri(pred_merged["camera_pos_enc"][-1])
    R_merged = extrinsics_merged[:, :, :3, :3]
    t_merged = extrinsics_merged[:, :, :3, 3]
    pts_merged = torch.einsum("bsij,bshwj->bshwi", R_merged, pred_merged['pts_local']) + t_merged[:, :, None, None]
    
    # Compute differences
    pts_diff = torch.abs(pts_baseline - pts_merged).mean()
    conf_baseline = torch.sigmoid(pred_baseline['conf']).mean()
    conf_merged = torch.sigmoid(pred_merged['conf']).mean()
    
    print(f"{'Avg Point Diff':<20} {'-':<15} {'-':<15} {pts_diff.item():.6f}")
    print(f"{'Avg Confidence':<20} {conf_baseline.item():<15.4f} {conf_merged.item():<15.4f} {(conf_merged/conf_baseline).item():.3f}x")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Baseline vs Token Merging (ratio={merge_ratio})', fontsize=16)
    
    # Show depth maps
    for i in range(3):
        if i < pred_baseline['pts_local'].shape[1]:
            depth_baseline = pred_baseline['pts_local'][0, i, :, :, 2].cpu().numpy()
            depth_merged = pred_merged['pts_local'][0, i, :, :, 2].cpu().numpy()
            
            axes[0, i].imshow(depth_baseline, cmap='viridis')
            axes[0, i].set_title(f'Baseline - Frame {i}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(depth_merged, cmap='viridis')
            axes[1, i].set_title(f'Merged - Frame {i}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: comparison_visualization.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if speedup > 1.3:
        print(f"✅ Good speedup: {speedup:.2f}x faster!")
    else:
        print(f"⚠️  Modest speedup: {speedup:.2f}x")
    
    if vram_reduction > 20:
        print(f"✅ Good memory reduction: {vram_reduction:.1f}%")
    else:
        print(f"⚠️  Small memory reduction: {vram_reduction:.1f}%")
    
    if pts_diff.item() < 0.01:
        print(f"✅ Minimal quality impact: diff={pts_diff.item():.6f}")
    elif pts_diff.item() < 0.05:
        print(f"⚠️  Moderate quality impact: diff={pts_diff.item():.6f}")
    else:
        print(f"❌ Significant quality impact: diff={pts_diff.item():.6f}")
    
    return {
        'speedup': speedup,
        'vram_reduction': vram_reduction,
        'point_diff': pts_diff.item(),
        'conf_baseline': conf_baseline.item(),
        'conf_merged': conf_merged.item(),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='examples/002')
    parser.add_argument("--ckpt", type=str, default='checkpoints/pytorch_model.bin')
    parser.add_argument("--merge_ratio", type=float, default=0.8)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--inference_mode", type=str, default='online')
    args = parser.parse_args()
    
    results = compare_models(
        args.data_path,
        args.ckpt,
        args.merge_ratio,
        args.device,
        args.inference_mode
    )
    
    print(results)
    
    print("\n🎉 Comparison complete!")