"""
t-SNE Feature Distribution Visualization
Publication-quality visualization following SimCLR/MoCo style
Shows how mask guidance creates well-separated feature clusters
"""

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.utils as vutils
import torch.nn.functional as F


def visualize_feature_tsne(
        matcher,
        batch,
        pred_mask0_bin,
        pred_mask1_bin,
        sample_idx,
        instance_id,
        save_dir="debug_feature_tsne",
        n_samples=2000,
        perplexity=30,
        use_pca_init=True
):
    """
    Publication-quality t-SNE feature distribution visualization

    Args:
        matcher: LoFTR matcher model
        batch: input batch containing images and masks
        pred_mask0_bin: predicted binary mask for image0 [H, W]
        pred_mask1_bin: predicted binary mask for image1 [H, W]
        sample_idx: sample index
        instance_id: instance id from batch
        save_dir: directory to save visualizations
        n_samples: number of points to sample for t-SNE
        perplexity: t-SNE perplexity parameter
        use_pca_init: whether to use PCA for initialization (faster convergence)
    """

    # Create sample directory
    if hasattr(instance_id, 'item'):
        instance_id = instance_id.item()
    sample_dir = os.path.join(save_dir, f"sample_{instance_id}")
    os.makedirs(sample_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"[t-SNE VISUALIZATION] Sample {sample_idx} (instance_id={instance_id})")
    print(f"{'='*80}")

    # Extract images and masks
    img0 = batch['image0'][sample_idx:sample_idx + 1]  # [1, 3, H, W]
    img1 = batch['image1'][sample_idx:sample_idx + 1]
    mask0_gt = batch['mask0_gt'][sample_idx]  # [H, W]
    mask1_gt = batch['mask1_gt'][sample_idx]

    # Convert to numpy for saving
    img0_np = img0.squeeze().permute(1, 2, 0).cpu().numpy()
    img1_np = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img0_np = (img0_np - img0_np.min()) / (img0_np.max() - img0_np.min() + 1e-8) * 255
    img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min() + 1e-8) * 255
    img0_np = img0_np.astype(np.uint8)
    img1_np = img1_np.astype(np.uint8)

    # Convert to grayscale for LoFTR
    def rgb_to_gray(img):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return (0.299 * r + 0.587 * g + 0.114 * b).float()

    img0_gray = rgb_to_gray(img0)
    img1_gray = rgb_to_gray(img1)

    print(f"\n[1] Extracting features from LoFTR backbone...")

    # Extract features from LoFTR
    with torch.no_grad():
        matcher.eval()
        match_batch = {'image0': img0_gray, 'image1': img1_gray}
        matcher(match_batch)

        # Extract coarse-level features (raw CNN features before transformer)
        if 'feat_c0_raw' in match_batch and 'feat_c1_raw' in match_batch:
            feat0 = match_batch['feat_c0_raw']  # [1, C, H', W']
            feat1 = match_batch['feat_c1_raw']  # [1, C, H', W']
            print(f"  ✓ Using raw coarse features from LoFTR backbone")
        else:
            raise RuntimeError(
                "LoFTR features not found in output. "
                "Please ensure LOFTER/src/loftr/loftr.py has been modified to output 'feat_c0_raw' and 'feat_c1_raw'. "
                "See TSNE_FEATURE_VISUALIZATION_GUIDE.md for setup instructions."
            )

    print(f"  Feature shapes: feat0={feat0.shape}, feat1={feat1.shape}")

    # Downsample masks to feature resolution
    _, C, H_feat, W_feat = feat0.shape

    mask0_gt_down = F.interpolate(
        mask0_gt.unsqueeze(0).unsqueeze(0).float(),
        size=(H_feat, W_feat), mode='nearest'
    ).squeeze()

    mask1_gt_down = F.interpolate(
        mask1_gt.unsqueeze(0).unsqueeze(0).float(),
        size=(H_feat, W_feat), mode='nearest'
    ).squeeze()

    mask0_pred_down = F.interpolate(
        pred_mask0_bin.unsqueeze(0).unsqueeze(0).float(),
        size=(H_feat, W_feat), mode='nearest'
    ).squeeze()

    mask1_pred_down = F.interpolate(
        pred_mask1_bin.unsqueeze(0).unsqueeze(0).float(),
        size=(H_feat, W_feat), mode='nearest'
    ).squeeze()

    print(f"\n[2] Preparing feature vectors...")

    # Flatten features and masks
    feat0_flat = feat0.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    feat1_flat = feat1.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()

    mask0_gt_flat = mask0_gt_down.reshape(-1).cpu().numpy() > 0.5
    mask1_gt_flat = mask1_gt_down.reshape(-1).cpu().numpy() > 0.5
    mask0_pred_flat = mask0_pred_down.reshape(-1).cpu().numpy() > 0.5
    mask1_pred_flat = mask1_pred_down.reshape(-1).cpu().numpy() > 0.5

    # Combine features from both images
    all_features = np.vstack([feat0_flat, feat1_flat])
    all_masks_gt = np.concatenate([mask0_gt_flat, mask1_gt_flat])
    all_masks_pred = np.concatenate([mask0_pred_flat, mask1_pred_flat])

    # Create image source labels (for coloring)
    img_source = np.concatenate([
        np.zeros(len(feat0_flat), dtype=int),  # 0 for image0
        np.ones(len(feat1_flat), dtype=int)    # 1 for image1
    ])

    print(f"  Total features: {len(all_features)}")
    print(f"  Feature dimension: {C}")
    print(f"  GT foreground ratio: {all_masks_gt.mean():.2%}")
    print(f"  Pred foreground ratio: {all_masks_pred.mean():.2%}")

    # Sample points for visualization
    n_samples = min(n_samples, len(all_features))
    idx = np.random.choice(len(all_features), n_samples, replace=False)
    features_sampled = all_features[idx]
    masks_gt_sampled = all_masks_gt[idx]
    masks_pred_sampled = all_masks_pred[idx]
    img_source_sampled = img_source[idx]

    print(f"\n[3] Running t-SNE on {n_samples} feature vectors...")
    print(f"  Perplexity: {perplexity}")

    # Optional: PCA initialization for faster convergence
    if use_pca_init and C > 50:
        print(f"  Using PCA initialization (reducing {C}D -> 50D)...")
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features_sampled)
        init_method = 'pca'
    else:
        features_pca = features_sampled
        init_method = 'random'

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init=init_method,
        n_iter=1000,
        verbose=1
    )
    features_2d = tsne.fit_transform(features_pca)

    print(f"  t-SNE completed. Embedding range:")
    print(f"    X: [{features_2d[:, 0].min():.2f}, {features_2d[:, 0].max():.2f}]")
    print(f"    Y: [{features_2d[:, 1].min():.2f}, {features_2d[:, 1].max():.2f}]")

    # ========== Create Publication-Quality Visualizations ==========

    print(f"\n[4] Creating visualizations...")

    # ===== Figure 1: Three-panel comparison (Baseline vs GT vs Pred) =====
    fig1 = plt.figure(figsize=(21, 6))

    # Panel (a): Baseline - all features mixed
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                c='gray', s=8, alpha=0.4, edgecolors='none')
    ax1.set_title("(a) Baseline: All Features\n(No Mask Guidance)",
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.text(0.02, 0.98, f"n={n_samples} points",
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    # Panel (b): GT Mask - foreground cluster highlighted
    ax2 = plt.subplot(1, 3, 2)
    fg_gt = masks_gt_sampled
    bg_gt = ~masks_gt_sampled

    # Plot background first (lighter)
    ax2.scatter(features_2d[bg_gt, 0], features_2d[bg_gt, 1],
                c='#CCCCCC', s=6, alpha=0.3, edgecolors='none',
                label=f'Background ({bg_gt.sum()})')

    # Plot foreground on top (more prominent)
    ax2.scatter(features_2d[fg_gt, 0], features_2d[fg_gt, 1],
                c='#00CED1', s=12, alpha=0.7, edgecolors='darkblue',
                linewidths=0.3, label=f'Foreground ({fg_gt.sum()})')

    ax2.set_title("(b) GT Mask-Guided\n(Oracle Upper Bound)",
                  fontsize=13, fontweight='bold', color='darkblue', pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.2, linestyle='--')

    # Panel (c): Pred Mask - our method
    ax3 = plt.subplot(1, 3, 3)
    fg_pred = masks_pred_sampled
    bg_pred = ~masks_pred_sampled

    ax3.scatter(features_2d[bg_pred, 0], features_2d[bg_pred, 1],
                c='#CCCCCC', s=6, alpha=0.3, edgecolors='none',
                label=f'Background ({bg_pred.sum()})')

    ax3.scatter(features_2d[fg_pred, 0], features_2d[fg_pred, 1],
                c='#32CD32', s=12, alpha=0.7, edgecolors='darkgreen',
                linewidths=0.3, label=f'Foreground ({fg_pred.sum()})')

    ax3.set_title("(c) Pred Mask-Guided (Ours)\n(Learned Foreground Focus)",
                  fontsize=13, fontweight='bold', color='darkgreen', pad=10)
    ax3.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax3.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    fig1.savefig(os.path.join(sample_dir, "feature_tsne_comparison.png"),
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    print(f"  ✓ Saved: feature_tsne_comparison.png")

    # ===== Figure 2: Density-based visualization (SimCLR style) =====
    fig2, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: GT Mask with density
    ax_left = axes[0]

    # Create density plot using hexbin
    fg_gt_pts = features_2d[fg_gt]
    bg_gt_pts = features_2d[bg_gt]

    # Plot background as scatter
    ax_left.scatter(bg_gt_pts[:, 0], bg_gt_pts[:, 1],
                    c='lightgray', s=5, alpha=0.2, edgecolors='none',
                    label='Background', zorder=1)

    # Plot foreground with density coloring
    if len(fg_gt_pts) > 0:
        hb = ax_left.hexbin(fg_gt_pts[:, 0], fg_gt_pts[:, 1],
                            gridsize=30, cmap='Blues', alpha=0.8,
                            mincnt=1, edgecolors='none', zorder=2)
        cb = plt.colorbar(hb, ax=ax_left, label='Foreground Density')

    ax_left.set_title("GT Mask: Foreground Cluster Density",
                      fontsize=13, fontweight='bold', color='darkblue')
    ax_left.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax_left.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax_left.grid(True, alpha=0.2, linestyle='--')

    # Right: Pred Mask with density
    ax_right = axes[1]

    fg_pred_pts = features_2d[fg_pred]
    bg_pred_pts = features_2d[bg_pred]

    ax_right.scatter(bg_pred_pts[:, 0], bg_pred_pts[:, 1],
                     c='lightgray', s=5, alpha=0.2, edgecolors='none',
                     label='Background', zorder=1)

    if len(fg_pred_pts) > 0:
        hb = ax_right.hexbin(fg_pred_pts[:, 0], fg_pred_pts[:, 1],
                             gridsize=30, cmap='Greens', alpha=0.8,
                             mincnt=1, edgecolors='none', zorder=2)
        cb = plt.colorbar(hb, ax=ax_right, label='Foreground Density')

    ax_right.set_title("Pred Mask (Ours): Foreground Cluster Density",
                       fontsize=13, fontweight='bold', color='darkgreen')
    ax_right.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax_right.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax_right.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    fig2.savefig(os.path.join(sample_dir, "feature_tsne_density.png"),
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    print(f"  ✓ Saved: feature_tsne_density.png")

    # ===== Figure 3: Image source coloring (show both images contribute) =====
    fig3 = plt.figure(figsize=(14, 6))

    # Left: colored by image source
    ax_left = plt.subplot(1, 2, 1)

    img0_mask = img_source_sampled == 0
    img1_mask = img_source_sampled == 1

    ax_left.scatter(features_2d[img0_mask, 0], features_2d[img0_mask, 1],
                    c='#FF6B6B', s=8, alpha=0.5, edgecolors='none',
                    label=f'Image 0 ({img0_mask.sum()})')
    ax_left.scatter(features_2d[img1_mask, 0], features_2d[img1_mask, 1],
                    c='#4ECDC4', s=8, alpha=0.5, edgecolors='none',
                    label=f'Image 1 ({img1_mask.sum()})')

    ax_left.set_title("Feature Distribution by Image Source",
                      fontsize=13, fontweight='bold')
    ax_left.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax_left.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax_left.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_left.grid(True, alpha=0.2, linestyle='--')

    # Right: colored by foreground/background (Pred Mask)
    ax_right = plt.subplot(1, 2, 2)

    ax_right.scatter(features_2d[bg_pred, 0], features_2d[bg_pred, 1],
                     c='#CCCCCC', s=6, alpha=0.3, edgecolors='none',
                     label=f'Background ({bg_pred.sum()})')
    ax_right.scatter(features_2d[fg_pred, 0], features_2d[fg_pred, 1],
                     c='#32CD32', s=12, alpha=0.7, edgecolors='darkgreen',
                     linewidths=0.3, label=f'Foreground ({fg_pred.sum()})')

    ax_right.set_title("Feature Distribution by Mask (Ours)",
                       fontsize=13, fontweight='bold', color='darkgreen')
    ax_right.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax_right.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax_right.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_right.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    fig3.savefig(os.path.join(sample_dir, "feature_tsne_source.png"),
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)

    print(f"  ✓ Saved: feature_tsne_source.png")

    # ===== Save original images and masks =====
    Image.fromarray(img0_np).save(os.path.join(sample_dir, "image0.png"))
    Image.fromarray(img1_np).save(os.path.join(sample_dir, "image1.png"))

    # Save masks
    mask0_gt_np = mask0_gt.cpu().numpy()
    mask1_gt_np = mask1_gt.cpu().numpy()
    mask0_pred_np = pred_mask0_bin.cpu().numpy()
    mask1_pred_np = pred_mask1_bin.cpu().numpy()

    Image.fromarray((mask0_gt_np * 255).astype(np.uint8)).save(
        os.path.join(sample_dir, "mask0_gt.png"))
    Image.fromarray((mask1_gt_np * 255).astype(np.uint8)).save(
        os.path.join(sample_dir, "mask1_gt.png"))
    Image.fromarray((mask0_pred_np * 255).astype(np.uint8)).save(
        os.path.join(sample_dir, "mask0_pred.png"))
    Image.fromarray((mask1_pred_np * 255).astype(np.uint8)).save(
        os.path.join(sample_dir, "mask1_pred.png"))

    print(f"\n[5] Summary:")
    print(f"  ✓ All visualizations saved to: {sample_dir}/")
    print(f"  ✓ Files created:")
    print(f"    - feature_tsne_comparison.png (main 3-panel figure)")
    print(f"    - feature_tsne_density.png (density heatmaps)")
    print(f"    - feature_tsne_source.png (image source analysis)")
    print(f"    - image0.png, image1.png (original images)")
    print(f"    - mask0_gt.png, mask1_gt.png (GT masks)")
    print(f"    - mask0_pred.png, mask1_pred.png (Pred masks)")
    print(f"{'='*80}\n")
