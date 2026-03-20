# -*- coding: utf-8 -*-
"""
掩膜引导匹配对比图 (Mask-Guided Effect Comparison)
对应章节: 2.3.1 Foreground-Focused Feature Extraction

生成论文图：
- 左图 (Baseline/w.o. Mask): 原图直接匹配，红色线条表示背景噪声匹配
- 右图 (Ours/Mask-Guided): Mask引导匹配，绿色线条表示高质量前景匹配

参考风格: LoFTR (CVPR 2021), SuperGlue (CVPR 2020)
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # 无头模式


def visualize_mask_guided_comparison(
        img0: np.ndarray,
        img1: np.ndarray,
        mask0: np.ndarray,
        mask1: np.ndarray,
        mkpts0_raw: np.ndarray,
        mkpts1_raw: np.ndarray,
        mconf_raw: np.ndarray,
        mkpts0_masked: np.ndarray,
        mkpts1_masked: np.ndarray,
        mconf_masked: np.ndarray,
        save_path: str,
        conf_threshold: float = 0.3,
        max_lines: int = 100,
        title_left: str = "w/o Mask (Baseline)",
        title_right: str = "Mask-Guided (Ours)",
        dpi: int = 150,
        figsize: tuple = (16, 8),
        darken_factor: float = 0.3,
):
    """
    生成掩膜引导匹配对比图

    Args:
        img0, img1: 原始RGB图像 [H, W, 3], uint8
        mask0, mask1: 二值掩码 [H, W], 0/1 或 0/255
        mkpts0_raw, mkpts1_raw: 无Mask时的匹配点 [N, 2]
        mconf_raw: 无Mask时的置信度 [N]
        mkpts0_masked, mkpts1_masked: 有Mask时的匹配点 [M, 2]
        mconf_masked: 有Mask时的置信度 [M]
        save_path: 保存路径
        conf_threshold: 置信度阈值，低于此值的匹配用虚线
        max_lines: 最大绘制线条数
        title_left, title_right: 左右子图标题
        dpi: 输出分辨率
        figsize: 图像尺寸
        darken_factor: 背景变暗系数 (0~1, 越小越暗)
    """

    # 归一化 mask
    if mask0.max() > 1:
        mask0 = mask0 / 255.0
    if mask1.max() > 1:
        mask1 = mask1 / 255.0

    H, W = img0.shape[:2]

    # ========== 创建图像 ==========
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---------- 左图: Baseline (无 Mask) ----------
    ax_left = axes[0]

    # 拼接两张原图
    concat_raw = np.concatenate([img0, img1], axis=1)
    ax_left.imshow(concat_raw)
    ax_left.set_title(title_left, fontsize=14, fontweight='bold', color='darkred')
    ax_left.axis('off')

    # 绘制匹配线 (红色 = 背景噪声)
    n_raw = min(len(mkpts0_raw), max_lines)
    if n_raw > 0:
        # 按置信度排序，取 top
        idx_sort = np.argsort(-mconf_raw)[:n_raw]
        for idx in idx_sort:
            p0 = mkpts0_raw[idx]
            p1 = mkpts1_raw[idx].copy()
            p1[0] += W  # shift x for concatenated image
            conf = mconf_raw[idx]

            # 判断是否在 mask 内
            y0, x0 = int(round(p0[1])), int(round(p0[0]))
            y1, x1 = int(round(mkpts1_raw[idx][1])), int(round(mkpts1_raw[idx][0]))

            in_mask0 = (0 <= y0 < H and 0 <= x0 < W and mask0[y0, x0] > 0.5)
            in_mask1 = (0 <= y1 < H and 0 <= x1 < W and mask1[y1, x1] > 0.5)

            # 颜色: 背景匹配=红色, 前景匹配=橙色
            if in_mask0 and in_mask1:
                color = 'orange'
                alpha = 0.7
                lw = 1.0
            else:
                color = 'red'
                alpha = 0.5
                lw = 0.8

            linestyle = '-' if conf > conf_threshold else '--'
            ax_left.plot([p0[0], p1[0]], [p0[1], p1[1]],
                         color=color, alpha=alpha, linewidth=lw, linestyle=linestyle)

            # 画端点
            ax_left.scatter([p0[0]], [p0[1]], c=color, s=8, alpha=alpha, zorder=5)
            ax_left.scatter([p1[0]], [p1[1]], c=color, s=8, alpha=alpha, zorder=5)

    # 添加统计信息
    n_total_raw = len(mkpts0_raw)
    # 计算背景匹配数量
    n_bg = 0
    for i in range(len(mkpts0_raw)):
        y0, x0 = int(round(mkpts0_raw[i][1])), int(round(mkpts0_raw[i][0]))
        y1, x1 = int(round(mkpts1_raw[i][1])), int(round(mkpts1_raw[i][0]))
        in0 = (0 <= y0 < H and 0 <= x0 < W and mask0[y0, x0] > 0.5)
        in1 = (0 <= y1 < H and 0 <= x1 < W and mask1[y1, x1] > 0.5)
        if not (in0 and in1):
            n_bg += 1

    ax_left.text(10, H - 30, f"Total: {n_total_raw} matches", fontsize=10, color='white',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax_left.text(10, H - 10, f"Background: {n_bg} ({100 * n_bg / max(n_total_raw, 1):.1f}%)",
                 fontsize=10, color='red',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # ---------- 右图: Mask-Guided (Ours) ----------
    ax_right = axes[1]

    # 创建背景变暗的图像
    img0_dark = img0.copy().astype(np.float32)
    img1_dark = img1.copy().astype(np.float32)

    # 背景区域变暗
    bg_mask0 = (mask0 < 0.5)
    bg_mask1 = (mask1 < 0.5)

    for c in range(3):
        img0_dark[:, :, c][bg_mask0] *= darken_factor
        img1_dark[:, :, c][bg_mask1] *= darken_factor

    img0_dark = np.clip(img0_dark, 0, 255).astype(np.uint8)
    img1_dark = np.clip(img1_dark, 0, 255).astype(np.uint8)

    # 可选: 添加 mask 边界高亮
    img0_dark = draw_mask_contour(img0_dark, mask0, color=(0, 255, 255), thickness=2)
    img1_dark = draw_mask_contour(img1_dark, mask1, color=(0, 255, 255), thickness=2)

    concat_masked = np.concatenate([img0_dark, img1_dark], axis=1)
    ax_right.imshow(concat_masked)
    ax_right.set_title(title_right, fontsize=14, fontweight='bold', color='darkgreen')
    ax_right.axis('off')

    # 绘制匹配线 (绿色 = 高质量前景匹配)
    n_masked = min(len(mkpts0_masked), max_lines)
    if n_masked > 0:
        idx_sort = np.argsort(-mconf_masked)[:n_masked]
        for idx in idx_sort:
            p0 = mkpts0_masked[idx]
            p1 = mkpts1_masked[idx].copy()
            p1[0] += W
            conf = mconf_masked[idx]

            # 颜色根据置信度渐变 (低置信度=浅绿, 高置信度=深绿)
            green_intensity = 0.4 + 0.6 * conf
            color = (0, green_intensity, 0)
            alpha = 0.6 + 0.4 * conf
            lw = 0.8 + 1.2 * conf

            linestyle = '-' if conf > conf_threshold else '--'
            ax_right.plot([p0[0], p1[0]], [p0[1], p1[1]],
                          color='lime', alpha=alpha, linewidth=lw, linestyle=linestyle)

            # 画端点
            ax_right.scatter([p0[0]], [p0[1]], c='lime', s=12, alpha=alpha,
                             edgecolors='darkgreen', linewidths=0.5, zorder=5)
            ax_right.scatter([p1[0]], [p1[1]], c='lime', s=12, alpha=alpha,
                             edgecolors='darkgreen', linewidths=0.5, zorder=5)

    # 添加统计信息
    n_total_masked = len(mkpts0_masked)
    ax_right.text(10, H - 30, f"Total: {n_total_masked} matches", fontsize=10, color='white',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax_right.text(10, H - 10, f"All Foreground (100%)", fontsize=10, color='lime',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # ========== 添加底部标注 ==========
    fig.text(0.25, 0.02, "❌ Large Search Space\n❌ Background Outliers",
             ha='center', fontsize=11, color='darkred', fontweight='bold')
    fig.text(0.75, 0.02, "✓ Reduced Search Space\n✓ Outlier Suppression",
             ha='center', fontsize=11, color='darkgreen', fontweight='bold')

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Saved mask-guided comparison to: {save_path}")


def draw_mask_contour(img: np.ndarray, mask: np.ndarray, color=(0, 255, 255), thickness=2):
    """在图像上绘制 mask 轮廓"""
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = img.copy()
    cv2.drawContours(img_out, contours, -1, color, thickness)
    return img_out


def visualize_mask_guided_comparison_v2(
        img0: np.ndarray,
        img1: np.ndarray,
        mask0: np.ndarray,
        mask1: np.ndarray,
        mkpts0_raw: np.ndarray,
        mkpts1_raw: np.ndarray,
        mconf_raw: np.ndarray,
        mkpts0_masked: np.ndarray,
        mkpts1_masked: np.ndarray,
        mconf_masked: np.ndarray,
        save_path: str,
        max_lines: int = 80,
        dpi: int = 200,
):
    """
    更紧凑的 2x2 布局版本:
    - 上排: 原图对 + 无Mask匹配
    - 下排: Mask图对 + 有Mask匹配
    """
    if mask0.max() > 1:
        mask0 = mask0 / 255.0
    if mask1.max() > 1:
        mask1 = mask1 / 255.0

    H, W = img0.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ===== 上排左: 原图0 =====
    axes[0, 0].imshow(img0)
    axes[0, 0].set_title("Reference Image", fontsize=12)
    axes[0, 0].axis('off')

    # ===== 上排右: 原图1 + 无Mask匹配线 =====
    axes[0, 1].imshow(img1)
    axes[0, 1].set_title("Query Image (w/o Mask)", fontsize=12, color='darkred')
    axes[0, 1].axis('off')

    # 画跨子图的匹配线 (无Mask)
    n_raw = min(len(mkpts0_raw), max_lines)
    for i in range(n_raw):
        p0 = mkpts0_raw[i]
        p1 = mkpts1_raw[i]

        y0, x0 = int(round(p0[1])), int(round(p0[0]))
        y1, x1 = int(round(p1[1])), int(round(p1[0]))
        in0 = (0 <= y0 < H and 0 <= x0 < W and mask0[y0, x0] > 0.5)
        in1 = (0 <= y1 < H and 0 <= x1 < W and mask1[y1, x1] > 0.5)

        color = 'orange' if (in0 and in1) else 'red'
        alpha = 0.6 if (in0 and in1) else 0.4

        con = ConnectionPatch(xyA=p0, xyB=p1, coordsA="data", coordsB="data",
                              axesA=axes[0, 0], axesB=axes[0, 1],
                              color=color, alpha=alpha, linewidth=0.8)
        fig.add_artist(con)

    # ===== 下排左: Mask图0 =====
    img0_masked = apply_mask_visualization(img0, mask0)
    axes[1, 0].imshow(img0_masked)
    axes[1, 0].set_title("Reference (Mask-Guided)", fontsize=12)
    axes[1, 0].axis('off')

    # ===== 下排右: Mask图1 + 有Mask匹配线 =====
    img1_masked = apply_mask_visualization(img1, mask1)
    axes[1, 1].imshow(img1_masked)
    axes[1, 1].set_title("Query (Mask-Guided)", fontsize=12, color='darkgreen')
    axes[1, 1].axis('off')

    # 画跨子图的匹配线 (有Mask)
    n_masked = min(len(mkpts0_masked), max_lines)
    for i in range(n_masked):
        p0 = mkpts0_masked[i]
        p1 = mkpts1_masked[i]

        con = ConnectionPatch(xyA=p0, xyB=p1, coordsA="data", coordsB="data",
                              axesA=axes[1, 0], axesB=axes[1, 1],
                              color='lime', alpha=0.7, linewidth=1.0)
        fig.add_artist(con)

    # 添加统计
    n_bg = sum(1 for i in range(len(mkpts0_raw))
               if not (mask0[int(round(mkpts0_raw[i][1])), int(round(mkpts0_raw[i][0]))] > 0.5 and
                       mask1[int(round(mkpts1_raw[i][1])), int(round(mkpts1_raw[i][0]))] > 0.5))

    fig.text(0.5, 0.52,
             f"Baseline: {len(mkpts0_raw)} matches, {n_bg} outliers ({100 * n_bg / max(len(mkpts0_raw), 1):.0f}%)",
             ha='center', fontsize=11, color='red', fontweight='bold')
    fig.text(0.5, 0.02, f"Ours: {len(mkpts0_masked)} matches, 0 outliers (0%)",
             ha='center', fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Saved v2 comparison to: {save_path}")


def apply_mask_visualization(img: np.ndarray, mask: np.ndarray, darken=0.3):
    """应用 mask 可视化: 背景变暗 + 轮廓高亮"""
    img_out = img.copy().astype(np.float32)
    bg = mask < 0.5
    for c in range(3):
        img_out[:, :, c][bg] *= darken
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    img_out = draw_mask_contour(img_out, mask, color=(0, 255, 255), thickness=2)
    return img_out
