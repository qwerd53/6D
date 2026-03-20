"""
诊断深度图和内参问题
"""
import os
import sys

sys.path.append(os.getcwd())

import cv2
import numpy as np
from ObjectMatch.optim.common import load_matrix

img0 = "tempObjectMatching/scene_76788_3/color/0.png"
scene_dir = os.path.dirname(os.path.dirname(img0))

print("=" * 70)
print("深度图和内参诊断")
print("=" * 70)

# 1. 检查内参
print("\n1. 检查内参")
intrinsic_file = os.path.join(scene_dir, "intrinsic_depth.txt")
if os.path.exists(intrinsic_file):
    intr = load_matrix(intrinsic_file)
    print(f"  ✓ 内参文件: {intrinsic_file}")
    print(f"  内参矩阵:\n{intr}")
    print(f"  fx={intr[0, 0]:.2f}, fy={intr[1, 1]:.2f}")
    print(f"  cx={intr[0, 2]:.2f}, cy={intr[1, 2]:.2f}")
else:
    print(f"  ✗ 内参文件不存在: {intrinsic_file}")

# 2. 检查深度图
print("\n2. 检查深度图")
depth0_path = img0.replace("/color/", "/depth/").replace(".jpg", ".png")
print(f"  深度图路径: {depth0_path}")

if os.path.exists(depth0_path):
    depth = cv2.imread(depth0_path, cv2.IMREAD_ANYDEPTH)
    print(f"  ✓ 深度图存在")
    print(f"  Shape: {depth.shape}")
    print(f"  Dtype: {depth.dtype}")
    print(f"  范围: [{depth.min()}, {depth.max()}]")
    print(f"  非零像素: {np.count_nonzero(depth)} / {depth.size} ({np.count_nonzero(depth) / depth.size * 100:.1f}%)")

    # 检查深度单位
    if depth.max() > 10000:
        print(f"  ⚠ 深度值很大 (max={depth.max()})，可能单位是毫米")
        print(f"     需要除以 1000 转换为米")
    elif depth.max() > 100:
        print(f"  ⚠ 深度值较大 (max={depth.max()})，可能单位是厘米")
        print(f"     需要除以 100 转换为米")
    elif depth.max() < 0.1:
        print(f"  ⚠ 深度值很小 (max={depth.max()})，可能已经是米但缩放了")
    else:
        print(f"  ✓ 深度值范围合理 (max={depth.max()})，可能单位是米")

    # 统计深度分布
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        print(f"\n  有效深度统计:")
        print(f"    Mean: {valid_depth.mean():.2f}")
        print(f"    Median: {np.median(valid_depth):.2f}")
        print(f"    Std: {valid_depth.std():.2f}")
        print(f"    Percentiles: 25%={np.percentile(valid_depth, 25):.2f}, "
              f"75%={np.percentile(valid_depth, 75):.2f}")
else:
    print(f"  ✗ 深度图不存在")

# 3. 检查 RGB 图像
print("\n3. 检查 RGB 图像")
if os.path.exists(img0):
    rgb = cv2.imread(img0)
    print(f"  ✓ RGB 图像: {img0}")
    print(f"  Shape: {rgb.shape}")

    if os.path.exists(depth0_path):
        depth = cv2.imread(depth0_path, cv2.IMREAD_ANYDEPTH)
        if rgb.shape[:2] != depth.shape:
            print(f"  ✗ 错误：RGB 和深度图尺寸不匹配！")
            print(f"     RGB: {rgb.shape[:2]}, Depth: {depth.shape}")
        else:
            print(f"  ✓ RGB 和深度图尺寸匹配")
else:
    print(f"  ✗ RGB 图像不存在")

# 4. 测试深度反投影
print("\n4. 测试深度反投影")
if os.path.exists(intrinsic_file) and os.path.exists(depth0_path):
    intr = load_matrix(intrinsic_file)[:3, :3]
    depth = cv2.imread(depth0_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

    # 假设深度单位是毫米，转换为米
    if depth.max() > 10000:
        print(f"  转换深度单位：毫米 → 米")
        depth = depth / 1000.0

    # 选择图像中心的一个点测试
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    d = depth[cy, cx]

    if d > 0:
        # 反投影公式
        fx, fy = intr[0, 0], intr[1, 1]
        cx_intr, cy_intr = intr[0, 2], intr[1, 2]

        X = (cx - cx_intr) * d / fx
        Y = (cy - cy_intr) * d / fy
        Z = d

        print(f"  测试点: 像素({cx}, {cy}), 深度={d:.3f}m")
        print(f"  3D 坐标: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

        if abs(X) > 10 or abs(Y) > 10 or abs(Z) > 10:
            print(f"  ⚠ 警告：3D 坐标异常大，可能内参或深度单位有问题")
        else:
            print(f"  ✓ 3D 坐标看起来合理")
    else:
        print(f"  ⚠ 图像中心深度为 0，无法测试")

# 5. 加载 SuperGlue 匹配并检查关键点深度
print("\n5. 检查 SuperGlue 匹配点的深度")
match_file = "ObjectMatch/dump_features/scene0031_0_1_matches.npz"
if os.path.exists(match_file) and os.path.exists(depth0_path):
    match_data = np.load(match_file)
    matches = match_data['matches']
    kpts0 = match_data['keypoints0']

    ind0 = np.argwhere(matches > 0).ravel()

    depth = cv2.imread(depth0_path, cv2.IMREAD_ANYDEPTH)

    zero_count = 0
    valid_depths = []

    for idx in ind0[:10]:  # 检查前 10 个
        kp = kpts0[idx]
        c, r = int(kp[0]), int(kp[1])

        if 0 <= r < depth.shape[0] and 0 <= c < depth.shape[1]:
            d = depth[r, c]
            if d == 0:
                zero_count += 1
            else:
                valid_depths.append(d)
            print(f"  匹配点 {idx}: 像素({c},{r}), 深度={d}")
        else:
            print(f"  匹配点 {idx}: 越界")
            zero_count += 1

    print(f"\n  前 10 个匹配点中，{zero_count} 个深度为 0")
    if len(valid_depths) > 0:
        print(f"  有效深度范围: [{min(valid_depths)}, {max(valid_depths)}]")
        if max(valid_depths) > 10000:
            print(f"  ⚠ 深度单位可能是毫米，需要转换")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
