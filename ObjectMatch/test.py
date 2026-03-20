# import os
# import sys
# sys.path.append(os.getcwd())
#
# import numpy as np
# from ObjectMatch.pairwise_backend import (
#     KPConfig,
#     NOCPredConfig,
#     ObjectIDConfig,
#     OptimConfig,
#     PairwiseSolver,
#     PairwiseVisualizer,
# )
# from ObjectMatch.optim.solver import test
# from ObjectMatch.optim.common import load_matrix
#
# def run_superglue_match(img0, img1, out_dir="ObjectMatch/dump_features"):
#     os.makedirs(out_dir, exist_ok=True)
#
#     # 提取 SuperGlue 实际命名规则
#     scene = img0.split('/')[-3]  # e.g., scene0030_00
#     frame0 = os.path.basename(img0).split('.')[0]
#     frame1 = os.path.basename(img1).split('.')[0]
#
#     # SuperGlue 生成的真实名字
#     basename = f"{scene}_{frame0}_{frame1}"
#     match_path = os.path.join(out_dir, f"{basename}_matches.npz")
#
#     # 如果已存在则直接返回
#     if os.path.isfile(match_path):
#         print(f"[SG] Load cached match: {match_path}")
#         return match_path
#
#     print("[SG] Running SuperGlue matching...")
#
#     temp_txt = "temp_sg_test.txt"
#     with open(temp_txt, "w") as f:
#         intr_flat = " ".join(["0"] * 9)
#         pose_dummy = " ".join(["0"] * 16)
#         f.write(f"{img0} {img1} 0 0 {intr_flat} {intr_flat} {pose_dummy}\n")
#
#     cmd = (
#         f"{sys.executable} -u "
#         f"ObjectMatch/SuperGluePretrainedNetwork/match_pairs_scannet.py "
#         f"--input_dir '.' "
#         f"--input_pairs {temp_txt} "
#         f"--output_dir {out_dir}"
#     )
#
#     os.system(cmd)
#     os.unlink(temp_txt)
#
#     print(f"[SG] match saved to {match_path}")
#     return match_path
#
#
#
# def run_match(
#     img0, img1,
#     ckpt_noc="ObjectMatch/checkpoints/model_sym.pth",
#     ckpt_objid="ObjectMatch/checkpoints/all_5"
# ):
#
#     # ① 构建 ObjectMatch 求解器
#     solver = PairwiseSolver(
#         KPConfig(),
#         NOCPredConfig(ckpt_noc, "configs/NOCPred.yaml"),
#         ObjectIDConfig(ckpt_objid),
#         OptimConfig(verbose=True),
#     )
#
#     # ② 读取 RGBD + Mask + NOCS（ObjectMatch 的标准数据结构）
#     record0 = solver.load_record(img0)
#     record1 = solver.load_record(img1)
#
#     # ③ SuperGlue 匹配
#     match_file = run_superglue_match(img0, img1)
#     match_data = np.load(match_file)
#
#     print("Running ObjectMatch...")
#
#     # ④ GN 优化求解 6D 位姿
#     pred_pose, gn_output, extras = solver(
#         record0,
#         record1,
#         match_data=match_data,
#         ret_extra_outputs=True
#     )
#
#     print("\nPredicted Pose:")
#     print(pred_pose)
#
#     return pred_pose, extras
#
#
# if __name__ == "__main__":
#     img0 = "ObjectMatch/assets/sample_pairs/scene0030_00/color/1800.jpg"
#     img1 = "ObjectMatch/assets/sample_pairs/scene0030_00/color/1900.jpg"
#
#     run_match(img0, img1)
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from ObjectMatch.pairwise_backend import (
    KPConfig,
    NOCPredConfig,
    ObjectIDConfig,
    OptimConfig,
    PairwiseSolver,
    PairwiseVisualizer,
)
from ObjectMatch.optim.solver import test
from ObjectMatch.optim.common import load_matrix


def run_superglue_match(img0, img1, out_dir="ObjectMatch/dump_features",
                        match_threshold=0.2, keypoint_threshold=0.005,
                        max_keypoints=1024, force_rerun=False):
    """
    运行 SuperGlue 匹配

    Args:
        match_threshold: 匹配阈值 (默认 0.2)
        keypoint_threshold: 关键点检测阈值 (默认 0.005)
        max_keypoints: 最大关键点数 (默认 1024)
        force_rerun: 强制重新运行，忽略缓存
    """
    os.makedirs(out_dir, exist_ok=True)

    # 提取 SuperGlue 实际命名规则
    scene = img0.split('/')[-3]  # e.g., scene0030_00
    frame0 = os.path.basename(img0).split('.')[0]
    frame1 = os.path.basename(img1).split('.')[0]

    # SuperGlue 生成的真实名字
    basename = f"{scene}_{frame0}_{frame1}"
    match_path = os.path.join(out_dir, f"{basename}_matches.npz")

    # 如果已存在则直接返回（除非强制重新运行）
    if os.path.isfile(match_path) and not force_rerun:
        print(f"[SG] Load cached match: {match_path}")
        print(f"[SG] ⚠ 提示：使用缓存文件，如需重新匹配请删除此文件")
        return match_path

    print(f"[SG] Running SuperGlue matching...")
    print(f"[SG]   match_threshold={match_threshold}")
    print(f"[SG]   keypoint_threshold={keypoint_threshold}")
    print(f"[SG]   max_keypoints={max_keypoints}")

    # ========== 关键修复：加载真实的相机内参 ==========
    # 尝试加载内参文件（参照官方 pair_eval.py）
    scene_dir = os.path.dirname(os.path.dirname(img0))  # 回到 scene 目录
    intrinsic_file = os.path.join(scene_dir, "intrinsic.txt")
    if not os.path.exists(intrinsic_file):
        intrinsic_file = os.path.join(scene_dir, "intrinsic_depth.txt")

    if os.path.exists(intrinsic_file):
        print(f"[SG] ✓ 加载相机内参: {intrinsic_file}")
        intr = load_matrix(intrinsic_file)[:3, :3]
        intr_flat = " ".join(map(str, intr.ravel().tolist()))
    else:
        print(f"[SG] ⚠ 警告：未找到内参文件，使用默认值")
        print(f"[SG]   尝试的路径: {intrinsic_file}")
        # 使用 ScanNet 的典型内参作为默认值
        intr_flat = "577.871 0 319.5 0 577.871 239.5 0 0 1"

    # 尝试加载 pose 文件（可选，SuperGlue 不强制需要）
    pose0_file = os.path.join(scene_dir, "pose", f"{frame0}.txt")
    pose1_file = os.path.join(scene_dir, "pose", f"{frame1}.txt")

    if os.path.exists(pose0_file) and os.path.exists(pose1_file):
        pose0 = load_matrix(pose0_file)
        pose1 = load_matrix(pose1_file)
        gt_pose_sg = np.linalg.inv(np.linalg.inv(pose0) @ pose1)
        pose_flat = " ".join(map(str, gt_pose_sg.ravel().tolist()))
        print(f"[SG] ✓ 加载 GT pose（用于 SuperGlue 参考）")
    else:
        pose_flat = " ".join(["0"] * 16)
        print(f"[SG] ⚠ 未找到 pose 文件，使用 dummy 值")

    temp_txt = "temp_sg_test.txt"
    with open(temp_txt, "w") as f:
        f.write(f"{img0} {img1} 0 0 {intr_flat} {intr_flat} {pose_flat}\n")

    cmd = (
        f"{sys.executable} -u "
        f"ObjectMatch/SuperGluePretrainedNetwork/match_pairs_scannet.py "
        f"--input_dir '.' "
        f"--input_pairs {temp_txt} "
        f"--output_dir {out_dir} "
        f"--match_threshold {match_threshold} "
        f"--keypoint_threshold {keypoint_threshold} "
        f"--max_keypoints {max_keypoints}"
    )

    os.system(cmd)
    os.unlink(temp_txt)

    print(f"[SG] match saved to {match_path}")

    return match_path


# ============================================================
#  主函数：运行 ObjectMatch
#  visualize=True 则可视化
# ============================================================
def run_match(
        img0, img1,
        ckpt_noc="ObjectMatch/checkpoints/model_sym.pth",
        ckpt_objid="ObjectMatch/checkpoints/all_5",
        visualize=False,
):
    # ① 构建 ObjectMatch 求解器（使用默认配置）
    solver = PairwiseSolver(
        KPConfig(),  # 默认: min_points=4, proc_thresh=0.2
        NOCPredConfig(ckpt_noc, "configs/NOCPred.yaml"),
        ObjectIDConfig(ckpt_objid),
        OptimConfig(verbose=True),
    )

    # ② 读取 RGB + 深度 + mask + NOCS
    record0 = solver.load_record(img0)
    record1 = solver.load_record(img1)

    # ③ SuperGlue 匹配（使用默认参数）
    match_path = run_superglue_match(img0, img1)
    match_data = np.load(match_path)

    # 显示 SuperGlue 匹配统计信息
    matches = match_data['matches']
    kpts0 = match_data['keypoints0']
    kpts1 = match_data['keypoints1']
    conf = match_data['match_confidence']

    num_matches = np.sum(matches > -1)
    print(f"\n{'=' * 60}")
    print(f"SuperGlue 匹配统计")
    print(f"{'=' * 60}")
    print(f"图像 0 关键点数: {len(kpts0)}")
    print(f"图像 1 关键点数: {len(kpts1)}")
    print(f"有效匹配数: {num_matches}")
    if num_matches > 0:
        valid_conf = conf[matches > -1]
        print(f"匹配置信度: min={valid_conf.min():.3f}, max={valid_conf.max():.3f}, mean={valid_conf.mean():.3f}")
    else:
        print("⚠ 警告: SuperGlue 没有找到任何匹配!")
        print("   可能原因: 视角变化太大 / 场景重叠度低 / 低纹理")
    print(f"{'=' * 60}\n")

    print("Running ObjectMatch...")

    # ④ GN 优化求解 6D 位姿
    pred_pose, gn_output, extras = solver(
        record0,
        record1,
        match_data=match_data,
        ret_extra_outputs=True
    )

    print("\nPredicted Pose (T_0_to_1):")
    print(pred_pose)

    # =====================================================
    # ⑤ 可视化（严格按 paireval 的调用方式）
    # =====================================================
    if visualize:
        print("\n[Vis] 可视化匹配 / 关键点 / 深度 / 配准点云")

        # ---- 1) 生成 vis_name（字符串）----
        scene = img0.split('/')[-3]  # e.g. scene0030_00
        f0 = os.path.basename(img0).split('.')[0]
        f1 = os.path.basename(img1).split('.')[0]
        vis_name = f"{scene}_{f0}_{f1}"

        # ---- 2) 传入 vis_name 与正确参数 ----
        visualizer = PairwiseVisualizer("vis_pairs")  # 输出文件夹
        visualizer(
            vis_name,
            record0,
            record1,
            pred_pose,
            gn_output,
            extras
        )

    # return pred_pose, extras
    return pred_pose


if __name__ == "__main__":
    # 可以修改这里测试不同的样本
    img0 = "tempObjectMatching/scene_2055_17/color/0.png"
    img1 = "tempObjectMatching/scene_2055_17/color/1.png"
    # 使用默认参数（内参已修正）
    run_match(img0, img1, visualize=True)
