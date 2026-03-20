"""
ObjectMatch 完整封装类 - 包含SuperGlue集成
一站式解决方案：从图像输入到配准结果
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

from ObjectMatch.pairwise_backend import (
    KPConfig,
    NOCPredConfig,
    ObjectIDConfig,
    OptimConfig,
    PairwiseSolver,
    PairwiseVisualizer,
    GNOutput,
)


@dataclass
class RegistrationResult:
    """配准结果"""
    camera_pose: np.ndarray          # 4x4 相机位姿矩阵
    success: bool = True             # 是否成功
    translation_error: Optional[float] = None  # 平移误差(cm)
    rotation_error: Optional[float] = None     # 旋转误差(度)
    gn_output: Optional[GNOutput] = None       # 全局优化输出
    extras: Optional[Any] = None               # 额外信息


class ObjectMatch:
    """
    ObjectMatch 完整封装类

    功能：
    1. 自动运行SuperGlue进行关键点匹配
    2. NOC预测和对象识别
    3. 全局优化和ICP精化
    4. 结果可视化和评估

    使用示例：
        # 最简单的使用
        matcher = ObjectMatch()
        result = matcher.register(image0_path, image1_path)

        # 带可视化
        result = matcher.register(image0_path, image1_path, visualize=True)

        # 批量处理
        results = matcher.batch_register(image_pairs)
    """

    def __init__(
        self,
        checkpoint_dir: str = 'ObjectMatch/checkpoints',
        superglue_dir: str = 'ObjectMatch/SuperGluePretrainedNetwork',
        superglue_model: str = 'indoor',  # 'indoor' or 'outdoor'
        match_cache_dir: str = './dump_features',
        verbose: bool = True,
    ):
        """
        初始化 ObjectMatch

        参数:
            checkpoint_dir: ObjectMatch模型目录
            superglue_dir: SuperGlue代码目录
            superglue_model: SuperGlue模型类型 ('indoor' 或 'outdoor')
            match_cache_dir: 关键点匹配缓存目录
            verbose: 是否打印详细信息
        """
        self.checkpoint_dir = checkpoint_dir
        self.superglue_dir = superglue_dir
        self.superglue_model = superglue_model
        self.match_cache_dir = match_cache_dir
        self.verbose = verbose

        # 创建缓存目录
        os.makedirs(match_cache_dir, exist_ok=True)

        # 初始化求解器
        self._init_solver()

        # 可视化器（按需初始化）
        self.visualizer = None

        if verbose:
            print("✓ ObjectMatch 初始化完成")

    def _init_solver(self):
        """初始化配准求解器"""
        print("model_file:", f'{self.checkpoint_dir}/model_sym.pth')
        self.solver = PairwiseSolver(
            kp_cfg=KPConfig(),
            nocpred_cfg=NOCPredConfig(
                model_file=f'{self.checkpoint_dir}/model_sym.pth',
                config_file='configs/NOCPred.yaml',
            ),
            objectid_cfg=ObjectIDConfig(
                folder=f'{self.checkpoint_dir}/all_5',
            ),
            optim_cfg=OptimConfig(verbose=self.verbose),
            print_fn=print if self.verbose else lambda *args: None,
        )

    def register(
        self,
        image0_path: str,
        image1_path: str,
        intrinsic0: Optional[np.ndarray] = None,
        intrinsic1: Optional[np.ndarray] = None,
        pose0: Optional[np.ndarray] = None,
        pose1: Optional[np.ndarray] = None,
        match_file: Optional[str] = None,
        visualize: bool = False,
        vis_name: Optional[str] = None,
    ) -> RegistrationResult:
        """
        配准两张图像（完整流程）

        参数:
            image0_path: 第一张图像路径
            image1_path: 第二张图像路径
            intrinsic0: 第一张图像的相机内参 (可选，会自动查找)
            intrinsic1: 第二张图像的相机内参 (可选，会自动查找)
            pose0: 第一张图像的真实位姿 (可选，用于评估)
            pose1: 第二张图像的真实位姿 (可选，用于评估)
            match_file: 预先计算的匹配文件 (可选，否则自动运行SuperGlue)
            visualize: 是否可视化结果
            vis_name: 可视化名称 (可选)

        返回:
            RegistrationResult: 配准结果
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"配准图像对:")
            print(f"  图像0: {image0_path}")
            print(f"  图像1: {image1_path}")
            print(f"{'='*60}")

        # 1. 运行SuperGlue获取关键点匹配
        if match_file is None:
            if self.verbose:
                print("\n[1/4] 运行 SuperGlue 进行关键点匹配...")
            match_file = self._run_superglue(
                image0_path, image1_path,
                intrinsic0, intrinsic1,
                pose0, pose1
            )
        else:
            if self.verbose:
                print(f"\n[1/4] 使用已有匹配文件: {match_file}")

        # 2. 加载图像记录
        if self.verbose:
            print("[2/4] 加载图像数据...")
        record0 = self.solver.load_record(image0_path)
        record1 = self.solver.load_record(image1_path)
        match_data = np.load(match_file)

        # 3. 执行配准
        if self.verbose:
            print("[3/4] 执行配准优化...")
        pred_pose, gn_output, extras = self.solver(
            record0, record1, match_data,
            ret_extra_outputs=True
        )

        # 4. 评估（如果提供了真实位姿）
        te, ae = None, None
        if pose0 is not None and pose1 is not None:
            if self.verbose:
                print("[4/4] 评估配准精度...")
            from optim.solver import test
            gt_pose = np.linalg.inv(pose0) @ pose1
            te, ae = test(pred_pose, gt_pose)
            if self.verbose:
                print(f"  平移误差: {te:.2f} cm")
                print(f"  旋转误差: {ae:.2f} 度")
                print(f"  配准{'成功' if (te <= 30 and ae <= 15) else '失败'}")

        # 5. 可视化
        if visualize:
            if vis_name is None:
                vis_name = f"{Path(image0_path).stem}_{Path(image1_path).stem}"
            if self.verbose:
                print(f"[可视化] 生成可视化结果...")
            self._visualize(vis_name, record0, record1, pred_pose, gn_output, extras)

        if self.verbose:
            print(f"{'='*60}\n")

        return RegistrationResult(
            camera_pose=pred_pose,
            success=gn_output.success,
            translation_error=te,
            rotation_error=ae,
            gn_output=gn_output,
            extras=extras,
        )

    def _run_superglue(
        self,
        image0_path: str,
        image1_path: str,
        intrinsic0: Optional[np.ndarray] = None,
        intrinsic1: Optional[np.ndarray] = None,
        pose0: Optional[np.ndarray] = None,
        pose1: Optional[np.ndarray] = None,
    ) -> str:
        """
        运行SuperGlue生成关键点匹配（参考 pair_eval.py 的 run_sg 函数）

        返回:
            match_file: 匹配文件路径
        """
        # 生成匹配文件名（参考 pair_eval.py）
        name0 = Path(image0_path).stem
        name1 = Path(image1_path).stem

        # 从路径中提取场景名
        scene_name = None
        for part in Path(image0_path).parts:
            if part.startswith('scene') or part.startswith('rgbd_dataset'):
                scene_name = part
                break

        if scene_name:
            match_file = os.path.join(
                self.match_cache_dir,
                f"{scene_name}_{name0}_{name1}_matches.npz"
            )
        else:
            match_file = os.path.join(
                self.match_cache_dir,
                f"{name0}_{name1}_matches.npz"
            )

        # 如果已存在，直接返回
        if os.path.exists(match_file):
            if self.verbose:
                print(f"  使用缓存的匹配文件")
            return match_file

        # 自动查找内参
        if intrinsic0 is None:
            intrinsic0 = self._find_intrinsic(image0_path)
        if intrinsic1 is None:
            intrinsic1 = self._find_intrinsic(image1_path)

        # 确保内参是3x3矩阵
        if intrinsic0.shape == (4, 4):
            intrinsic0 = intrinsic0[:3, :3]
        if intrinsic1.shape == (4, 4):
            intrinsic1 = intrinsic1[:3, :3]

        # 自动查找位姿
        if pose0 is None:
            pose0 = self._find_pose(image0_path)
        if pose1 is None:
            pose1 = self._find_pose(image1_path)

        # 创建临时配对文件（参考 pair_eval.py）
        temp_pairs = os.path.join(self.match_cache_dir, 'temp_pairs.txt')
        with open(temp_pairs, 'w') as f:
            # 计算相对位姿（参考 pair_eval.py）
            if pose0 is not None and pose1 is not None:
                gt_pose_sg = np.linalg.inv(np.linalg.inv(pose0) @ pose1)
            else:
                gt_pose_sg = np.eye(4)

            # 格式: image0 image1 rot0 rot1 K0 K1 pose
            line = '{} {} 0 0 {} {} {}\n'.format(
                image0_path,
                image1_path,
                ' '.join(map(str, intrinsic0.ravel().tolist())),
                ' '.join(map(str, intrinsic1.ravel().tolist())),
                ' '.join(map(str, gt_pose_sg.ravel().tolist())),
            )
            f.write(line)

        # 运行SuperGlue（参考 pair_eval.py）
        sg_script = os.path.join(self.superglue_dir, 'match_pairs_scannet.py')
        cmd = (
            f'{sys.executable} -u {sg_script} '
            f'--input_dir "/" '
            f'--input_pairs {temp_pairs} '
            f'--output_dir {self.match_cache_dir}'
        )

        if self.verbose:
            print(f"  运行 SuperGlue...")

        ret = os.system(cmd)

        # 清理临时文件
        if os.path.exists(temp_pairs):
            os.unlink(temp_pairs)

        if ret != 0:
            raise RuntimeError(f"SuperGlue 运行失败，返回码: {ret}")

        if not os.path.exists(match_file):
            raise RuntimeError(f"SuperGlue 未生成匹配文件: {match_file}")

        return match_file

    def _find_intrinsic(self, image_path: str) -> np.ndarray:
        """自动查找相机内参"""
        from optim.common import load_matrix

        # 尝试多个可能的位置
        image_dir = Path(image_path).parent
        scene_dir = image_dir.parent

        possible_paths = [
            scene_dir / 'intrinsic_depth.txt',
            scene_dir / 'intrinsic.txt',
            scene_dir / 'camera_intrinsic.txt',
        ]

        for path in possible_paths:
            if path.exists():
                intrinsic = load_matrix(str(path))
                if intrinsic.shape == (3, 3):
                    return intrinsic
                elif intrinsic.shape == (4, 4):
                    return intrinsic[:3, :3]

        # 使用默认内参
        if self.verbose:
            print(f"  警告: 未找到内参文件，使用默认值")
        return np.array([
            [577.87, 0, 319.5],
            [0, 577.87, 239.5],
            [0, 0, 1]
        ])

    def _find_pose(self, image_path: str) -> Optional[np.ndarray]:
        """自动查找相机位姿"""
        from optim.common import load_matrix

        image_path = Path(image_path)
        scene_dir = image_path.parent.parent
        pose_dir = scene_dir / 'pose'

        if pose_dir.exists():
            pose_file = pose_dir / f"{image_path.stem}.txt"
            if pose_file.exists():
                return load_matrix(str(pose_file))

        return None

    def _visualize(
        self,
        vis_name: str,
        record0: Dict,
        record1: Dict,
        pred_pose: np.ndarray,
        gn_output: GNOutput,
        extras: Any,
        output_dir: str = 'vis_pairs',
    ):
        """可视化配准结果"""
        if self.visualizer is None:
            self.visualizer = PairwiseVisualizer(
                vis_output_dir=output_dir,
                print_fn=print if self.verbose else lambda *args: None,
            )

        # 需要从record中提取图像路径
        # 这里简化处理，直接调用可视化器
        self.visualizer(
            vis_name=vis_name,
            record0=record0,
            record1=record1,
            pred_pose=pred_pose,
            gn_output=gn_output,
            extra_outputs=extras,
        )

    def batch_register(
        self,
        image_pairs: List[Tuple[str, str]],
        intrinsics: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        poses: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        visualize: bool = False,
        output_dir: str = 'vis_pairs',
    ) -> List[RegistrationResult]:
        """
        批量配准多对图像

        参数:
            image_pairs: 图像对列表 [(img0, img1), ...]
            intrinsics: 内参列表 (可选)
            poses: 位姿列表 (可选，用于评估)
            visualize: 是否可视化
            output_dir: 可视化输出目录

        返回:
            results: 配准结果列表
        """
        results = []

        for i, (img0, img1) in enumerate(image_pairs):
            if self.verbose:
                print(f"\n处理第 {i+1}/{len(image_pairs)} 对图像")

            # 获取内参和位姿
            intr0, intr1 = None, None
            pose0, pose1 = None, None

            if intrinsics is not None and i < len(intrinsics):
                intr0, intr1 = intrinsics[i]

            if poses is not None and i < len(poses):
                pose0, pose1 = poses[i]

            # 配准
            vis_name = f"pair_{i:04d}" if visualize else None
            result = self.register(
                img0, img1,
                intrinsic0=intr0,
                intrinsic1=intr1,
                pose0=pose0,
                pose1=pose1,
                visualize=visualize,
                vis_name=vis_name,
            )

            results.append(result)

        # 统计
        if self.verbose:
            success_count = sum(1 for r in results if r.success)
            print(f"\n{'='*60}")
            print(f"批量处理完成:")
            print(f"  总数: {len(results)}")
            print(f"  成功: {success_count}")

            if poses is not None:
                eval_success = sum(
                    1 for r in results
                    if r.translation_error is not None
                    and r.translation_error <= 30
                    and r.rotation_error <= 15
                )
                print(f"  评估成功率: {eval_success}/{len(results)} "
                      f"({eval_success/len(results)*100:.1f}%)")
            print(f"{'='*60}\n")

        return results

    def update_config(
        self,
        kp_config: Optional[Dict] = None,
        noc_config: Optional[Dict] = None,
        objectid_config: Optional[Dict] = None,
        optim_config: Optional[Dict] = None,
    ):
        """更新配置参数"""
        # 重新初始化求解器
        self._init_solver()
        if self.verbose:
            print("✓ 配置已更新")


# 便捷函数
def quick_register(
    image0_path: str,
    image1_path: str,
    checkpoint_dir: str = 'checkpoints',
    visualize: bool = False,
) -> RegistrationResult:
    """
    最简单的配准函数 - 一行代码完成配准

    参数:
        image0_path: 第一张图像路径
        image1_path: 第二张图像路径
        checkpoint_dir: 模型目录
        visualize: 是否可视化

    返回:
        RegistrationResult: 配准结果

    示例:
        result = quick_register('img0.jpg', 'img1.jpg', visualize=True)
        print(result.camera_pose)
    """
    matcher = ObjectMatch(checkpoint_dir=checkpoint_dir, verbose=True)
    return matcher.register(image0_path, image1_path, visualize=visualize)
