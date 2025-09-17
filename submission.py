import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import torch
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat

from config.default import cfg as default_cfg
from yacs.config import CfgNode as CN
from omegaconf import DictConfig

from lib.models.builder import build_model
from lib.datasets.datamodules import DataModule


@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {'float': lambda v: f'{v:.6f}'}
        max_line_width = 1000
        q_str = np.array2string(self.q, formatter=formatter, max_line_width=max_line_width)[1:-1]
        t_str = np.array2string(self.t, formatter=formatter, max_line_width=max_line_width)[1:-1]
        return f'{self.image_name} {q_str} {t_str} {self.inliers}'


def get_dataset_args_dict(dataset_name: str, root_path: str, seed: int = 42):
    assert dataset_name in ['Shapenet6D', 'NOCS', 'TOYL'], f"Unsupported dataset: {dataset_name}"
    if dataset_name == 'Shapenet6D':
        obj_id, name = '03001627', 'ShapeNet6D'
    elif dataset_name == 'NOCS':
        obj_id, name = 'all', 'NOCS'
    elif dataset_name == 'TOYL':
        obj_id, name = '1', 'TOYL'

    args_dict = {
        'dataset': {
            'root': root_path,
            'img_size': [192, 192],
            'max_corrs': 4,
            'train': {'name': name, 'split': 'train', 'obj': obj_id},
            'test': {'name': name, 'split': 'val', 'obj': obj_id}
        },
        'TRAINING': {
            'BATCH_SIZE': 1,
            'NUM_WORKERS': 1,
            'SAMPLER': 'scene_balance',
            'N_SAMPLES_SCENE': 4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter': True, 'bright': True, 'hflip': True, 'vflip': True}
        },
        'test': {'mask': 'oracle', 'add_description': 'yes'},
        'use_seed': True,
        'seed': seed,
        'debug_valid': 'anchor'
    }
    return args_dict, dataset_name

def recursive_to_cuda(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, dict):
        return {k: recursive_to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cuda(v) for v in obj]
    else:
        return obj

#
# def predict(loader, model):
#     results_dict = defaultdict(list)
#
#     for data in tqdm(loader):
#         # 将所有 tensor 移动到 GPU
#         data = recursive_to_cuda(data)
#
#         with torch.no_grad():
#             R_batched, t_batched = model(data)
#
#         for i_batch in range(len(data['scene_id'])):
#             R = R_batched[i_batch].unsqueeze(0).detach().cpu().numpy()
#             t = t_batched[i_batch].reshape(-1).detach().cpu().numpy()
#             inliers = data['inliers'][i_batch].item()
#
#             scene = data['scene_id'][i_batch]
#             query_img = data['pair_names'][1][i_batch]
#
#             # ignore frames without poses
#             if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
#                 continue
#
#             estimated_pose = Pose(
#                 image_name=query_img,
#                 q=mat2quat(R).reshape(-1),
#                 t=t.reshape(-1),
#                 inliers=inliers
#             )
#             results_dict[scene].append(estimated_pose)
#
#     return results_dict
#
# def predict(loader, model):
#     results_dict = defaultdict(list)
#
#     for data in tqdm(loader):
#         data = recursive_to_cuda(data)
#
#         with torch.no_grad():
#             R_batched, t_batched = model(data)
#
#         B = R_batched.shape[0]
#         for i_batch in range(B):
#             R = R_batched[i_batch].unsqueeze(0).detach().cpu().numpy()
#             t = t_batched[i_batch].reshape(-1).detach().cpu().numpy()
#
#             inliers = data['inliers'][i_batch].item()
#             print(inliers)
#             # scene = f'scene_{i_batch:04d}'
#             # query_img = f'query_{i_batch:04d}.jpg'
#             scene = data['scene_id'][i_batch]
#             query_img = data['pair_names'][1][i_batch]
#
#              # ignore frames without poses
#             if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
#                 continue
#
#             estimated_pose = Pose(
#                 image_name=query_img,
#                 q=mat2quat(R).reshape(-1),
#                 t=t.reshape(-1),
#                 inliers=inliers
#             )
#             results_dict[scene].append(estimated_pose)
#
#     return results_dict


def predict(loader, model):
    results_dict = defaultdict(list)

    for data in tqdm(loader):
        data = recursive_to_cuda(data)
        print(data)
        with torch.no_grad():
            R_batched, t_batched = model(data)

        B = R_batched.shape[0]
        for i_batch in range(B):
            R = R_batched[i_batch].unsqueeze(0).detach().cpu().numpy()
            t = t_batched[i_batch].reshape(-1).detach().cpu().numpy()

            # 读取 instance_id

            instance_id = data['instance_id'][i_batch]  # 直接读取

            inliers = data.get('inliers', [torch.tensor(1.0)])[i_batch].item()  # 如果没有 inliers 默认 1.0

            # 如果姿态非法，跳过
            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                continue

            estimated_pose = Pose(
                image_name=instance_id,  # 改为 instance_id
                q=mat2quat(R).reshape(-1),
                t=t.reshape(-1),
                inliers=inliers
            )
            results_dict['scene'].append(estimated_pose)  # 所有结果放同一个 scene

    return results_dict


def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, 'w') as zip:
        for scene, poses in results_dict.items():
            poses_str = '\n'.join(str(pose) for pose in poses)
            zip.writestr(f'pose_{scene}.txt', poses_str.encode('utf-8'))


 # def save_submission(results_dict: dict, output_path: Path):
 #    with ZipFile(output_path, 'w') as zip:
 #        for scene, poses in results_dict.items():
 #            # 第一列 instance_id
 #            poses_str = '\n'.join(str(pose) for pose in poses)
 #            zip.writestr(f'pose_{scene}.txt', poses_str.encode('utf-8'))

def main(args):
    # Load and merge configs
    with open(args.config, "r") as f:
        yaml_cfg = CN.load_cfg(f)
    cfg = default_cfg.clone()
    cfg.merge_from_other_cfg(yaml_cfg)

    # 添加默认种子
    if cfg.DATASET.SEED is None:
        cfg.DATASET.SEED = 42

    # Dataset + dataloader
    args_dict, dataset_name = get_dataset_args_dict(args.dataset, args.dataset_root, seed=cfg.DATASET.SEED)
    args_dict = DictConfig(args_dict)

    datamodule = DataModule(args_dict, dataset_name)

    if args.split == 'test':
        loader = datamodule.test_dataloader()
    elif args.split == 'val':
        loader = datamodule.val_dataloader()
    else:
        raise NotImplementedError(f"Invalid split: {args.split}")

    # Create model
    model = build_model(cfg, args.checkpoint)

    # Predict & save
    results_dict = predict(loader, model)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_submission(results_dict, args.output_root / 'submission.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  help='Path to YAML config file used in training',default='config/MicKey/curriculum_learning_warm_up.yaml')
    parser.add_argument('--checkpoint',  default='weights/MicKey_default/version_242/checkpoints/epoch=77-best_vcre.ckpt', help='Path to trained model checkpoint')
    parser.add_argument('--dataset', choices=['Shapenet6D', 'NOCS', 'TOYL'],default='NOCS')
    parser.add_argument('--dataset_root', type=str, help='Root path to dataset files',default='filesOfOryon/data')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output_root', '-o', type=Path, default=Path('results/'))

    args = parser.parse_args()
    main(args)