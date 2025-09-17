import argparse
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from transforms3d.quaternions import mat2quat, quat2mat
from omegaconf import DictConfig
from collections import defaultdict

from filesOfOryon.bop_toolkit_lib import pose_error
from filesOfOryon.bop_toolkit_lib.pose_error import my_mssd, my_mspd, vsd
from filesOfOryon.bop_toolkit_lib.renderer_vispy import RendererVispy
from filesOfOryon.utils.data.nocs import get_obj_rendering

from config.default import cfg as default_cfg
from lib.models.builder import build_model
from lib.datasets.datamodules import DataModule

from transforms3d.axangles import axangle2mat

# ---------------------- Config ----------------------
MODELS_INFO_PATH = 'filesOfOryon/data/NOCS/obj_models/real_test/models_info.json'
MODELS_ROOT = 'filesOfOryon/data/NOCS'
CATEGORIES_PATH = 'filesOfOryon/data/NOCS/categories.json'
ANNOTATION_PATH = 'filesOfOryon/data/NOCS/fixed_split/val/annots.pkl'
DEPTH_ROOT = 'filesOfOryon/data/NOCS/split/real_test'

ADD_THRESHOLD = 0.005
MSSD_REC = np.arange(0.05, 0.51, 0.05)
MSPD_REC = np.arange(5, 51, 5)
VSD_REC = np.arange(0.05, 0.51, 0.05)
VSD_DELTA = 15.0

K_dict = {"fx": 591.0125, "fy": 590.16775, "cx": 322.525, "cy": 244.11084}
K = np.array([
    [K_dict["fx"], 0, K_dict["cx"]],
    [0, K_dict["fy"], K_dict["cy"]],
    [0, 0, 1]
], dtype=np.float32)


# ---------------------- Helper Functions ----------------------
def get_symmetries(models_info, obj_name):
    info = models_info.get(obj_name, {})
    syms = info.get('symmetries_continuous', [])
    sym_matrices = []
    for sym in syms:
        axis = np.array(sym['axis'], dtype=np.float32)
        axis_norm = axis / np.linalg.norm(axis)
        R_sym = axangle2mat(axis_norm, np.pi * 2)
        T_sym = np.eye(4, dtype=np.float32)
        T_sym[:3, :3] = R_sym
        sym_matrices.append(T_sym)
    if len(sym_matrices) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    else:
        return np.stack(sym_matrices, axis=0).astype(np.float32)


def fix_instance_id(instance_id, name_to_category_id):
    parts = instance_id.split('_')
    if len(parts) < 6:
        return instance_id
    obj_name = parts[4]
    category_id = name_to_category_id.get(obj_name, None)
    if category_id is None:
        return instance_id
    if parts[4] != category_id:
        parts.insert(4, category_id)
    return '_'.join(parts)


def load_depth(scene_q, img_q):
    import imageio.v2 as imageio
    path = os.path.join(DEPTH_ROOT, f'scene_{scene_q}', f'{img_q.zfill(4)}_depth.png')
    depth = imageio.imread(path).astype(np.float32) / 1000.0
    return depth


def pose_to_matrix(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def recursive_to_cuda(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, dict):
        return {k: recursive_to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cuda(v) for v in obj]
    else:
        return obj


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
            'BATCH_SIZE': 4,
            'NUM_WORKERS': 4,
            'SAMPLER': 'scene_balance',
            'N_SAMPLES_SCENE': 4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter': False, 'bright': False, 'hflip': False, 'vflip': False}
        },
        'test': {'mask': 'oracle', 'add_description': 'yes'},
        'use_seed': True,
        'seed': seed,
        'debug_valid': 'anchor'
    }
    return args_dict, dataset_name

# ---------------------- Main Function ----------------------
def main(args):
    # Load and merge configs
    with open(args.config, "r") as f:
        yaml_cfg = CN.load_cfg(f)
    cfg = default_cfg.clone()
    cfg.merge_from_other_cfg(yaml_cfg)

    args_dict, dataset_name = get_dataset_args_dict(args.dataset, args.dataset_root, seed=cfg.DATASET.SEED or 42)
    args_dict = DictConfig(args_dict)
    datamodule = DataModule(args_dict, dataset_name)
    loader = datamodule.val_dataloader() if args.split == 'val' else datamodule.test_dataloader()

    model = build_model(cfg, args.checkpoint)

    # Load GT and models info
    with open(ANNOTATION_PATH, 'rb') as f:
        annots_all = pickle.load(f)
    with open(MODELS_INFO_PATH, 'r') as f:
        models_info = json.load(f)
    model_diameters = {k: v['diameter'] for k, v in models_info.items()}
    with open(CATEGORIES_PATH, 'r') as f:
        category_id_to_name = json.load(f)
    name_to_category_id = {v: k for k, v in category_id_to_name.items()}

    renderer = RendererVispy(640, 480, mode='depth')
    models_cache = {}
    for obj_name in model_diameters.keys():
        try:
            model_data = get_obj_rendering(MODELS_ROOT, obj_name)
            renderer.my_add_object(model_data, obj_name)
            models_cache[obj_name] = model_data
        except:
            continue

    # Evaluation results
    adds_accurate, mssd_all, mspd_all, vsd_all = [], [], [], []

    for data in tqdm(loader, desc='Evaluating'):
        data = recursive_to_cuda(data)
        #print(data)
        with torch.no_grad():
            R_pred_batch, t_pred_batch = model(data)

        B = R_pred_batch.shape[0]
        for i in range(B):
            instance_id = data['instance_id'][i]
            fixed_id = fix_instance_id(instance_id, name_to_category_id)
            if fixed_id not in annots_all:
                continue

            # Get absolute anchor pose
            T_a = data['item_a_pose'][i].detach().cpu().numpy()
            T_rel_gt = data['pose'][i].detach().cpu().numpy()
            T_rel_pred = pose_to_matrix(R_pred_batch[i].detach().cpu().numpy(), t_pred_batch[i].detach().cpu().numpy())

            T_gt = T_rel_gt @ T_a
            T_pred = T_rel_pred @ T_a

            obj_name = '_'.join(fixed_id.split('_')[5:])
            scene_q, img_q = fixed_id.split('_')[2], fixed_id.split('_')[3]

            try:
                model_data = models_cache[obj_name]
                model_pts = model_data['pts'] / 1000.0
                depth = load_depth(scene_q, img_q)
            except:
                continue

            diameter = model_diameters.get(obj_name, None)
            if diameter is None:
                continue

            symmetries = get_symmetries(models_info, obj_name)
            if symmetries.shape[0] == 0:
                error = pose_error.add(T_pred[:3, :3], T_pred[:3, 3], T_gt[:3, :3], T_gt[:3, 3], model_pts)
            else:
                error = pose_error.adi(T_pred[:3, :3], T_pred[:3, 3], T_gt[:3, :3], T_gt[:3, 3], model_pts)
            adds_accurate.append(error < ADD_THRESHOLD * diameter)

            pred_R, pred_t = T_pred[:3, :3], T_pred[:3, 3:4]  ##* 1000
            gt_R, gt_t = T_gt[:3, :3], T_gt[:3, 3:4]  ##* 1000

            print("pred_t:",T_pred)
            print("gt_t:",gt_t)
            if symmetries.shape[0] == 0:
                symmetries = np.eye(4, dtype=np.float32)[None, ...]

            mssd_all.append(np.mean(my_mssd(pred_R, pred_t, gt_R, gt_t, model_pts, symmetries) < MSSD_REC * diameter))
            mspd_all.append(np.mean(my_mspd(pred_R, pred_t, gt_R, gt_t, K, model_pts, symmetries) < MSPD_REC))
            vsd_score = np.mean(np.asarray(vsd(pred_R, pred_t, gt_R, gt_t, depth, K, VSD_DELTA, VSD_REC, True, diameter, renderer, obj_name)) < VSD_REC[:, None])
            vsd_all.append(vsd_score)

    # Report
    def report(name, values):
        print(f'{name} Accuracy: {100. * np.mean(values):.2f}%' if values else f'{name} Accuracy: NaN')

    #report('ADD(S)-0.1d', adds_accurate)
    print(f'ADD(S)-{ADD_THRESHOLD}d Accuracy: {100. * np.mean(adds_accurate):.2f}%')
    report('MSSD', mssd_all)
    report('MSPD', mspd_all)
    report('VSD', vsd_all)


# ---------------------- Run ----------------------
if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/MicKey/curriculum_learning_warm_up.yaml')
    parser.add_argument('--checkpoint', default='weights/temp_48*48_480_mask_poseERR/pose0.34vcre0.51/checkpoints/epoch=349-best_poseval_AUC_pose/auc_pose=0.342.ckpt')
    parser.add_argument('--dataset', choices=['Shapenet6D', 'NOCS', 'TOYL'], default='NOCS')
    parser.add_argument('--dataset_root', default='filesOfOryon/data')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    args = parser.parse_args()
    main(args)
