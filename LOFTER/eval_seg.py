import torch
from tqdm import tqdm
from src.loftr import LoFTR, default_cfg
import os, sys

sys.path.append(os.getcwd())
from lib.datasets.datamodules import DataModule
#import filesOfOryon.datasets
from filesOfOryon.datasets import Shapenet6DDataset, NOCSDataset, TOYLDataset
from filesOfOryon.utils.geo6d import best_fit_transform_with_RANSAC
from filesOfOryon.utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver

from omegaconf import DictConfig
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KDTree  # ADD-S计算
from filesOfOryon.utils.metrics import *
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt



default_cfg['coarse']['temp_bug_fix'] =False
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("LOFTER/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

import cv2
import numpy as np

def adjust_intrinsics_for_resize(K, original_size=(640, 480), current_size=(640, 480)):
    """
    调整相机内参矩阵 K，缩放。

    Args:
        K: numpy.ndarray，形状为 (3, 3)，原始内参矩阵。
        original_size: tuple，原始图像尺寸，格式为 (width, height)。
        current_size: tuple，当前图像尺寸，格式为 (width, height)。

    Returns:
        numpy.ndarray，调整后的内参矩阵。
    """
    original_width, original_height = original_size
    current_width, current_height = current_size
    scale_x = current_width / original_width
    scale_y = current_height / original_height

    K_adj = K.copy()
    K_adj[0, 0] *= scale_x  # fx
    K_adj[0, 2] *= scale_x  # cx
    K_adj[1, 1] *= scale_y  # fy
    K_adj[1, 2] *= scale_y  # cy

    return K_adj


def get_dataset_args_dict(dataset_name: str, root_path: str, seed: int = 42):
    obj_id, name = 'all', dataset_name
    return {
        'dataset': {
            'root': root_path,
            #'img_size':[480,640], #[480,640],\
            'img_size': [224, 224],
            'max_corrs': 4,
            'train': {'name': name, 'split': 'train', 'obj': obj_id},
            'test': {'name': name, 'split': 'val', 'obj': obj_id}
        },
        'TRAINING': {
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'SAMPLER': 'scene_balance',
            'N_SAMPLES_SCENE': 4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {'rgb': {}, 'text': {}},
        'test': {'mask': 'oracle', 'add_description': 'yes'},
        'use_seed': True,
        'seed': seed,
        'debug_valid': 'anchor'
    }, dataset_name

def backproject(kpts, depth, K):
    N = len(kpts)
    pts3d_full = np.full((N, 3), np.nan, dtype=np.float32)

    x, y = kpts[:, 0], kpts[:, 1]
    x_int, y_int = x.round().astype(int), y.round().astype(int)
    valid_xy = (x_int >= 0) & (x_int < depth.shape[1]) & (y_int >= 0) & (y_int < depth.shape[0])

    x_int_valid, y_int_valid = x_int[valid_xy], y_int[valid_xy]
    x_valid, y_valid = x[valid_xy], y[valid_xy]
    z = depth[y_int_valid, x_int_valid]
    valid_z = z > 0
    final_valid_idx = np.where(valid_xy)[0][valid_z]

    x, y, z = x_valid[valid_z], y_valid[valid_z], z[valid_z]
    pts = np.linalg.inv(K) @ np.vstack([x * z, y * z, z])
    pts3d_full[final_valid_idx] = pts.T

    final_valid_mask = np.zeros(N, dtype=bool)
    final_valid_mask[final_valid_idx] = True
    return pts3d_full, final_valid_mask

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1,:] *= -1
        R_ = Vt.T @ U.T
    t_ = centroid_B - R_ @ centroid_A
    return R_, t_

def translation_euclidean_distance(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt)

def rotation_cosine_similarity(R_pred, R_gt):
    r_pred = R_pred.flatten()
    r_gt = R_gt.flatten()
    cos_sim = np.dot(r_pred, r_gt) / (np.linalg.norm(r_pred) * np.linalg.norm(r_gt) + 1e-8)
    return np.clip(cos_sim, -1.0, 1.0)

if __name__ == '__main__':
    from omegaconf import OmegaConf

    #  ('ransac' or 'pointdsc')
    solver_type = 'pointdsc'  # 或 'ransac'

    if solver_type == 'pointdsc':
        pointdsc_solver = get_pointdsc_solver("/data/WDY/mickey-main/pretrained_models/pointdsc", 'cuda')

    dataset_root = 'filesOfOryon/data'
    dataset_name = 'NOCS'
    args_dict, dataset_name = get_dataset_args_dict(dataset_name, dataset_root)

    dataset_args = DictConfig(args_dict)
    if dataset_name.lower() == 'nocs':
        dataset = NOCSDataset(dataset_args, eval=True)
    elif dataset_name.lower() == 'shapenet6d':
        dataset = Shapenet6DDataset(dataset_args, eval=True)
    elif dataset_name.lower() == 'toyl':
        dataset = TOYLDataset(dataset_args, eval=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    datamodule = DataModule(DictConfig(args_dict), dataset_name)
    datamodule.setup(stage='fit')

    dataloader = datamodule.val_dataloader()


    #cnt
    total_count = 0
    success_count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        img0_rgb, img1_rgb = batch['image0'].cuda().float(), batch['image1'].cuda().float()
        #print(img0_rgb.shape)
        obj_ids = batch['obj_id']
        item_a_pose = batch['item_a_pose'] #shape:(B,4,4)
        item_q_pose = batch['item_q_pose'] #shape:(B,4,4)

        #print("obj_id:",batch["obj_id"])
        def rgb_to_gray(tensor_rgb):
            gray = (0.2989 * tensor_rgb[:, 0, :, :] +
                    0.5870 * tensor_rgb[:, 1, :, :] +
                    0.1140 * tensor_rgb[:, 2, :, :])
            return gray.unsqueeze(1)

        img0_gray = rgb_to_gray(img0_rgb)#img0_gray.shape:([B, 1, H, W])
        img1_gray = rgb_to_gray(img1_rgb)
        #mask
        #mask0_gt.shape: ([B, 1, H, W])


        #if 'mask0_gt' in batch and 'mask1_gt' in batch and 'metadata0' in batch and 'metadata1' in batch:
        mask0_gt = batch['mask0_gt'].cuda().clone().float().unsqueeze(1)
        #mask0_gt.shape: ([B,1,H,W])
        #print("mask0_gt:",mask0_gt.shape)
        mask1_gt = batch['mask1_gt'].cuda().clone().float().unsqueeze(1)
        ## 灰度图应用掩码过滤
        img0_gray= img0_gray * mask0_gt
        img1_gray= img1_gray * mask1_gt
        #print("img0_gray:",img0_gray)
            # #print(mask0_gt.shape)
            # mask_ids0 = [m['mask_ids'][0] for m in batch['metadata0']]
            # mask_ids1 = [m['mask_ids'][0] for m in batch['metadata1']]
            #
            # for i in range(mask0_gt.shape[0]):
            #     mask0_gt[i] = torch.where(mask0_gt[i] == mask_ids0[i], 1.0, 0.0)
            #     mask1_gt[i] = torch.where(mask1_gt[i] == mask_ids1[i], 1.0, 0.0)

            # mask0_gt = F.interpolate(mask0_gt.unsqueeze(1), size=img0_gray.shape[-2:], mode='nearest') #shape:([B, 1, H, W])
            # mask1_gt = F.interpolate(mask1_gt.unsqueeze(1), size=img1_gray.shape[-2:], mode='nearest')

        batch_size = img0_gray.shape[0]
        for i in range(batch_size):
            match_batch = {'image0': img0_gray[i:i+1], 'image1': img1_gray[i:i+1]}
            with torch.no_grad():
                matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].cpu().numpy()
            mkpts1 = match_batch['mkpts1_f'].cpu().numpy()
            # print("mkpts0.shape:",mkpts0.shape) # print("mkpts1.shape:", mkpts1.shape) # ##print("mkpts0:", mkpts0)
            mconf = match_batch['mconf'].cpu().numpy()

            # if len(mkpts0) < 3:
            #     continue

            #mask
            mask0_i = mask0_gt[i, 0].cpu().numpy()
            mask1_i = mask1_gt[i, 0].cpu().numpy()
            in_mask = (mask0_i[mkpts0[:, 1].round().astype(int), mkpts0[:, 0].round().astype(int)] > 0) & \
                      (mask1_i[mkpts1[:, 1].round().astype(int), mkpts1[:, 0].round().astype(int)] > 0)
            mkpts0 = mkpts0[in_mask]
            mkpts1 = mkpts1[in_mask]
            mconf = mconf[in_mask]



            print(f"Total matches: {len(match_batch['mkpts0_f'])}, after mask filtering: {len(mkpts0)}")

            depth0 = batch['depth0'][i].cpu().numpy()
            depth1 = batch['depth1'][i].cpu().numpy() #print(depth0,depth0.shape)
            K0 = batch['K_color0'][i].cpu().numpy()   #(3,3)
            K1 = batch['K_color1'][i].cpu().numpy()
            current_height, current_width = batch['image0'][i].shape[1:]

            K0 = adjust_intrinsics_for_resize(K0, original_size=(640, 480),
                                              current_size=(current_width, current_height))
            K1 = adjust_intrinsics_for_resize(K1, original_size=(640, 480),
                                              current_size=(current_width, current_height))

            #print(K0,K1)
            pose_gt = batch['pose'][i].cpu().numpy()
            #print("K0,K1",K0,K1)


            #print("depth0.shape:", depth0.shape)#(H,W)
            pts3d_0, valid0 = backproject(mkpts0, depth0, K0)
            pts3d_1, valid1 = backproject(mkpts1, depth1, K1)
            valid = valid0 & valid1

            if np.count_nonzero(valid) < 3:

                continue


            #confidence
            top_idx = np.argsort(mconf[valid])#[-9:]
            pts3d_0_top = pts3d_0[valid][top_idx]
            pts3d_1_top = pts3d_1[valid][top_idx]

            #topvalid
            #R_pred, t_pred = kabsch_umeyama(pts3d_0_top, pts3d_1_top) #top

            #valid
            pts3d_0_valid = pts3d_0[valid]
            pts3d_1_valid = pts3d_1[valid]

            #getpose
            R_pred, t_pred = kabsch_umeyama(pts3d_0_valid, pts3d_1_valid) #kabsch
            # pts3d_0_valid_tensor = torch.from_numpy(pts3d_0_valid).float().cuda()
            # pts3d_1_valid_tensor = torch.from_numpy(pts3d_1_valid).float().cuda()
            #
            # if solver_type == 'ransac':
            #     pose = best_fit_transform_with_RANSAC(pts3d_0_valid, pts3d_1_valid,
            #                                           max_iter=10000, fix_percent=0.9999, match_err=0.001)
            #     R_pred = pose[:3, :3]
            #     t_pred = pose[:3, 3]
            # elif solver_type == 'pointdsc':
            #     pose = get_pointdsc_pose(pointdsc_solver, pts3d_0_valid_tensor, pts3d_1_valid_tensor, 'cuda')
            #     R_pred = pose[:3, :3].cpu().numpy()
            #     t_pred = pose[:3, 3].cpu().numpy()

            #vonvert
            T_a = item_a_pose[i].cpu().numpy()
            T_q_gt = item_q_pose[i].cpu().numpy()

            T_rel_pred = np.eye(4)
            T_rel_pred[:3, :3] = R_pred
            T_rel_pred[:3, 3] = t_pred/1000.0

            T_q_pred = T_rel_pred @ T_a
            R_pred = T_q_pred[:3, :3]
            t_pred = T_q_pred[:3, 3] # m
            R_gt = T_q_gt[:3, :3]
            t_gt = T_q_gt[:3, 3] # m
            print(" t_pred:", t_pred)
            print(" t_gt:", t_gt)

            #add
            # 模型点云、直径和对称性
            obj_model, obj_diam, obj_sym = dataset.get_obj_info(obj_ids[i])
            pts3d_model = obj_model['pts'] / 1000.  # 单位 m

            # 创建4x4姿态矩阵
            pose_pred = np.eye(4)
            pose_pred[:3, :3] = R_pred
            pose_pred[:3, 3] = t_pred

            pose_gt = np.eye(4)
            pose_gt[:3, :3] = R_gt
            pose_gt[:3, 3] = t_gt

            # 计算ADD/ADD-S
            if len(obj_sym) > 0:  # 对称
                add_metric = compute_adds(pts3d_model, pose_pred, pose_gt)
                metric_name = "ADD-S"
            else:  # 非对称
                add_metric = compute_add(pts3d_model, pose_pred, pose_gt)
                metric_name = "ADD"

            # 阈值 = 0.1 x 物体直径（单位：m）
            threshold = 0.1 * obj_diam / 1000.0
            success = add_metric < threshold

            print(
                f"{metric_name}: {add_metric:.4f} m | "
                f"Threshold (0.1D): {threshold:.4f} m | "
                f"Success: {success}"
            )

            #cnt
            total_count += 1
            if success:
                success_count += 1


            #os.makedirs('LOFTER/results/visual_matches', exist_ok=True)
            # 可视化匹配图像
            # img0_np = batch['image0'][i].permute(1, 2, 0).cpu().numpy() * 255
            # img1_np = batch['image1'][i].permute(1, 2, 0).cpu().numpy() * 255
            # img0_np = img0_np.astype(np.uint8)
            # img1_np = img1_np.astype(np.uint8)
            #
            # # Resize to same height if needed
            # if img0_np.shape[:2] != img1_np.shape[:2]:
            #     img1_np = cv2.resize(img1_np, (img0_np.shape[1], img0_np.shape[0]))
            #
            # # 拼接图像
            # concat_img = np.concatenate([img0_np, img1_np], axis=1)
            # concat_img = np.ascontiguousarray(concat_img)
            #
            # # 只可视化 top 匹配点
            # for idx in top_idx:
            #     pt0 = tuple(np.round(mkpts0[idx]).astype(int))
            #     pt1 = tuple(np.round(mkpts1[idx]).astype(int) + np.array([img0_np.shape[1], 0]))  # 右图横坐标偏移
            #     color = tuple(np.random.randint(0, 255, 3).tolist())
            #     cv2.circle(concat_img, pt0, 5, color, -1)
            #     cv2.circle(concat_img, pt1, 5, color, -1)
            #     cv2.line(concat_img, pt0, pt1, color, 2)
            #
            # # 保存图像
            # save_path = f'LOFTER/results/visual_matches/match_b{batch_idx}_p{i}.jpg'
            # cv2.imwrite(save_path, concat_img[:, :, ::-1])  # BGR->RGB
            # print(f"Saved match visualization to {save_path}")


            # R_gt = pose_gt[:3, :3]
            # t_gt = pose_gt[:3, 3] * 1000

            rot_cos_sim = rotation_cosine_similarity(R_pred, R_gt)
            trans_dist = translation_euclidean_distance(t_pred, t_gt)

            # print(f"Batch {batch_idx} Pair {i}: Rotation cosine similarity: {rot_cos_sim:.3f} | Translation L2 distance: {trans_dist:.3f} m")
            # print("")

    #cnt
    print("=" * 50)
    if total_count > 0:
        success_rate = 100.0 * success_count / total_count
        print(f"Total evaluated: {total_count}")
        print(f"Success under 0.1D threshold: {success_count}")
        print(f"ADD(S)-0.1d Success Rate: {success_rate:.2f}%")
    else:
        print("No valid predictions to evaluate.")

