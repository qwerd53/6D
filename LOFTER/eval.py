import torch
from tqdm import tqdm
from src.loftr import LoFTR, default_cfg
import os, sys
sys.path.append(os.getcwd())
from lib.datasets.datamodules import DataModule
from omegaconf import DictConfig
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

# 初始化 LoFTR
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("LOFTER/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# 数据集设置
def get_dataset_args_dict(dataset_name: str, root_path: str, seed: int = 42):
    obj_id, name = 'all', dataset_name
    return {
        'dataset': {
            'root': root_path,
            'img_size': [384, 384],
            'max_corrs': 4,
            'train': {'name': name, 'split': 'train', 'obj': obj_id},
            'test': {'name': name, 'split': 'val', 'obj': obj_id}
        },
        'TRAINING': {
            'BATCH_SIZE': 4,
            'NUM_WORKERS': 8,
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

# 回投影函数，2D关键点 + 深度 + 内参 -> 3D点
def backproject(kpts, depth, K):
    N = len(kpts)
    pts3d_full = np.full((N, 3), np.nan, dtype=np.float32)  # 用 NaN 占位

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


# Kabsch 算法，求刚体变换
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

# 平移向量欧几里得距离
def translation_euclidean_distance(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt)

# 旋转矩阵余弦相似度
def rotation_cosine_similarity(R_pred, R_gt):
    # Flatten matrix to vector
    r_pred = R_pred.flatten()
    r_gt = R_gt.flatten()
    cos_sim = np.dot(r_pred, r_gt) / (np.linalg.norm(r_pred) * np.linalg.norm(r_gt) + 1e-8)
    return np.clip(cos_sim, -1.0, 1.0)

if __name__ == '__main__':
    from omegaconf import OmegaConf

    dataset_root = 'filesOfOryon/data'
    dataset_name = 'NOCS'
    args_dict, dataset_name = get_dataset_args_dict(dataset_name, dataset_root)
    datamodule = DataModule(DictConfig(args_dict), dataset_name)
    datamodule.setup(stage='fit')

    dataloader = datamodule.val_dataloader()

    #
    # batch_dict = {...}

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        img0_rgb, img1_rgb = batch['image0'].cuda(), batch['image1'].cuda()  # [B, 3, H, W]


        def rgb_to_gray(tensor_rgb):
            gray = (0.2989 * tensor_rgb[:, 0, :, :] +
                    0.5870 * tensor_rgb[:, 1, :, :] +
                    0.1140 * tensor_rgb[:, 2, :, :])
            return gray.unsqueeze(1)  # [B, 1, H, W]


        img0_gray = rgb_to_gray(img0_rgb)
        img1_gray = rgb_to_gray(img1_rgb)

        # 保留目标物体区域
        if 'mask0_gt' in batch and 'mask1_gt' in batch and 'metadata0' in batch and 'metadata1' in batch:
            mask0_gt = batch['mask0_gt'].clone().float()  # [B, H, W]
            mask1_gt = batch['mask1_gt'].clone().float()  # [B, H, W]

            mask_ids0 = [m['mask_ids'][0] for m in batch['metadata0']]
            mask_ids1 = [m['mask_ids'][0] for m in batch['metadata1']]

            for i in range(mask0_gt.shape[0]):
                mask0_gt[i] = torch.where(mask0_gt[i] == mask_ids0[i], 1.0, 0.0)
                mask1_gt[i] = torch.where(mask1_gt[i] == mask_ids1[i], 1.0, 0.0)

            mask0_gt = F.interpolate(mask0_gt.unsqueeze(1).float(), size=img0_gray.shape[-2:], mode='nearest')
            mask1_gt = F.interpolate(mask1_gt.unsqueeze(1).float(), size=img0_gray.shape[-2:], mode='nearest')
            # 非目标物体区域置为0
            img0_gray = img0_gray * mask0_gt.to(img0_gray.device)  # [B,1,H,W]
            img1_gray = img1_gray * mask1_gt.to(img1_gray.device)

        batch_size = img0_gray.shape[0]
        for i in range(batch_size):
            img0_single = img0_gray[i:i + 1]
            img1_single = img1_gray[i:i + 1]

            match_batch = {'image0': img0_single, 'image1': img1_single}

            with torch.no_grad():
                matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].cpu().numpy()  # [N, 2]
            mkpts1 = match_batch['mkpts1_f'].cpu().numpy()  # [N, 2]
            mconf = match_batch['mconf'].cpu().numpy()  # [N]

            print(f"Batch {batch_idx} Pair {i}: matched {len(mkpts0)} keypoints")

            if len(mconf) < 3:
                print(f"Batch {batch_idx} Pair {i}: less than 3 matches, skip pose estimation.")
                continue

            # 深度图、相机内参和GT姿态
            depth0 = batch['depth0'][i].cpu().numpy()
            depth1 = batch['depth1'][i].cpu().numpy()
            K0 = batch['K_color0'][i].cpu().numpy()
            K1 = batch['K_color1'][i].cpu().numpy()
            pose_gt = batch['pose'][i].cpu().numpy()  # 4x4

            # 全部回投影,筛选有效,筛Top3
            pts3d_0, valid0 = backproject(mkpts0, depth0, K0)
            pts3d_1, valid1 = backproject(mkpts1, depth1, K1)
            valid = valid0 & valid1

            if np.count_nonzero(valid) < 3:
                print(f"Batch {batch_idx} Pair {i}: Not enough valid 3D points after filtering.")
                continue

            # 从有效匹配中取出对应的置信度和3D点
            valid_mconf = mconf[valid]
            valid_pts3d_0 = pts3d_0[valid]
            valid_pts3d_1 = pts3d_1[valid]

            #
            top3_idx = np.argsort(valid_mconf)[-8:]
            pts3d_0_top3 = valid_pts3d_0[top3_idx]
            pts3d_1_top3 = valid_pts3d_1[top3_idx]

            # 估计位姿
            R_pred, t_pred = kabsch_umeyama(pts3d_0_top3, pts3d_1_top3)

            #vis
            os.makedirs('LOFTER/results/visual_matches', exist_ok=True)

            # 可视化匹配图像
            img0_np = batch['image0'][i].permute(1, 2, 0).cpu().numpy() * 255
            img1_np = batch['image1'][i].permute(1, 2, 0).cpu().numpy() * 255
            img0_np = img0_np.astype(np.uint8)
            img1_np = img1_np.astype(np.uint8)

            # Resize to same height if needed
            if img0_np.shape[:2] != img1_np.shape[:2]:
                img1_np = cv2.resize(img1_np, (img0_np.shape[1], img0_np.shape[0]))

            # 拼接
            concat_img = np.concatenate([img0_np, img1_np], axis=1)
            concat_img = np.ascontiguousarray(concat_img)
            # 匹配线
            for idx in top3_idx:
                pt0 = tuple(np.round(mkpts0[valid][idx]).astype(int))
                pt1 = tuple(np.round(mkpts1[valid][idx]).astype(int) + np.array([img0_np.shape[1], 0]))  # 右图需偏移
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(concat_img, pt0, 5, color, -1)
                cv2.circle(concat_img, pt1, 5, color, -1)
                cv2.line(concat_img, pt0, pt1, color, 2)

            # 保存图像
            save_path = f'LOFTER/results/visual_matches/match_b{batch_idx}_p{i}.jpg'
            cv2.imwrite(save_path, concat_img[:, :, ::-1])  # BGR -> RGB for saving
            print(f"Saved match visualization to {save_path}")



            R_gt = pose_gt[:3, :3]
            t_gt = pose_gt[:3, 3]*1000
            print("t_gt(pose):",t_gt)
            print("t_pred:", t_pred)

            # print("item_a_pose:\n",batch['item_a_pose'][i])#.shape) (B,4,4)
            # print("item_q_pose:\n", batch['item_q_pose'][i])  # .shape) (B,4,4)

            #gt_R, gt_t = T_gt[:3, :3], T_gt[:3, 3:4] * 1000

            rot_cos_sim = rotation_cosine_similarity(R_pred, R_gt)
            trans_dist = translation_euclidean_distance(t_pred, t_gt)

            print(
                f"Batch {batch_idx} Pair {i}: "
                f"Rotation cosine similarity: {rot_cos_sim:.3f} | "
                f"Translation L2 distance: {trans_dist:.3f} mm"
            )
            print('')
