import torch
import pytorch_lightning as pl

from lib.models.MicKey.modules.loss.loss_class import MetricPoseLoss
from lib.models.MicKey.modules.compute_correspondences import ComputeCorrespondences
from lib.models.MicKey.modules.utils.training_utils import log_image_matches, debug_reward_matches_log, vis_inliers, \
    log_mask_images
from lib.models.MicKey.modules.utils.probabilisticProcrustes import e2eProbabilisticProcrustesSolver

from lib.utils.metrics import pose_error_torch, vcre_torch
from lib.benchmarks.utils import precision_recall
from lib.models.Oryon.oryon import Oryon
# metric
from filesOfOryon.bop_toolkit_lib.pose_error import my_mssd, my_mspd, vsd
from filesOfOryon.bop_toolkit_lib.renderer_vispy import RendererVispy
from filesOfOryon.utils.pcd import get_diameter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import os

import numpy as np
# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from filesOfOryon.bop_toolkit_lib.misc import format_sym_set
import numpy as np
from omegaconf import OmegaConf

# ==== 外部依赖 ====
from lib.models.Oryon.oryon import Oryon

# LoFTR
from LOFTER.src.loftr import LoFTR, default_cfg

# 你工程内已有的工具
from lib.utils.metrics import pose_error_torch  # 仅用于可选对齐检查（未用于loss）
from lib.benchmarks.utils import precision_recall  # 日志需要的话可以继续用
from filesOfOryon.utils.metrics import compute_add, compute_adds  # 用于 ADD/ADD-S
from filesOfOryon.utils.geo6d import best_fit_transform_with_RANSAC  # 可选的RANSAC
# from filesOfOryon.utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver  # 如需PointDSC就打开
from filesOfOryon.utils.losses import DiceLoss, LovaszLoss, FocalLoss


# from lib.models.MicKey.debug_loftr import debug_loftr
# =========================
#   MicKeyTrainingModel
# =========================
def sharpen_binary_mask(mask_bin, alpha=1.0):
    """
    mask_bin: [B,H,W] float 0/1 二值图
    alpha: 边缘扩张强度（0.5~2.0）
    """
    B, H, W = mask_bin.shape[0], mask_bin.shape[1], mask_bin.shape[2]

    # 1) Sobel 或 Laplacian 获取二值 mask 的边缘
    lap_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32,
                              device=mask_bin.device).unsqueeze(0).unsqueeze(0)

    edge = F.conv2d(mask_bin.unsqueeze(1), lap_kernel, padding=1).abs()  # [B,1,H,W]
    edge = (edge > 0.1).float()  # 二值化边缘

    # 2) 扩张边缘（增强锐化效果）
    dilate_kernel = torch.ones((1, 1, 3, 3), device=mask_bin.device)
    edge_dilate = F.conv2d(edge, dilate_kernel, padding=1)
    edge_dilate = (edge_dilate > 0).float()

    # 3) 将边缘加回 mask（锐化）
    sharpened = mask_bin.unsqueeze(1) + alpha * edge_dilate
    sharpened = (sharpened > 0.5).float()  # 再次二值化保持干净

    return sharpened.squeeze(1)


class MicKeyTrainingModel(pl.LightningModule):
    """
    精简后的训练模型：
    - 仅保留 Oryon 产生 mask 的模块
    - LoFTR 进行匹配 + 掩码过滤 + 回投影 → Kabsch 求姿态
    - 损失：mask_loss（BCEWithLogits） + compute_pose_loss（rot_angle + trans_l1，可选tanh clipping）
    - 每半个 epoch 跑一次 ADD(S)-0.1D 评估并记录到 TensorBoard
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg

        # ---------- Oryon ----------
        self.oryon_model = Oryon(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')

        # ---------- LoFTR ----------
        # # 你可在 cfg.LOFTR.WEIGHTS 指定权重路径；否则回退到默认路径
        # loftr_weights = getattr(getattr(cfg, 'LOFTR', {}), 'WEIGHTS', 'LOFTER/weights/outdoor_ds.ckpt')
        # default_cfg['coarse']['temp_bug_fix'] = False
        # self.matcher = LoFTR(config=default_cfg)
        # state = torch.load(loftr_weights, map_location='cpu')
        # self.matcher.load_state_dict(state['state_dict'])
        # self.matcher = self.matcher.eval()  # LoFTR 推理模式

        default_cfg['coarse']['temp_bug_fix'] = False
        self.matcher = LoFTR(config=default_cfg)
        self.matcher.load_state_dict(torch.load("LOFTER/weights/outdoor_ds.ckpt")['state_dict'])
        self.matcher = self.matcher.eval().cuda()

        # ---------- 损失 ----------
        # self._mask_loss = nn.BCEWithLogitsLoss(reduction='mean')  # 用 logits
        self._mask_loss = DiceLoss(weight=torch.tensor([0.5, 0.5]))
        self.mask_th = getattr(getattr(cfg, 'LOSS', {}), 'MASK_TH', 0.5)
        self.soft_clip = getattr(getattr(cfg, 'LOSS', {}), 'SOFT_CLIP', True)

        # ---------- 训练控制 ----------
        self.automatic_optimization = True  # Lightning 自动优化
        self.multi_gpu = True
        self.validation_step_outputs = []
        self.log_interval = getattr(cfg.TRAINING, 'LOG_INTERVAL', 50)

        # 半 epoch 评估控制
        self._ran_half_eval_for_epoch = False
        self._half_epoch_batch_idx = None  # 每个 epoch 开头计算

        # VSD渲染器
        self.compute_vsd = True
        self.vsd_renderer = RendererVispy(640, 480, mode='depth')
        self.vsd_taus = list(np.arange(0.05, 0.51, 0.05))
        self.vsd_rec = np.arange(0.05, 0.51, 0.05)
        self.vsd_delta = 15.
        self._vsd_objects_loaded = False

        # 初始化指标记录
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.useGTmask = False

        self.debug_loftr_flag = False  # 调试开关
        self.top_k_matches = 8  # =8 for lm and ycbv
        if (self.top_k_matches == 0):
            print("all valid points to 3d")
        else:
            print("top conf_valid points to 3d num:", self.top_k_matches)
        # ... (原有代码)
        self.debug_mask_guided_flag = False  # 关闭之前的调试

        # [NEW] 开启 2D-3D 配准与对齐可视化 (Section 2.4)
        self.debug_registration_flag = False

        self.debug_boundingbox_flag = True
        self.debug_cad_rendering_flag = False  # CAD 模型渲染可视化
        self.debug_maps = False  # t-SNE 特征分布可视化

    def _load_vsd_objects(self):
        if self._vsd_objects_loaded:
            return

        # 从 datamodule 获取所有 obj 信息
        obj_models, obj_diams, obj_symms = self.trainer.datamodule.val_dataloader().dataset.get_object_info()
        self.add_object_info(obj_models, obj_diams, obj_symms)
        self._vsd_objects_loaded = True

        print("Loaded VSD objects:", list(self.vsd_renderer.model_bbox_corners.keys()))

    def forward(self, batch):
        return self.forward_once(batch)

    # -------------------------
    #   = = = 关键 Loss = = =
    # -------------------------
    def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor):
        """
        pred_logits: [B,1,H_pred,W_pred] — 掩码 logits
        gt:          [B,H_gt,W_gt]       — ground truth binary mask

        返回: loss, pred_mask(0/1), pred_logits, IoU
        """
        # print("(pred.shape:",pred_logits.shape)
        # print("(gt.shape:",gt.shape)
        # if pred_logits.dim() == 3:
        #     pred =pred_logits.unsqueeze(1)  # [B, 1, H, W]
        # if gt.dim() == 3:
        #     gt = gt.unsqueeze(1)  # [B, 1, H, W]

        # pred_logits.shape)
        gt_shape = gt.shape[1:]
        pred_shape = pred_logits.shape[2:]
        # print("pred_shape", pred_shape)
        # print("gt_shape", gt_shape)
        gt_c = gt.clone().to(torch.float32)
        if gt_shape != pred_shape:
            gt_c = F.interpolate(gt.unsqueeze(1), size=pred_shape, mode='nearest').squeeze(1)

        if gt_c.max() > 1.0:
            gt_c = gt_c / 255.0

        logits = pred_logits.squeeze(1)  # [B, H, W]
        loss = self._mask_loss(logits, gt_c.to(torch.float32))

        with torch.no_grad():
            pred_mask = (torch.sigmoid(logits) > self.mask_th).float()
            intersection = (pred_mask * gt_c).sum(dim=(1, 2))
            union = (pred_mask + gt_c - pred_mask * gt_c).sum(dim=(1, 2)) + 1e-6
            iou = (intersection / union).mean()

        return loss, pred_mask, logits, iou

    def compute_pose_loss(self, R, t, Rgt_i, tgt_i, soft_clipping=True):
        """
        与你给的 compute_pose_loss 一致：rot_angle_loss + trans_l1_loss（可 tanh soft clipping）
        R:    [B,3,3]
        t:    [B,1,3]
        Rgt:  [B,3,3]
        tgt:  [B,1,3]
        """
        loss_rot, _ = self.rot_angle_loss(R, Rgt_i)  # [B,1]
        loss_trans = self.trans_l1_loss(t, tgt_i)  # [B,1,3] -> [B,1]

        if soft_clipping:
            loss_trans_soft = torch.tanh(loss_trans / 0.9)
            loss_rot_soft = torch.tanh(loss_rot / 0.9)
            loss = loss_rot_soft + loss_trans_soft
        else:
            loss = loss_rot + loss_trans

        return loss.mean(), loss_rot.mean(), loss_trans.mean()

    # ---- 工具函数 ----
    @staticmethod
    def trans_l1_loss(t, tgt):
        return torch.abs(t - tgt).sum(-1)  # [B,1,3] -> [B,1]

    @staticmethod
    def rot_angle_loss(R, Rgt):
        residual = R.transpose(1, 2) @ Rgt
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        cosine = torch.clip(cosine, -0.99999, 0.99999)
        R_err = torch.acos(cosine)
        loss = torch.abs(R_err - torch.zeros_like(R_err)).unsqueeze(-1)
        return loss, R_err

    # -------------------------
    #     LoFTR + 后端求姿态
    # -------------------------
    @staticmethod
    def rgb_to_gray(tensor_rgb: torch.Tensor) -> torch.Tensor:
        gray = (0.2989 * tensor_rgb[:, 0, :, :] +
                0.5870 * tensor_rgb[:, 1, :, :] +
                0.1140 * tensor_rgb[:, 2, :, :])
        return gray.unsqueeze(1)

    @staticmethod
    def adjust_intrinsics_for_resize(K, original_size=(640, 480), current_size=(640, 480)):
        """K: [3,3],  size:(width,height)"""
        original_width, original_height = original_size
        current_width, current_height = current_size
        scale_x = current_width / original_width
        scale_y = current_height / original_height
        K_adj = K.copy()
        K_adj[0, 0] *= scale_x
        K_adj[0, 2] *= scale_x
        K_adj[1, 1] *= scale_y
        K_adj[1, 2] *= scale_y
        return K_adj

    @staticmethod
    def backproject(kpts, depth, K):
        """
        kpts: [N,2] (x,y), depth:[H,W], K:[3,3]
        return: pts3d_full:[N,3] (NaN填充), valid_mask:[N]
        """
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

    @staticmethod
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
            Vt[-1, :] *= -1
            R_ = Vt.T @ U.T
        t_ = centroid_B - R_ @ centroid_A
        return R_, t_

    # -------------------------
    #         前向部分
    # -------------------------
    def forward_once(self, batch):

        # 掩膜引导对比图调试
        if getattr(self, 'debug_mask_guided_flag', False):
            self.debug_mask_guided_comparison(batch, num_samples=10)
            print("Mask-guided comparison debug finished...")
            exit()

        if self.debug_loftr_flag:
            self.debug_loftr(batch)
            print("debug finished...")
            exit()

        """
        单次前向（用于 train/val/eval）
        """
        device = batch['image0'].device
        B, _, H, W = batch['image0'].shape

        # 1) Oryon 输出（假定返回 logits 形态）
        oryon_out = self.oryon_model.forward(batch)  # 包含 'mask_a', 'mask_q'
        pred_mask0_logits = oryon_out['mask_a']  # [B,1,Hm,Wm]
        pred_mask1_logits = oryon_out['mask_q']  # [B,1,Hm,Wm]

        # 2) mask loss
        mask0_gt = batch['mask0_gt']  # [B,H,W]
        mask1_gt = batch['mask1_gt']  # [B,H,W]

        # 2) mask loss
        mask0_gt = batch['mask0_gt']  # [B,H,W]
        mask1_gt = batch['mask1_gt']  # [B,H,W]

        # # -------------------------------
        # # ✅ 调试保存：前3组样本到 debug 目录
        # # -------------------------------
        # from PIL import Image
        # import torchvision.utils as vutils
        # import numpy as np
        # import torch
        # import os
        #
        # save_dir = "debug"
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # 只保存前三组，防止太多
        # num_save = min(3, B)
        #
        # # 把预测 logits 通过 sigmoid 变成 0~1 概率
        # pred_mask0 = torch.sigmoid(pred_mask0_logits)
        # pred_mask1 = torch.sigmoid(pred_mask1_logits)
        #
        # # 如果分辨率与GT不同，可先插值
        # if pred_mask0.shape[-2:] != mask0_gt.shape[-2:]:
        #     pred_mask0 = torch.nn.functional.interpolate(pred_mask0, size=(H, W), mode='bilinear', align_corners=False)
        #     pred_mask1 = torch.nn.functional.interpolate(pred_mask1, size=(H, W), mode='bilinear', align_corners=False)
        #
        # for i in range(num_save):
        #     img0 = batch['image0'][i].detach().cpu()  # [3,H,W]
        #     img1 = batch['image1'][i].detach().cpu()
        #     gt0 = mask0_gt[i].detach().cpu().numpy()  # [H,W]
        #     gt1 = mask1_gt[i].detach().cpu().numpy()
        #     pm0 = pred_mask0[i, 0].detach().cpu().numpy()  # [H,W]
        #     pm1 = pred_mask1[i, 0].detach().cpu().numpy()
        #
        #     # 转成 [0,255] uint8 掩码
        #     gt0_img = Image.fromarray((gt0 * 255).astype(np.uint8))
        #     gt1_img = Image.fromarray((gt1 * 255).astype(np.uint8))
        #     pm0_img = Image.fromarray((pm0 * 255).astype(np.uint8))
        #     pm1_img = Image.fromarray((pm1 * 255).astype(np.uint8))
        #
        #     # 保存原图和掩码
        #     vutils.save_image(img0, f"{save_dir}/image0_{i}.png", normalize=True)
        #     vutils.save_image(img1, f"{save_dir}/image1_{i}.png", normalize=True)
        #     gt0_img.save(f"{save_dir}/mask0_gt_{i}.png")
        #     gt1_img.save(f"{save_dir}/mask1_gt_{i}.png")
        #     pm0_img.save(f"{save_dir}/mask0_pred_{i}.png")
        #     pm1_img.save(f"{save_dir}/mask1_pred_{i}.png")
        #
        #     # ⭐ 用 GT 掩码过滤原图（物体区域保留，其余全黑）
        #     mask0_tensor = torch.from_numpy(gt0).float().unsqueeze(0)  # [1,H,W]
        #     mask1_tensor = torch.from_numpy(gt1).float().unsqueeze(0)
        #     filtered0_gt = img0 * mask0_tensor
        #     filtered1_gt = img1 * mask1_tensor
        #     vutils.save_image(filtered0_gt, f"{save_dir}/image0_filtered_gt_{i}.png", normalize=True)
        #     vutils.save_image(filtered1_gt, f"{save_dir}/image1_filtered_gt_{i}.png", normalize=True)
        #
        #     # # ⭐ 用预测掩码过滤原图（查看预测区域效果）
        #     # mask0_pred_tensor = torch.from_numpy(pm0).float().unsqueeze(0)
        #     # mask1_pred_tensor = torch.from_numpy(pm1).float().unsqueeze(0)
        #     # filtered0_pred = img0 * mask0_pred_tensor
        #     # filtered1_pred = img1 * mask1_pred_tensor
        #     # vutils.save_image(filtered0_pred, f"{save_dir}/image0_filtered_pred_{i}.png", normalize=True)
        #     # vutils.save_image(filtered1_pred, f"{save_dir}/image1_filtered_pred_{i}.png", normalize=True)
        #
        # print(f"已保存 {num_save} 组图像与掩码到 {save_dir}/")

        if (self.useGTmask):
            mask0_gt = mask0_gt.unsqueeze(1)  # [B,1,H,W]
            mask1_gt = mask1_gt.unsqueeze(1)  # [B,1,H,W]
            mask0_loss, _, _, mask0_iou = self.mask_loss(mask0_gt, batch['mask0_gt'])
            mask1_loss, _, _, mask1_iou = self.mask_loss(mask1_gt, batch['mask1_gt'])
        else:
            mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, mask0_gt)
            mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, mask1_gt)
        mask_loss_all = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2.

        # 3) logits -> 概率
        pred_mask0_prob = torch.sigmoid(pred_mask0_logits).squeeze(1)  # [B,Hm,Wm]
        pred_mask1_prob = torch.sigmoid(pred_mask1_logits).squeeze(1)  # [B,Hm,Wm]

        # # ----🔥基于拉普拉斯算子的边缘锐化 ----
        # sharpen = LaplacianSharpen(alpha=1.0).to(pred_mask0_prob.device)
        # pred_mask0_prob = sharpen(pred_mask0_prob.unsqueeze(1)).squeeze(1)
        # pred_mask1_prob = sharpen(pred_mask1_prob.unsqueeze(1)).squeeze(1)

        # 4) resize 到输入图像大小
        pred_mask0_prob = F.interpolate(pred_mask0_prob.unsqueeze(1),
                                        size=(H, W), mode='bilinear',
                                        align_corners=False).squeeze(1)
        pred_mask1_prob = F.interpolate(pred_mask1_prob.unsqueeze(1),
                                        size=(H, W), mode='bilinear',
                                        align_corners=False).squeeze(1)

        # 5) 二值化
        pred_mask0_bin = (pred_mask0_prob > 0.5).float()
        pred_mask1_bin = (pred_mask1_prob > 0.5).float()
        # pred_mask0_bin = (pred_mask0_prob > 0.3).float()
        # pred_mask1_bin = (pred_mask1_prob > 0.3).float()

        # #Laplacian 检测边缘 + 膨胀，得到锐化边界
        # pred_mask0_bin = sharpen_binary_mask(pred_mask0_bin)
        # pred_mask1_bin = sharpen_binary_mask(pred_mask1_bin)

        # 6) pred灰度图过滤
        # img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
        # img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)

        if (self.useGTmask):

            ## 6) 灰度图过滤 - GT 掩码
            img0_gray = self.rgb_to_gray(batch['image0']) * batch['mask0_gt'].unsqueeze(1)
            img1_gray = self.rgb_to_gray(batch['image1']) * batch['mask1_gt'].unsqueeze(1)
        else:
            # pred灰度图过滤
            img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
            img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)

        # 7) LoFTR 匹配
        R_preds, t_preds = [], []
        for i in range(B):
            match_batch = {'image0': img0_gray[i:i + 1], 'image1': img1_gray[i:i + 1]}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()  # [N, 2]
            mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()  # [N, 2]
            mconf = match_batch['mconf'].detach().cpu().numpy()  # [N]
            # print("len(mkpts0 after loftr", len(mkpts0))

            if (self.useGTmask):
                # if使用 GT 掩码
                m0 = batch['mask0_gt'][i].detach().cpu().numpy()  # ★
                m1 = batch['mask1_gt'][i].detach().cpu().numpy()  # ★
            else:
                # if pred mask
                m0 = pred_mask0_bin[i].detach().cpu().numpy()
                m1 = pred_mask1_bin[i].detach().cpu().numpy()

            if len(mkpts0) == 0:
                print("error,len(mkpts0)<0 after loftr")
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))
                continue

            # # 按掩码过滤关键点
            # in_mask = (m0[mkpts0[:, 1].round().astype(int),
            # mkpts0[:, 0].round().astype(int)] > 0) & \
            #           (m1[mkpts1[:, 1].round().astype(int),
            #           mkpts1[:, 0].round().astype(int)] > 0)
            # mkpts0 = mkpts0[in_mask]
            # mkpts1 = mkpts1[in_mask]
            #
            # if len(mkpts0) < 3:
            #     print("error,len(mkpts0)<3 after filtering")
            #     R_preds.append(torch.eye(3, device=device))
            #     t_preds.append(torch.zeros(1, 3, device=device))
            #     continue

            # ✅ 按掩码过滤关键点
            in_mask = (m0[mkpts0[:, 1].round().astype(int),
            mkpts0[:, 0].round().astype(int)] > 0) & \
                      (m1[mkpts1[:, 1].round().astype(int),
                      mkpts1[:, 0].round().astype(int)] > 0)

            mkpts0 = mkpts0[in_mask]
            mkpts1 = mkpts1[in_mask]
            mconf = mconf[in_mask]  # ★ 同步过滤置信度
            # print("len(mkpts0 after mask2d:", len(mkpts0))
            # # ✅ 按置信度排序并取 Top-8
            # if len(mkpts0) >= 8:
            #     idx = np.argsort(-mconf)[:8]  # 从大到小
            #     mkpts0 = mkpts0[idx]
            #     mkpts1 = mkpts1[idx]
            #     mconf = mconf[idx]  # 保持一致
            # elif len(mkpts0) < 3:  # 仍不够用于Kabsch
            #     print("error,len(mkpts0)<3 after 2nd filtering")
            #     R_preds.append(torch.eye(3, device=device))
            #     t_preds.append(torch.zeros(1, 3, device=device))
            #     continue

            # 根据 self.top_k_matches 控制是否取 Top-K 置信度匹配点
            top_k = self.top_k_matches

            if top_k is not None and top_k > 0:
                # 置信度排序
                if len(mkpts0) >= top_k:
                    idx = np.argsort(-mconf)[:top_k]  # 取置信度最高 top_k 坐标
                    mkpts0 = mkpts0[idx]
                    mkpts1 = mkpts1[idx]
                    mconf = mconf[idx]
                elif len(mkpts0) < 3:  # 仍然不够用于 Kabsch（至少需要3点）
                    print(f"error: len(mkpts0)<3 after filtering, got {len(mkpts0)}")
                    R_preds.append(torch.eye(3, device=device))
                    t_preds.append(torch.zeros(1, 3, device=device))
                    continue
            else:
                # ✅ 不过滤
                if len(mkpts0) < 3:
                    print(f"error: len(mkpts0)<3 (no filtering mode), got {len(mkpts0)}")
                    R_preds.append(torch.eye(3, device=device))
                    t_preds.append(torch.zeros(1, 3, device=device))
                    continue

            # 8) 回投影到 3D
            depth0 = batch['depth0'][i].detach().cpu().numpy()  # /10.
            depth1 = batch['depth1'][i].detach().cpu().numpy()  # /10.
            # print("depth0:",depth0)
            K0 = batch['K_color0'][i].detach().cpu().numpy()
            K1 = batch['K_color1'][i].detach().cpu().numpy()

            current_h, current_w = batch['image0'][i].shape[1:]
            K0 = self.adjust_intrinsics_for_resize(K0, original_size=(640, 480),
                                                   current_size=(current_w, current_h))
            K1 = self.adjust_intrinsics_for_resize(K1, original_size=(640, 480),
                                                   current_size=(current_w, current_h))

            # print("K0",K0)
            # print("K1", K1)
            pts3d_0, valid0 = self.backproject(mkpts0, depth0, K0)
            pts3d_1, valid1 = self.backproject(mkpts1, depth1, K1)
            valid = valid0 & valid1
            if np.count_nonzero(valid) < 3:
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))
                continue

            A = pts3d_0[valid]
            Bp = pts3d_1[valid]

            # 9) Kabsch
            R_np, t_np = self.kabsch_umeyama(A, Bp)
            # R_t = torch.from_numpy(R_np).float().to(device)
            # t_t = torch.from_numpy(t_np).float().to(device).unsqueeze(0) / 1000.0

            # =========================================================
            # [NEW] Section 2.4: Point Clouds Registration Visualization
            # =========================================================
            if getattr(self, 'debug_registration_flag', False) and i < 5:  # 保存5个样本
                instance_id = batch['instance_id'][i] if 'instance_id' in batch else i
                self.debug_registration_alignment(
                    src_pts=A,  # Source Point Cloud (Camera 0)
                    tgt_pts=Bp,  # Target Point Cloud (Camera 1)
                    R_pred=R_np,  # Predicted Rotation
                    t_pred=t_np,  # Predicted Translation
                    sample_idx=i,
                    instance_id=instance_id,  # 传递 instance_id
                    image0=batch['image0'][i],  # 传递原图
                    image1=batch['image1'][i],  # 传递原图
                )
                print(f"Registration visualization saved for sample {i} (instance_id={instance_id}).")
                # exit() # 如果想画一张就停，取消注释
            # =========================================================

            # 10) 转到绝对位姿
            T_a = batch['item_a_pose'][i].detach().cpu().numpy()
            T_rel = np.eye(4, dtype=np.float32)
            T_rel[:3, :3] = R_np
            T_rel[:3, 3] = (t_np / 1000.0)
            T_q_pred = T_rel @ T_a
            R_q = torch.from_numpy(T_q_pred[:3, :3]).float().to(device)
            t_q = torch.from_numpy(T_q_pred[:3, 3]).float().to(device).unsqueeze(0)

            # =========================================================
            # [NEW] 3D Bounding Box Visualization
            # =========================================================
            if getattr(self, 'debug_boundingbox_flag', False) and i < 5:  # 保存5个样本
                instance_id = batch['instance_id'][i] if 'instance_id' in batch else i
                obj_id = batch['obj_id'][i] if 'obj_id' in batch else None

                # 获取相机内参
                K1 = batch['K_color1'][i].detach().cpu().numpy()
                current_h, current_w = batch['image1'][i].shape[1:]
                K1_adj = self.adjust_intrinsics_for_resize(K1, original_size=(640, 480),
                                                           current_size=(current_w, current_h))

                # 获取 GT 姿态
                T_q_gt = batch['item_q_pose'][i].detach().cpu().numpy()
                R_gt = T_q_gt[:3, :3]
                t_gt = T_q_gt[:3, 3]

                # ---- 🔍 调试打印 GT 和 Pred 姿态 ----
                print(f"\n{'=' * 60}")
                print(f"[DEBUG] Sample {i} Pose Comparison")
                print(f"{'=' * 60}")
                print(f"GT Rotation (R_gt):\n{R_gt}")
                print(f"\nGT Translation (t_gt): {t_gt}")
                print(f"\nPred Rotation (R_pred):\n{T_q_pred[:3, :3]}")
                print(f"\nPred Translation (t_pred): {T_q_pred[:3, 3]}")
                print(f"{'=' * 60}\n")

                self.debug_boundingbox_visualization(
                    image0=batch['image0'][i],  # Anchor image
                    image1=batch['image1'][i],  # Query image
                    R_pred=T_q_pred[:3, :3],  # Predicted rotation (absolute pose)
                    t_pred=T_q_pred[:3, 3],  # Predicted translation (absolute pose)
                    R_gt=R_gt,  # GT rotation
                    t_gt=t_gt,  # GT translation
                    K=K1_adj,  # Camera intrinsics for image1
                    obj_id=obj_id,  # Object ID for getting 3D model
                    sample_idx=i,
                    instance_id=instance_id,
                )
                print(f"Bounding box visualization saved for sample {i} (instance_id={instance_id}).")
                # exit() # 如果想画一张就停，取消注释
            # =========================================================

            # =========================================================
            # [NEW] CAD Model Rendering Visualization
            # =========================================================
            if getattr(self, 'debug_cad_rendering_flag', False) and i < 5:  # 保存5个样本
                instance_id = batch['instance_id'][i] if 'instance_id' in batch else i
                obj_id = batch['obj_id'][i] if 'obj_id' in batch else None

                # 获取相机内参
                K1 = batch['K_color1'][i].detach().cpu().numpy()
                current_h, current_w = batch['image1'][i].shape[1:]
                K1_adj = self.adjust_intrinsics_for_resize(K1, original_size=(640, 480),
                                                           current_size=(current_w, current_h))

                # 获取 GT 姿态
                T_q_gt = batch['item_q_pose'][i].detach().cpu().numpy()
                R_gt = T_q_gt[:3, :3]
                t_gt = T_q_gt[:3, 3]

                print(f"\n{'=' * 80}")
                print(f"[FORWARD_ONCE] Calling CAD rendering for sample {i}")
                print(f"{'=' * 80}")
                print(f"  instance_id: {instance_id}")
                print(f"  obj_id: {obj_id}")
                print(f"  Image size: {current_w}x{current_h}")
                print(f"  Original K (640x480):\n{K1}")
                print(f"  Adjusted K ({current_w}x{current_h}):\n{K1_adj}")
                print(f"  T_q_pred (4x4):\n{T_q_pred}")
                print(f"  T_q_gt (4x4):\n{T_q_gt}")
                print(f"  R_pred:\n{T_q_pred[:3, :3]}")
                print(f"  t_pred (m): {T_q_pred[:3, 3]}")
                print(f"  R_gt:\n{R_gt}")
                print(f"  t_gt (m): {t_gt}")

                self.debug_cad_rendering_visualization(
                    image0=batch['image0'][i],  # Anchor image
                    image1=batch['image1'][i],  # Query image
                    R_pred=T_q_pred[:3, :3],  # Predicted rotation (absolute pose)
                    t_pred=T_q_pred[:3, 3],  # Predicted translation (absolute pose)
                    R_gt=R_gt,  # GT rotation
                    t_gt=t_gt,  # GT translation
                    K=K1_adj,  # Camera intrinsics for image1
                    obj_id=obj_id,  # Object ID for getting 3D model
                    sample_idx=i,
                    instance_id=instance_id,
                )

                print(f"\n[FORWARD_ONCE] CAD rendering completed for sample {i}")
                print(f"{'=' * 80}\n")
                # exit() # 如果想画一张就停，取消注释
            # =========================================================

            R_preds.append(R_q)
            t_preds.append(t_q)

        # =========================================================
        # [NEW] t-SNE Feature Distribution Visualization
        # Section 2.3.1: Foreground-Focused Feature Extraction
        # =========================================================
        if getattr(self, 'debug_maps', False):
            from lib.models.MicKey.visualize_feature_tsne import visualize_feature_tsne

            num_save = min(5, B)
            for i in range(num_save):
                instance_id = batch['instance_id'][i] if 'instance_id' in batch else i

                visualize_feature_tsne(
                    matcher=self.matcher,
                    batch=batch,
                    pred_mask0_bin=pred_mask0_bin[i],
                    pred_mask1_bin=pred_mask1_bin[i],
                    sample_idx=i,
                    instance_id=instance_id,
                    save_dir="debug_feature_tsne",
                    n_samples=2000,
                    perplexity=30,
                    use_pca_init=True
                )

                print(f"✅ Sample {i} (instance_id={instance_id}) t-SNE visualization completed.")

            print(f"\n🎉 All {num_save} samples t-SNE visualizations saved!")
            exit()
        # =========================================================

        R_pred = torch.stack(R_preds, dim=0)
        t_pred = torch.stack(t_preds, dim=0)

        return {
            'R_pred': R_pred,
            't_pred': t_pred,
            'mask_loss': mask_loss_all,
            'mask_iou': mask_iou_mean,
            'mask0_loss': mask0_loss,
            'mask1_loss': mask1_loss,
        }

    # -------------------------
    #     Lightning Hooks
    # -------------------------
    def training_step(self, batch, batch_idx):
        # 兼容你之前的自定义 collate 字段
        if 'pose' in batch and 'T_0to1' not in batch:
            batch['T_0to1'] = batch['pose']

        # 前向 + 得到预测位姿
        out = self.forward_once(batch)

        # GT 绝对位姿（query）
        T_q_gt = batch['item_q_pose']  # [B,4,4]
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)  # [B,1,3]，单位 m

        # 姿态损失
        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )

        total_loss = out['mask_loss'] + pose_loss

        # ---- 日志 ----
        self.log('train/mask_loss', out['mask_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mask_iou', out['mask_iou'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_loss', pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/pose_rot_loss', pose_rot_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_trans_loss', pose_trans_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        # ---- 半个 epoch 触发 ADD(S)-0.1D 评估 ----
        if (self._half_epoch_batch_idx is not None
                and batch_idx >= self._half_epoch_batch_idx
                and not self._ran_half_eval_for_epoch):
            self._ran_half_eval_for_epoch = True
            with torch.no_grad():
                add_acc = self.run_add01d_eval()
            if add_acc is not None:
                self.log('eval_half_epoch/add01d_acc(%)', add_acc, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss

    # def on_train_epoch_start(self):
    #     """在每个 epoch 开头确定“半个 epoch”的 batch 索引，并重置开关。"""
    #     self._ran_half_eval_for_epoch = False
    #     try:
    #         # 估计本 epoch 的 train batch 数
    #         train_loader = self.trainer.datamodule.train_dataloader()
    #         n_batches = len(train_loader)
    #         self._half_epoch_batch_idx = max(0, (n_batches // 2) - 1)
    #     except Exception:
    #         self._half_epoch_batch_idx = None

    # def validation_step(self, batch, batch_idx):
    #     out = self.forward_once(batch)
    #
    #     T_q_gt = batch['item_q_pose']
    #     R_gt = T_q_gt[:, :3, :3]
    #     t_gt = T_q_gt[:, :3, 3].unsqueeze(1)
    #
    #     pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
    #         out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
    #     )
    #     total_loss = out['mask_loss'] + pose_loss
    #
    #     logs = {
    #         'loss': total_loss.detach(),
    #         'pose_loss': pose_loss.detach(),
    #         'pose_rot_loss': pose_rot_loss.detach(),
    #         'pose_trans_loss': pose_trans_loss.detach(),
    #         'mask_loss': out['mask_loss'].detach(),
    #         'mask_iou': out['mask_iou'].detach(),
    #     }
    #     self.validation_step_outputs.append(logs)
    #     return logs
    # -------------------------
    #   Validation Step
    # -------------------------
    def validation_step(self, batch, batch_idx):
        # 确保 VSD 对象已经加载
        self._load_vsd_objects()

        out = self.forward_once(batch)

        T_q_gt = batch['item_q_pose']
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)

        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )
        total_loss = out['mask_loss'] + pose_loss

        # 计算ADD(S)-0.1d
        add01d_acc = self.compute_add01d(batch, out['R_pred'], out['t_pred'])

        # 计算MSSD, MSPD, VSD
        mssd_scores, mspd_scores, vsd_scores = [], [], []

        for i in range(len(batch['obj_id'])):
            obj_id = batch['obj_id'][i]

            #  用初始化时保存的 obj 信息
            obj_model = self.obj_models[obj_id]
            obj_diam = self.obj_diams[obj_id]
            obj_sym = self.obj_symms[obj_id]

            # 安全处理：如果 obj_id 不在 renderer 内，跳过 VSD
            if obj_id not in self.vsd_renderer.model_bbox_corners:
                print(f"Warning: missing obj_id {obj_id} in renderer, skipping VSD")
                vsd_err = 0.0
            else:
                depth = batch['orig_depth1'][i].cpu().numpy()
                K = batch['K_color1'][i].cpu().numpy()

                pred_pose = torch.eye(4, device=self.device)
                pred_pose[:3, :3] = out['R_pred'][i]
                pred_pose[:3, 3] = out['t_pred'][i].squeeze()

                gt_pose = batch['item_q_pose'][i]
                # print(" pose_pred:", pred_pose)
                # print(" pose_gt:", gt_pose)

                # 转换为毫米
                pred_R = pred_pose[:3, :3].cpu().numpy()
                pred_t = (pred_pose[:3, 3] * 1000).cpu().numpy().reshape(3, 1)
                gt_R = gt_pose[:3, :3].cpu().numpy()
                gt_t = (gt_pose[:3, 3] * 1000).cpu().numpy().reshape(3, 1)

                # print("gt_pose",gt_pose)
                obj_sym = self.obj_symms.get(obj_id, None)
                # 处理好的 ndarray
                bop_sym = obj_sym

                # MSSD/MSPD
                # mssd_err = my_mssd(pred_R, pred_t, gt_R, gt_t, obj_model['pts'], bop_sym)
                # mspd_err = my_mspd(pred_R, pred_t, gt_R, gt_t, K, obj_model['pts'], bop_sym)
                mssd_err = my_mssd(pred_R, pred_t, gt_R, gt_t, obj_model['pts'], bop_sym)
                mspd_err = my_mspd(pred_R, pred_t, gt_R, gt_t, K, obj_model['pts'], bop_sym)

                # VSD
                # print(depth.shape)
                #
                # depth_proc = depth.copy()
                # depth = depth.astype(np.int32)
                # depth_proc[depth_proc == 0] = depth_proc.max()
                pred_R, pred_t = self.normalize_pose(pred_R, pred_t)
                gt_R, gt_t = self.normalize_pose(gt_R, gt_t)
                vsd_err = vsd(pred_R, pred_t, gt_R, gt_t, depth, K,
                              self.vsd_delta, self.vsd_taus, True, obj_diam,
                              self.vsd_renderer, obj_id)
                # print("gt_t", gt_t)
                # print("gt_R", gt_R)
                # print("pred_t",pred_t)
                # print("pred_R", pred_R)
                # print("depth", depth)
                # print("camera", K)
                # print("obj_diam:", obj_diam)
                # print("cls_id", obj_id)
                # print("VSD errors:", vsd_err)
                # test
                # print("obj_diam:",obj_diam)

                # print("K.shape",K.shape)
                # # 处理 depth  0
                # depth_proc = depth.copy()
                # depth_proc[depth_proc == 0] = depth_proc.max()
                # vsd_err = vsd(pred_R, pred_t, gt_R, gt_t, depth_proc, K,
                #               self.vsd_delta, self.vsd_taus, True, obj_diam,
                #               self.vsd_renderer, obj_id)

            # 计算 recall 分数
            mssd_rec = np.arange(0.05, 0.51, 0.05) * obj_diam
            mssd_scores.append((mssd_err < mssd_rec).mean())  # if 'mssd_err' in locals() else 0.0)

            mspd_rec = np.arange(5, 51, 5)
            mspd_scores.append((mspd_err < mspd_rec).mean())  # if 'mspd_err' in locals() else 0.0)

            # vsd_rec = np.arange(0.05, 0.51, 0.05)
            # vsd_scores.append((np.array(vsd_err) < vsd_rec).mean())#if 'vsd_err' in locals() else 0.0)

            #  VSD preprocess
            vsd_err = np.asarray(vsd_err)
            all_vsd_recs = np.stack([vsd_err < rec_i for rec_i in self.vsd_rec], axis=1)
            mean_vsd = all_vsd_recs.mean()
        logs = {
            'loss': total_loss.detach(),
            'pose_loss': pose_loss.detach(),
            'pose_rot_loss': pose_rot_loss.detach(),
            'pose_trans_loss': pose_trans_loss.detach(),
            'mask_loss': out['mask_loss'].detach(),
            'mask_iou': out['mask_iou'].detach(),
            'add01d_acc': add01d_acc,
            'mssd': torch.tensor(np.mean(mssd_scores) if mssd_scores else 0.0),
            'mspd': torch.tensor(np.mean(mspd_scores) if mspd_scores else 0.0),
            'vsd': torch.tensor(mean_vsd)  # torch.tensor(np.mean(vsd_scores) if vsd_scores else 0.0),
        }

        self.validation_step_outputs.append(logs)
        return logs

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # 聚合所有batch的指标
        metrics = {
            'val/loss': torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_loss': torch.stack([x['pose_loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_rot_loss': torch.stack([x['pose_rot_loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_trans_loss': torch.stack([x['pose_trans_loss'] for x in self.validation_step_outputs]).mean(),
            'val/mask_loss': torch.stack([x['mask_loss'] for x in self.validation_step_outputs]).mean(),
            'val/mask_iou': torch.stack([x['mask_iou'] for x in self.validation_step_outputs]).mean(),
            'val/add01d_acc': torch.stack([x['add01d_acc'] for x in self.validation_step_outputs]).mean(),
            'val/mssd': torch.stack([x['mssd'] for x in self.validation_step_outputs]).mean(),
            'val/mspd': torch.stack([x['mspd'] for x in self.validation_step_outputs]).mean(),
            'val/vsd': torch.stack([x['vsd'] for x in self.validation_step_outputs]).mean(),
        }

        # 记录所有指标
        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=('loss' in name or 'acc' in name))

        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # 聚合所有batch的指标
        metrics = {
            'val/loss': torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_loss': torch.stack([x['pose_loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_rot_loss': torch.stack([x['pose_rot_loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_trans_loss': torch.stack([x['pose_trans_loss'] for x in self.validation_step_outputs]).mean(),
            'val/mask_loss': torch.stack([x['mask_loss'] for x in self.validation_step_outputs]).mean(),
            'val/mask_iou': torch.stack([x['mask_iou'] for x in self.validation_step_outputs]).mean(),
            'val/add01d_acc': torch.stack([x['add01d_acc'] for x in self.validation_step_outputs]).mean(),
            'val/mssd': torch.stack([x['mssd'] for x in self.validation_step_outputs]).mean(),
            'val/mspd': torch.stack([x['mspd'] for x in self.validation_step_outputs]).mean(),
            'val/vsd': torch.stack([x['vsd'] for x in self.validation_step_outputs]).mean(),
        }

        # 记录所有指标
        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=('loss' in name or 'acc' in name))

        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # 聚合所有batch的指标
        metrics = {
            'val/loss': torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_loss': torch.stack([x['pose_loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_rot_loss': torch.stack([x['pose_rot_loss'] for x in self.validation_step_outputs]).mean(),
            'val/pose_trans_loss': torch.stack([x['pose_trans_loss'] for x in self.validation_step_outputs]).mean(),
            'val/mask_loss': torch.stack([x['mask_loss'] for x in self.validation_step_outputs]).mean(),
            'val/mask_iou': torch.stack([x['mask_iou'] for x in self.validation_step_outputs]).mean(),
            'val/add01d_acc': torch.stack([x['add01d_acc'] for x in self.validation_step_outputs]).mean(),
            'val/mssd': torch.stack([x['mssd'] for x in self.validation_step_outputs]).mean(),
            'val/mspd': torch.stack([x['mspd'] for x in self.validation_step_outputs]).mean(),
            'val/vsd': torch.stack([x['vsd'] for x in self.validation_step_outputs]).mean(),
        }

        # 记录所有指标
        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=('loss' in name or 'acc' in name))

        self.validation_step_outputs.clear()

    # -------------------------
    #   Optim / Scheduler
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAINING.LR, eps=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(self.cfg.TRAINING, 'LR_GAMMA', 0.5),
            patience=getattr(self.cfg.TRAINING, 'LR_PATIENCE', 5),
            threshold=1e-2,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',  # ✅ validation_step 或 epoch_end log  key 对应
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # -------------------------
    #   ADD(S)-0.1D 评估
    # -------------------------
    def compute_add01d(self, batch, R_pred, t_pred):
        """计算ADD(S)-0.1d指标"""
        if not hasattr(self, 'obj_models'):
            return torch.tensor(0.0)

        total, success = 0, 0
        for i in range(len(batch['obj_id'])):
            obj_id = batch['obj_id'][i]

            obj_model = self.obj_models[obj_id]
            obj_diam = self.obj_diams[obj_id]
            obj_sym = self.obj_symms.get(obj_id, None)

            # 构造预测和GT姿态
            pred_pose = torch.eye(4, device=self.device)
            pred_pose[:3, :3] = R_pred[i]
            pred_pose[:3, 3] = t_pred[i].squeeze()

            gt_pose = batch['item_q_pose'][i]

            # 计算ADD(S)
            if obj_sym is not None and len(obj_sym) > 0:
                add_metric = compute_adds(obj_model['pts'] / 1000.,
                                          pred_pose.cpu().numpy(),
                                          gt_pose.cpu().numpy())
            else:
                add_metric = compute_add(obj_model['pts'] / 1000.,
                                         pred_pose.cpu().numpy(),
                                         gt_pose.cpu().numpy())

            threshold_m = 0.1 * (obj_diam / 1000.0)
            if add_metric < threshold_m:
                success += 1
            total += 1

        return torch.tensor(success / total if total > 0 else 0.0)

    @torch.no_grad()
    def add_object_info(self, obj_models: dict, obj_diams: dict, obj_symms: dict):
        # these are supposed to be in mm!
        self.obj_models = obj_models
        self.obj_diams = obj_diams
        self.obj_symms = {k: format_sym_set(sym_set) for k, sym_set in obj_symms.items()}

        if self.compute_vsd:
            for obj_id, obj in self.obj_models.items():
                self.vsd_renderer.my_add_object(obj, obj_id)

    def run_add01d_eval(self):
        """
        - 需要 val_dataloader().dataset 提供 dataset.get_obj_info(obj_id)
        - 逐 batch 用 forward_once 获取 R_pred,t_pred（已是 Query 的绝对姿态）
        - 计算物体 ADD/ADD-S，并用 0.1D 判定成功
        返回：成功率（百分比），若失败返回 None
        """
        try:
            dm = self.trainer.datamodule
            vloader = dm.val_dataloader()
        except Exception as e:
            print(f"[WARN] 获取 val_dataloader 失败：{e}")
            return None

        # dataset 必须实现 get_obj_info(obj_id) -> (model, diameter, sym)
        dataset = getattr(vloader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'get_obj_info'):
            print("[WARN] val dataset 未提供 get_obj_info(obj_id)，跳过 ADD-0.1D 评估")
            return None

        total, success = 0, 0
        device = self.device

        for batch in vloader:
            # 移动到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 预测绝对姿态
            out = self.forward_once(batch)
            R_pred = out['R_pred']  # [B,3,3]
            t_pred = out['t_pred'].squeeze(1)  # [B,3] (m)

            # GT 绝对姿态
            T_q_gt = batch['item_q_pose']
            R_gt = T_q_gt[:, :3, :3]
            t_gt = T_q_gt[:, :3, 3]  # [B,3] (m)

            obj_ids = batch['obj_id']  # [B]
            B = R_pred.shape[0]
            for i in range(B):
                try:
                    obj_model, obj_diam_mm, obj_sym = dataset.get_obj_info(obj_ids[i])
                    pts3d_model = (obj_model['pts']).astype(np.float32) / 1000.0  # m
                except Exception as e:
                    print(f"[WARN] get_obj_info 失败：{e}")
                    continue

                # 构造 4x4 姿态
                T_pred = np.eye(4, dtype=np.float32)
                T_pred[:3, :3] = R_pred[i].detach().cpu().numpy()
                T_pred[:3, 3] = t_pred[i].detach().cpu().numpy()

                T_gt = np.eye(4, dtype=np.float32)
                T_gt[:3, :3] = R_gt[i].detach().cpu().numpy()
                T_gt[:3, 3] = t_gt[i].detach().cpu().numpy()

                if len(obj_sym) > 0:
                    add_metric = compute_adds(pts3d_model, T_pred, T_gt)
                else:
                    add_metric = compute_add(pts3d_model, T_pred, T_gt)

                threshold_m = 0.1 * (obj_diam_mm / 1000.0)
                ok = (add_metric < threshold_m)
                total += 1
                success += int(ok)

        if total == 0:
            print("[WARN] ADD(S)-0.1D 评估没有有效样本")
            return None

        acc = 100.0 * success / total
        return acc

    def normalize_pose(self, R, t):
        # R: 3x3 旋转矩阵
        # t: 3x1 平移向量
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:  # 处理反射
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho, t

    # def debug_loftr(self, batch):
    #     """
    #     Debug LoFTR + mask filtering + visualization + top-8 mconf filtering
    #     保存原图、GT mask过滤图、LoFTR匹配可视化图
    #     """
    #
    #     import os
    #     import torch
    #     import numpy as np
    #     from PIL import Image
    #     import torchvision.utils as vutils
    #     import matplotlib.pyplot as plt
    #     import cv2
    #
    #     save_dir = "debug_loftr"
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     B = batch['image0'].shape[0]
    #     num_save = min(3, B)
    #     device = batch['image0'].device
    #
    #     for i in range(num_save):
    #         img0 = batch['image0'][i:i + 1]  # [1,3,H,W]
    #         img1 = batch['image1'][i:i + 1]
    #         mask0 = batch['mask0_gt'][i]  # [H,W]
    #         mask1 = batch['mask1_gt'][i]
    #
    #         # -------------------------
    #         # 1) 保存原图
    #         # -------------------------
    #         vutils.save_image(img0, f"{save_dir}/image0_raw_{i}.png", normalize=True)
    #         vutils.save_image(img1, f"{save_dir}/image1_raw_{i}.png", normalize=True)
    #
    #         # -------------------------
    #         # 2) GT mask 过滤图
    #         # -------------------------
    #         mask0_t = mask0.unsqueeze(0).unsqueeze(0).float()
    #         mask1_t = mask1.unsqueeze(0).unsqueeze(0).float()
    #
    #         img0_filtered = img0 * mask0_t
    #         img1_filtered = img1 * mask1_t
    #
    #         vutils.save_image(img0_filtered, f"{save_dir}/image0_gtmask_{i}.png", normalize=True)
    #         vutils.save_image(img1_filtered, f"{save_dir}/image1_gtmask_{i}.png", normalize=True)
    #
    #         def rgb_to_gray(img):
    #             """
    #             img: Tensor [B, 3, H, W] float, 0~1
    #             return: [B, 1, H, W]
    #             """
    #             # NTSC标准权重，更贴近视觉特性
    #             r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    #             gray = 0.299 * r + 0.587 * g + 0.114 * b
    #             return gray.float()
    #
    #         img0_gray_filtered = rgb_to_gray(img0_filtered)
    #         img1_gray_filtered = rgb_to_gray(img1_filtered)
    #         # -------------------------
    #         # 3）送入 LoFTR
    #         # -------------------------
    #         match_batch = {'image0': img0_gray_filtered, 'image1':img1_gray_filtered}
    #         with torch.no_grad():
    #             self.matcher.eval()
    #             self.matcher(match_batch)
    #
    #         mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()  # [N, 2]
    #         mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()  # [N, 2]
    #         mconf = match_batch['mconf'].detach().cpu().numpy()  # [N]
    #
    #         # if (self.useGTmask):
    #         # if使用 GT 掩码
    #         m0 = batch['mask0_gt'][i].detach().cpu().numpy()  # ★
    #         m1 = batch['mask1_gt'][i].detach().cpu().numpy()  # ★
    #         # else:
    #         #     # if pred mask
    #         #     m0 = pred_mask0_bin[i].detach().cpu().numpy()
    #         #     m1 = pred_mask1_bin[i].detach().cpu().numpy()
    #
    #         if len(mkpts0) == 0:
    #             print("error,len(mkpts0)<0 after loftr")
    #             continue
    #
    #         # # 按掩码过滤关键点
    #         # in_mask = (m0[mkpts0[:, 1].round().astype(int),
    #         # mkpts0[:, 0].round().astype(int)] > 0) & \
    #         #           (m1[mkpts1[:, 1].round().astype(int),
    #         #           mkpts1[:, 0].round().astype(int)] > 0)
    #         # mkpts0 = mkpts0[in_mask]
    #         # mkpts1 = mkpts1[in_mask]
    #         #
    #         # if len(mkpts0) < 3:
    #         #     print("error,len(mkpts0)<3 after filtering")
    #         #     R_preds.append(torch.eye(3, device=device))
    #         #     t_preds.append(torch.zeros(1, 3, device=device))
    #         #     continue
    #
    #         # ✅ 按掩码过滤关键点
    #         in_mask = (m0[mkpts0[:, 1].round().astype(int),
    #         mkpts0[:, 0].round().astype(int)] > 0) & \
    #                   (m1[mkpts1[:, 1].round().astype(int),
    #                   mkpts1[:, 0].round().astype(int)] > 0)
    #
    #         mkpts0 = mkpts0[in_mask]
    #         mkpts1 = mkpts1[in_mask]
    #         mconf = mconf[in_mask]  # ★ 同步过滤置信度
    #
    #         # -------------------------
    #         # 4）按置信度排序，取 top-8
    #         # -------------------------
    #         idx = np.argsort(-mconf)[:8]
    #         mkpts0 = mkpts0[idx]
    #         mkpts1 = mkpts1[idx]
    #         mconf = mconf[idx]
    #
    #         # -------------------------
    #         # 5）画匹配可视化图
    #         # -------------------------
    #         img0_vis = (img0.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #         img1_vis = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #
    #         # ✅ 拼接并确保 OpenCV-friendly 格式
    #         concat = np.concatenate([img0_vis, img1_vis], axis=1).astype(np.uint8).copy()
    #
    #         for (p0, p1) in zip(mkpts0, mkpts1):
    #             p1_shift = p1.copy()
    #             p1_shift[0] += img0_vis.shape[1]  # shift x of image1
    #
    #             p0 = p0.astype(int)
    #             p1_shift = p1_shift.astype(int)
    #
    #             cv2.line(concat, tuple(p0), tuple(p1_shift), (255, 0, 0), 1)
    #
    #         Image.fromarray(concat).save(f"{save_dir}/match_vis_{i}.png")
    def debug_loftr(self, batch):
        """
        Debug LoFTR + mask filtering + visualization + top-8 mconf filtering
        保存原图、GT mask过滤图、LoFTR匹配可视化图 +
        保存GT pose与Pred pose + mask点云 + 3D关键点可视化
        """
        import os, torch, numpy as np, cv2
        import torchvision.utils as vutils
        from PIL import Image
        try:
            import open3d as o3d
        except Exception:
            o3d = None

        save_dir = "debug_loftr"
        os.makedirs(save_dir, exist_ok=True)

        # pose log
        pose_log = os.path.join(save_dir, "pose_log.txt")

        # ✅ 清空 pose_log 文件（覆盖模式）
        with open(pose_log, "w") as f:
            f.write("=== Pose Debug Log ===\n")

        B = batch['image0'].shape[0]
        num_save = min(3, B)
        device = batch['image0'].device

        for i in range(num_save):

            # ✅ sample_i 子文件夹
            sample_dir = f"{save_dir}/sample_{i}"
            os.makedirs(sample_dir, exist_ok=True)

            img0 = batch['image0'][i:i + 1]  # [1,3,H,W]
            img1 = batch['image1'][i:i + 1]
            mask0 = batch['mask0_gt'][i]
            mask1 = batch['mask1_gt'][i]
            depth0 = batch['depth0'][i].detach().cpu().numpy() / 10.
            depth1 = batch['depth1'][i].detach().cpu().numpy() / 10.
            K0 = batch['K_color0'][i].detach().cpu().numpy()
            K1 = batch['K_color1'][i].detach().cpu().numpy()

            # 1) 保存原图
            vutils.save_image(img0, f"{sample_dir}/image0_raw.png", normalize=True)
            vutils.save_image(img1, f"{sample_dir}/image1_raw.png", normalize=True)

            # 2) 保存 mask 过滤图
            mask0_t = mask0.unsqueeze(0).unsqueeze(0).float()
            mask1_t = mask1.unsqueeze(0).unsqueeze(0).float()
            img0_filtered = img0 * mask0_t
            img1_filtered = img1 * mask1_t
            vutils.save_image(img0_filtered, f"{sample_dir}/image0_gtmask.png", normalize=True)
            vutils.save_image(img1_filtered, f"{sample_dir}/image1_gtmask.png", normalize=True)

            # rgb to gray
            r, g, b = img0_filtered[:, 0:1], img0_filtered[:, 1:2], img0_filtered[:, 2:3]
            img0_gray_filtered = (0.299 * r + 0.587 * g + 0.114 * b).float()
            r, g, b = img1_filtered[:, 0:1], img1_filtered[:, 1:2], img1_filtered[:, 2:3]
            img1_gray_filtered = (0.299 * r + 0.587 * g + 0.114 * b).float()

            # 3) LoFTR
            match_batch = {'image0': img0_gray_filtered, 'image1': img1_gray_filtered}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].cpu().numpy()
            mkpts1 = match_batch['mkpts1_f'].cpu().numpy()
            mconf = match_batch['mconf'].cpu().numpy()

            # mask filter (用GT mask)
            m0 = mask0.detach().cpu().numpy()
            m1 = mask1.detach().cpu().numpy()
            in_mask = (m0[mkpts0[:, 1].round().astype(int), mkpts0[:, 0].round().astype(int)] > 0) & \
                      (m1[mkpts1[:, 1].round().astype(int), mkpts1[:, 0].round().astype(int)] > 0)

            mkpts0 = mkpts0[in_mask]
            mkpts1 = mkpts1[in_mask]
            mconf = mconf[in_mask]

            if len(mkpts0) < 3:
                print("❌ less than 3 correspondences after mask filtering")
                continue

            # 4) Top-K filter (和 forward_once 保持一致)
            top_k = getattr(self, "top_k_matches", None)
            if top_k is None:
                top_k = getattr(self, "topk_filter", 8)
            if top_k is not None and top_k > 0:
                if len(mkpts0) >= top_k:
                    idx = np.argsort(-mconf)[:top_k]
                    mkpts0 = mkpts0[idx]
                    mkpts1 = mkpts1[idx]
                    mconf = mconf[idx]
                elif len(mkpts0) < 3:
                    print(f"❌ less than 3 correspondences after top-k (got {len(mkpts0)})")
                    continue

            # 5) 调整内参（如果需要）与回投影
            current_h = img0.shape[2]
            current_w = img0.shape[3]
            if hasattr(self, "adjust_intrinsics_for_resize"):
                K0_adj = self.adjust_intrinsics_for_resize(K0, original_size=(640, 480),
                                                           current_size=(current_w, current_h))
                K1_adj = self.adjust_intrinsics_for_resize(K1, original_size=(640, 480),
                                                           current_size=(current_w, current_h))
            else:
                K0_adj = K0.copy()
                K1_adj = K1.copy()

            if hasattr(self, "backproject"):
                pts3d_0, valid0 = self.backproject(mkpts0, depth0, K0_adj)
                pts3d_1, valid1 = self.backproject(mkpts1, depth1, K1_adj)
            else:
                def _local_backproject(pts2d, depth, K):
                    pts_3d = []
                    valid = []
                    fx = K[0, 0];
                    fy = K[1, 1];
                    cx = K[0, 2];
                    cy = K[1, 2]
                    for (u_f, v_f) in pts2d:
                        u = int(round(u_f));
                        v = int(round(v_f))
                        if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
                            valid.append(False);
                            pts_3d.append([0, 0, 0]);
                            continue
                        z = depth[v, u]
                        if z <= 0:
                            valid.append(False);
                            pts_3d.append([0, 0, 0])
                        else:
                            x = (u - cx) * z / fx
                            y = (v - cy) * z / fy
                            pts_3d.append([x, y, z]);
                            valid.append(True)
                    return np.array(pts_3d), np.array(valid, dtype=bool)

                pts3d_0, valid0 = _local_backproject(mkpts0, depth0, K0_adj)
                pts3d_1, valid1 = _local_backproject(mkpts1, depth1, K1_adj)

            valid = valid0 & valid1
            if np.count_nonzero(valid) < 3:
                print("❌ less than 3 valid 3D correspondences after backprojection")
                continue

            A = pts3d_0[valid]
            Bp = pts3d_1[valid]

            # Kabsch
            if hasattr(self, "kabsch_umeyama"):
                R_np, t_np = self.kabsch_umeyama(A, Bp)
            else:
                def _kabsch(A_pts, B_pts):
                    pA = A_pts.mean(axis=0)
                    pB = B_pts.mean(axis=0)
                    H = (A_pts - pA).T @ (B_pts - pB)
                    U, S, Vt = np.linalg.svd(H)
                    R = Vt.T @ U.T
                    if np.linalg.det(R) < 0:
                        Vt[2, :] *= -1;
                        R = Vt.T @ U.T
                    t = pB - R @ pA
                    return R, t

                R_np, t_np = _kabsch(A, Bp)

                # 10) 转到绝对位姿
            T_a = batch['item_a_pose'][i].detach().cpu().numpy()
            T_rel = np.eye(4, dtype=np.float32)
            T_rel[:3, :3] = R_np
            T_rel[:3, 3] = (t_np / 1000.0)
            T_q_pred = T_rel @ T_a
            R = torch.from_numpy(T_q_pred[:3, :3]).float().to(device)
            t = torch.from_numpy(T_q_pred[:3, 3]).float().to(device).unsqueeze(0)
            # R = R_np
            # t = t_np / 1000.

            # ---- ✅ 保存GT & Pred pose ----
            R_gt = batch['item_q_pose'][i, :3, :3].cpu().numpy()
            t_gt = batch['item_q_pose'][i, :3, 3].cpu().numpy()

            # ---- 🔍 调试打印 GT 和 Pred 姿态 ----
            print(f"\n{'=' * 60}")
            print(f"[DEBUG] Sample {i} Pose Comparison")
            print(f"{'=' * 60}")
            print(f"GT Rotation (R_gt):\n{R_gt}")
            print(f"\nGT Translation (t_gt): {t_gt}")
            print(f"\nPred Rotation (R_pred):\n{R}")
            print(f"\nPred Translation (t_pred): {t}")
            print(f"{'=' * 60}\n")

            with open(pose_log, "a") as f:
                f.write(f"\n==== Sample {i} ====\n")
                f.write("GT_R:\n" + str(R_gt) + "\nGT_t:" + str(t_gt) + "\n")
                f.write("Pred_R:\n" + str(R) + "\nPred_t:" + str(t) + "\n")

            # ==== ✅ image0 mask → 3D 点云 ====
            mask_pts0 = np.argwhere(m0 > 0)
            if mask_pts0.shape[0] > 0:
                mask_uv0 = mask_pts0[:, [1, 0]]
                if hasattr(self, "backproject"):
                    pc_mask0, _ = self.backproject(mask_uv0, depth0, K0_adj)
                else:
                    pc_mask0, _ = _local_backproject(mask_uv0, depth0, K0_adj)
            else:
                pc_mask0 = np.zeros((0, 3))

            # ==== ✅ image1 mask → 3D 点云（新加） ====
            mask_pts1 = np.argwhere(m1 > 0)
            if mask_pts1.shape[0] > 0:
                mask_uv1 = mask_pts1[:, [1, 0]]
                if hasattr(self, "backproject"):
                    pc_mask1, _ = self.backproject(mask_uv1, depth1, K1_adj)
                else:
                    pc_mask1, _ = _local_backproject(mask_uv1, depth1, K1_adj)
            else:
                pc_mask1 = np.zeros((0, 3))

            # final LoFTR 3D keypoints (image0 domain)
            pc_key0 = A.copy()
            pc_key1 = Bp.copy()
            # ==== ✅ 保存Ply ====
            if o3d is not None:
                # image0 mask cloud
                if pc_mask0.shape[0] > 0:
                    pcd0 = o3d.geometry.PointCloud()
                    pcd0.points = o3d.utility.Vector3dVector(pc_mask0)
                    pcd0.paint_uniform_color([0.5, 0.5, 0.5])
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_mask_img0.ply", pcd0)

                # image1 mask cloud（新加）
                if pc_mask1.shape[0] > 0:
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(pc_mask1)
                    pcd1.paint_uniform_color([0.0, 0.5, 1.0])  # 蓝色
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_mask_img1.ply", pcd1)

                # keypoints0
                if pc_key0.shape[0] > 0:
                    pcdk = o3d.geometry.PointCloud()
                    pcdk.points = o3d.utility.Vector3dVector(pc_key0)
                    pcdk.paint_uniform_color([1, 0, 0])
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_keypoints_0.ply", pcdk)
                if pc_key1.shape[0] > 0:
                    pcdk = o3d.geometry.PointCloud()
                    pcdk.points = o3d.utility.Vector3dVector(pc_key1)
                    pcdk.paint_uniform_color([1, 0, 0])
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_keypoints_1.ply", pcdk)

                ##merged
                if pc_mask0.shape[0] > 0:
                    pcd_mix0 = o3d.geometry.PointCloud()
                    #  mask points
                    pcd_mix0.points = o3d.utility.Vector3dVector(pc_mask0)
                    colors0 = np.tile(np.array([[0.5, 0.5, 0.5]]), (pc_mask0.shape[0], 1))  # gray

                    # keypoints0
                    if pc_key0.shape[0] > 0:
                        pts_all0 = np.vstack([pc_mask0, pc_key0])
                        pcd_mix0.points = o3d.utility.Vector3dVector(pts_all0)
                        colors0 = np.vstack([
                            colors0,
                            np.tile(np.array([[1.0, 0.0, 0.0]]), (pc_key0.shape[0], 1))  # red
                        ])

                    pcd_mix0.colors = o3d.utility.Vector3dVector(colors0)
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_mix_img0_mask_keypoints.ply", pcd_mix0)

                    # ==== ✅ image1 混合点云 ====
                if pc_mask1.shape[0] > 0:
                    pcd_mix1 = o3d.geometry.PointCloud()
                    pcd_mix1.points = o3d.utility.Vector3dVector(pc_mask1)
                    colors1 = np.tile(np.array([[0.5, 0.5, 0.5]]), (pc_mask1.shape[0], 1))  # gray

                    if pc_key1.shape[0] > 0:
                        pts_all1 = np.vstack([pc_mask1, pc_key1])
                        pcd_mix1.points = o3d.utility.Vector3dVector(pts_all1)
                        colors1 = np.vstack([
                            colors1,
                            np.tile(np.array([[1.0, 0.0, 0.0]]), (pc_key1.shape[0], 1))  # red
                        ])

                    pcd_mix1.colors = o3d.utility.Vector3dVector(colors1)
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_mix_img1_mask_keypoints.ply", pcd_mix1)

            # ==== ✅ 3D matplotlib 保存 ====

            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # image0 mask = gray
                if pc_mask0.shape[0] > 0:
                    ax.scatter(pc_mask0[:, 0], pc_mask0[:, 1], pc_mask0[:, 2], s=1, c='gray', label="mask0 (img0)")

                # image1 mask = blue
                if pc_mask1.shape[0] > 0:
                    ax.scatter(pc_mask1[:, 0], pc_mask1[:, 1], pc_mask1[:, 2], s=1, c='blue', label="mask1 (img1)")

                # image0 keypoints = red
                if pc_key0.shape[0] > 0:
                    ax.scatter(pc_key0[:, 0], pc_key0[:, 1], pc_key0[:, 2], s=30, c='red', label="keypoints0 (img0)")

                # image1 keypoints = green
                if pc_key1.shape[0] > 0:
                    ax.scatter(pc_key1[:, 0], pc_key1[:, 1], pc_key1[:, 2], s=30, c='lime', label="keypoints1 (img1)")

                # ---- ✅ draw match lines between keypoints in 3D ----
                if pc_key0.shape[0] > 0 and pc_key1.shape[0] > 0:
                    for p0, p1 in zip(pc_key0, pc_key1):
                        xs = [p0[0], p1[0]]
                        ys = [p0[1], p1[1]]
                        zs = [p0[2], p1[2]]
                        ax.plot(xs, ys, zs, linewidth=0.8, color='black')  # thin black line

                plt.title("3D Visualization: Mask Point Cloud + Keypoints")
                ax.legend(loc='upper right')

                plt.savefig(f"{sample_dir}/3d_vis.png", dpi=300)
                plt.close()

            except Exception as e:
                print("3D vis error:", e)
                pass

            # #3d merged two
            # 3d merged two -> Plotly HTML (headless-friendly)
            try:
                import numpy as np
                try:
                    import plotly.graph_objects as go
                except Exception as e_plot:
                    # plotly not installed: fallback to saving PLY only
                    print(
                        "plotly not available, fallback to PLY. Install plotly (`pip install plotly`) for interactive html.")
                    all_pts = []
                    for arr in [pc_mask0, pc_mask1, pc_key0, pc_key1]:
                        if arr is not None and arr.shape[0] > 0:
                            all_pts.append(arr)
                    if len(all_pts) > 0:
                        import open3d as o3d
                        pcd_all = o3d.geometry.PointCloud()
                        pcd_all.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
                        o3d.io.write_point_cloud(f"{sample_dir}/vis_all_points.ply", pcd_all)
                    raise

                # helper to create scatter trace for a point cloud
                def make_scatter3d(points, name, color, size=1, mode='markers'):
                    if points is None or points.shape[0] == 0:
                        return None
                    x = points[:, 0].tolist()
                    y = points[:, 1].tolist()
                    z = points[:, 2].tolist()
                    return go.Scatter3d(x=x, y=y, z=z,
                                        mode=mode,
                                        name=name,
                                        marker=dict(size=size, color=color, opacity=0.8),
                                        hoverinfo='name')

                traces = []
                # mask0 (gray small points)
                t = make_scatter3d(pc_mask0, "mask0 (img0)", 'rgb(160,160,160)', size=1)
                if t is not None: traces.append(t)
                # mask1 (blue small points)
                t = make_scatter3d(pc_mask1, "mask1 (img1)", 'rgb(50,150,255)', size=1)
                if t is not None: traces.append(t)
                # key0 (red bigger)
                t = make_scatter3d(pc_key0, "keypoints0 (img0)", 'rgb(255,30,30)', size=4)
                if t is not None: traces.append(t)
                # key1 (green bigger)
                t = make_scatter3d(pc_key1, "keypoints1 (img1)", 'rgb(30,200,30)', size=4)
                if t is not None: traces.append(t)

                # build one lines trace that contains all segments (use None separators)
                if pc_key0 is not None and pc_key1 is not None and pc_key0.shape[0] > 0 and pc_key1.shape[0] > 0:
                    n_lines = min(pc_key0.shape[0], pc_key1.shape[0])
                    xs, ys, zs = [], [], []
                    for idx in range(n_lines):
                        p0 = pc_key0[idx]
                        p1 = pc_key1[idx]
                        xs.extend([float(p0[0]), float(p1[0]), None])
                        ys.extend([float(p0[1]), float(p1[1]), None])
                        zs.extend([float(p0[2]), float(p1[2]), None])
                    line_trace = go.Scatter3d(x=xs, y=ys, z=zs,
                                              mode='lines',
                                              name='matches',
                                              line=dict(color='rgb(0,0,0)', width=2),
                                              hoverinfo='none')
                    traces.append(line_trace)

                # layout: set aspectmode to 'data' so axes are equal
                layout = go.Layout(
                    title="3D Visualization: Mask Point Cloud + Keypoints + Matches",
                    scene=dict(
                        xaxis=dict(title='X', visible=False),
                        yaxis=dict(title='Y', visible=False),
                        zaxis=dict(title='Z', visible=False),
                        aspectmode='data'
                    ),
                    legend=dict(itemsizing='constant'),
                    margin=dict(l=0, r=0, b=0, t=30)
                )

                fig = go.Figure(data=traces, layout=layout)
                html_path = os.path.join(sample_dir, "vis_3d.html")
                fig.write_html(html_path, include_plotlyjs='cdn')  # single-file html (uses CDN for plotly js)
                print(f"Saved interactive HTML viewer to: {html_path}")

                # also keep a PLY snapshot of points (no lines) for external tools
                try:
                    import open3d as o3d
                    all_pts = []
                    all_colors = []
                    if pc_mask0 is not None and pc_mask0.shape[0] > 0:
                        all_pts.append(pc_mask0);
                        all_colors.append(np.tile([0.6, 0.6, 0.6], (pc_mask0.shape[0], 1)))
                    if pc_mask1 is not None and pc_mask1.shape[0] > 0:
                        all_pts.append(pc_mask1);
                        all_colors.append(np.tile([0.2, 0.6, 1.0], (pc_mask1.shape[0], 1)))
                    if pc_key0 is not None and pc_key0.shape[0] > 0:
                        all_pts.append(pc_key0);
                        all_colors.append(np.tile([1.0, 0.2, 0.2], (pc_key0.shape[0], 1)))
                    if pc_key1 is not None and pc_key1.shape[0] > 0:
                        all_pts.append(pc_key1);
                        all_colors.append(np.tile([0.2, 1.0, 0.2], (pc_key1.shape[0], 1)))
                    if len(all_pts) > 0:
                        pts_all = np.vstack(all_pts)
                        cols_all = np.vstack(all_colors)
                        pcd_out = o3d.geometry.PointCloud()
                        pcd_out.points = o3d.utility.Vector3dVector(pts_all)
                        pcd_out.colors = o3d.utility.Vector3dVector(cols_all)
                        o3d.io.write_point_cloud(f"{sample_dir}/vis_all_points.ply", pcd_out)
                except Exception:
                    pass

            except Exception as e:
                print("3D web export failed:", e)
                pass

            # try:
            #     import open3d as o3d
            #     import numpy as np
            #
            #     vis_pcd = []
            #
            #     def make_pcd(points, color):
            #         if points.shape[0] == 0:
            #             return None
            #         p = o3d.geometry.PointCloud()
            #         p.points = o3d.utility.Vector3dVector(points)
            #         p.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
            #         return p
            #
            #     # === 点云 ===
            #     pcd_mask0 = make_pcd(pc_mask0, np.array([0.6, 0.6, 0.6]))  # gray
            #     pcd_mask1 = make_pcd(pc_mask1, np.array([0.2, 0.6, 1.0]))  # blue
            #     pcd_key0 = make_pcd(pc_key0, np.array([1.0, 0.2, 0.2]))  # red
            #     pcd_key1 = make_pcd(pc_key1, np.array([0.2, 1.0, 0.2]))  # green
            #
            #     for p in [pcd_mask0, pcd_mask1, pcd_key0, pcd_key1]:
            #         if p is not None:
            #             vis_pcd.append(p)
            #
            #     # === 连线（关键点匹配）===
            #     if pc_key0.shape[0] > 0 and pc_key1.shape[0] > 0:
            #         lines = []
            #         for i in range(min(pc_key0.shape[0], pc_key1.shape[0])):
            #             lines.append([i, i + pc_key0.shape[0]])
            #
            #         # 合并关键点
            #         all_keys = np.vstack([pc_key0, pc_key1])
            #         line_set = o3d.geometry.LineSet()
            #         line_set.points = o3d.utility.Vector3dVector(all_keys)
            #         line_set.lines = o3d.utility.Vector2iVector(lines)
            #         line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in lines])  # black
            #
            #         vis_pcd.append(line_set)
            #
            #     # === 可交互查看 ===
            #     #o3d.visualization.draw_geometries(vis_pcd, window_name="3D Keypoint Match Viewer")
            #     o3d.io.write_point_cloud(f"{sample_dir}/vis_all_points.ply", pcd_mask0 + pcd_mask1 + pcd_key0 + pcd_key1)
            #
            # except Exception as e:
            #     print("Open3D visualization failed:", e)
            #     pass

            # ✅ LoFTR match vis
            img0_vis = (img0.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img1_vis = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            concat = np.concatenate([img0_vis, img1_vis], axis=1).copy()
            W = img0_vis.shape[1]
            mk0_vis = mkpts0[valid]
            mk1_vis = mkpts1[valid]
            for p0, p1 in zip(mk0_vis, mk1_vis):
                p1s = p1.copy();
                p1s[0] += W
                cv2.line(concat, tuple(p0.astype(int)), tuple(p1s.astype(int)), (0, 255, 0), 1)
            Image.fromarray(concat).save(f"{sample_dir}/match_vis.png")

    # =========================================================================
    #   掩膜引导匹配对比图 (Mask-Guided Effect Comparison)
    #   对应章节: 2.3.1 Foreground-Focused Feature Extraction
    # =========================================================================
    def debug_mask_guided_comparison(self, batch, num_samples=3):
        """
        生成论文图: 掩膜引导匹配对比图

        左图 (Baseline/w.o. Mask): 原图直接匹配，红色线条表示背景噪声匹配
        右图 (Ours/Mask-Guided): Mask引导匹配，绿色线条表示高质量前景匹配

        新增: 使用预测掩码 (Pred Mask) 过滤的结果对比

        参考风格: LoFTR (CVPR 2021), SuperGlue (CVPR 2020)
        """
        import os
        import cv2
        import torch
        import numpy as np
        from PIL import Image
        import torchvision.utils as vutils
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 导入可视化函数
        from lib.models.MicKey.visualize_mask_guided_comparison import (
            visualize_mask_guided_comparison,
            visualize_mask_guided_comparison_v2,
            draw_mask_contour
        )

        save_dir = "debug_mask_guided_comparison"
        os.makedirs(save_dir, exist_ok=True)

        B = batch['image0'].shape[0]
        num_save = min(num_samples, B)
        device = batch['image0'].device
        _, _, H, W = batch['image0'].shape

        def rgb_to_gray(img):
            """img: [B, 3, H, W] -> [B, 1, H, W]"""
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            return (0.299 * r + 0.587 * g + 0.114 * b).float()

        # ========== 获取预测掩码 (Pred Mask) ==========
        with torch.no_grad():
            oryon_out = self.oryon_model.forward(batch)
            pred_mask0_logits = oryon_out['mask_a']  # [B,1,Hm,Wm]
            pred_mask1_logits = oryon_out['mask_q']  # [B,1,Hm,Wm]

            # logits -> 概率 -> 二值化
            pred_mask0_prob = torch.sigmoid(pred_mask0_logits).squeeze(1)  # [B,Hm,Wm]
            pred_mask1_prob = torch.sigmoid(pred_mask1_logits).squeeze(1)

            # resize 到输入图像大小
            pred_mask0_prob = F.interpolate(pred_mask0_prob.unsqueeze(1),
                                            size=(H, W), mode='bilinear',
                                            align_corners=False).squeeze(1)
            pred_mask1_prob = F.interpolate(pred_mask1_prob.unsqueeze(1),
                                            size=(H, W), mode='bilinear',
                                            align_corners=False).squeeze(1)

            # 二值化
            pred_mask0_bin = (pred_mask0_prob > 0.5).float()
            pred_mask1_bin = (pred_mask1_prob > 0.5).float()

        for i in range(num_save):
            sample_dir = f"{save_dir}/sample_{i}"
            os.makedirs(sample_dir, exist_ok=True)

            img0 = batch['image0'][i:i + 1]  # [1, 3, H, W]
            img1 = batch['image1'][i:i + 1]
            mask0_gt = batch['mask0_gt'][i]  # [H, W] GT mask
            mask1_gt = batch['mask1_gt'][i]
            mask0_pred = pred_mask0_bin[i]  # [H, W] Pred mask
            mask1_pred = pred_mask1_bin[i]

            H_img, W_img = img0.shape[2], img0.shape[3]

            # 转换为 numpy 格式
            img0_np = (img0.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img1_np = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            mask0_gt_np = mask0_gt.detach().cpu().numpy()
            mask1_gt_np = mask1_gt.detach().cpu().numpy()
            mask0_pred_np = mask0_pred.detach().cpu().numpy()
            mask1_pred_np = mask1_pred.detach().cpu().numpy()

            # ========== 1) 无 Mask 匹配 (Baseline) ==========
            img0_gray_raw = rgb_to_gray(img0)
            img1_gray_raw = rgb_to_gray(img1)

            match_batch_raw = {'image0': img0_gray_raw, 'image1': img1_gray_raw}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch_raw)

            mkpts0_raw = match_batch_raw['mkpts0_f'].cpu().numpy()
            mkpts1_raw = match_batch_raw['mkpts1_f'].cpu().numpy()
            mconf_raw = match_batch_raw['mconf'].cpu().numpy()

            print(f"[Sample {i}] Raw matches (w/o mask): {len(mkpts0_raw)}")

            # ========== 2) GT Mask 匹配 ==========
            mask0_gt_t = mask0_gt.unsqueeze(0).unsqueeze(0).float()
            mask1_gt_t = mask1_gt.unsqueeze(0).unsqueeze(0).float()

            img0_filtered_gt = img0 * mask0_gt_t
            img1_filtered_gt = img1 * mask1_gt_t

            img0_gray_gt_masked = rgb_to_gray(img0_filtered_gt)
            img1_gray_gt_masked = rgb_to_gray(img1_filtered_gt)

            match_batch_gt_masked = {'image0': img0_gray_gt_masked, 'image1': img1_gray_gt_masked}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch_gt_masked)

            mkpts0_gt_masked_all = match_batch_gt_masked['mkpts0_f'].cpu().numpy()
            mkpts1_gt_masked_all = match_batch_gt_masked['mkpts1_f'].cpu().numpy()
            mconf_gt_masked_all = match_batch_gt_masked['mconf'].cpu().numpy()

            # 进一步用 GT mask 过滤
            if len(mkpts0_gt_masked_all) > 0:
                in_mask_gt = (mask0_gt_np[mkpts0_gt_masked_all[:, 1].round().astype(int),
                mkpts0_gt_masked_all[:, 0].round().astype(int)] > 0) & \
                             (mask1_gt_np[mkpts1_gt_masked_all[:, 1].round().astype(int),
                             mkpts1_gt_masked_all[:, 0].round().astype(int)] > 0)
                mkpts0_gt_masked = mkpts0_gt_masked_all[in_mask_gt]
                mkpts1_gt_masked = mkpts1_gt_masked_all[in_mask_gt]
                mconf_gt_masked = mconf_gt_masked_all[in_mask_gt]
            else:
                mkpts0_gt_masked = mkpts0_gt_masked_all
                mkpts1_gt_masked = mkpts1_gt_masked_all
                mconf_gt_masked = mconf_gt_masked_all

            print(f"[Sample {i}] GT Masked matches: {len(mkpts0_gt_masked)}")

            # ========== 3) Pred Mask 匹配 ==========
            mask0_pred_t = mask0_pred.unsqueeze(0).unsqueeze(0).float()
            mask1_pred_t = mask1_pred.unsqueeze(0).unsqueeze(0).float()

            img0_filtered_pred = img0 * mask0_pred_t
            img1_filtered_pred = img1 * mask1_pred_t

            img0_gray_pred_masked = rgb_to_gray(img0_filtered_pred)
            img1_gray_pred_masked = rgb_to_gray(img1_filtered_pred)

            match_batch_pred_masked = {'image0': img0_gray_pred_masked, 'image1': img1_gray_pred_masked}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch_pred_masked)

            mkpts0_pred_masked_all = match_batch_pred_masked['mkpts0_f'].cpu().numpy()
            mkpts1_pred_masked_all = match_batch_pred_masked['mkpts1_f'].cpu().numpy()
            mconf_pred_masked_all = match_batch_pred_masked['mconf'].cpu().numpy()

            # 进一步用 Pred mask 过滤
            if len(mkpts0_pred_masked_all) > 0:
                in_mask_pred = (mask0_pred_np[mkpts0_pred_masked_all[:, 1].round().astype(int),
                mkpts0_pred_masked_all[:, 0].round().astype(int)] > 0) & \
                               (mask1_pred_np[mkpts1_pred_masked_all[:, 1].round().astype(int),
                               mkpts1_pred_masked_all[:, 0].round().astype(int)] > 0)
                mkpts0_pred_masked = mkpts0_pred_masked_all[in_mask_pred]
                mkpts1_pred_masked = mkpts1_pred_masked_all[in_mask_pred]
                mconf_pred_masked = mconf_pred_masked_all[in_mask_pred]
            else:
                mkpts0_pred_masked = mkpts0_pred_masked_all
                mkpts1_pred_masked = mkpts1_pred_masked_all
                mconf_pred_masked = mconf_pred_masked_all

            print(f"[Sample {i}] Pred Masked matches: {len(mkpts0_pred_masked)}")

            # ========== 4) 生成对比图 (GT Mask 版本) ==========
            visualize_mask_guided_comparison(
                img0=img0_np,
                img1=img1_np,
                mask0=mask0_gt_np,
                mask1=mask1_gt_np,
                mkpts0_raw=mkpts0_raw,
                mkpts1_raw=mkpts1_raw,
                mconf_raw=mconf_raw,
                mkpts0_masked=mkpts0_gt_masked,
                mkpts1_masked=mkpts1_gt_masked,
                mconf_masked=mconf_gt_masked,
                save_path=f"{sample_dir}/mask_guided_comparison_GT.png",
                title_right="GT Mask-Guided (Oracle)",
                max_lines=100,
                dpi=200,
            )

            # ========== 5) 生成对比图 (Pred Mask 版本) ==========
            visualize_mask_guided_comparison(
                img0=img0_np,
                img1=img1_np,
                mask0=mask0_pred_np,
                mask1=mask1_pred_np,
                mkpts0_raw=mkpts0_raw,
                mkpts1_raw=mkpts1_raw,
                mconf_raw=mconf_raw,
                mkpts0_masked=mkpts0_pred_masked,
                mkpts1_masked=mkpts1_pred_masked,
                mconf_masked=mconf_pred_masked,
                save_path=f"{sample_dir}/mask_guided_comparison_Pred.png",
                title_right="Pred Mask-Guided (Ours)",
                max_lines=100,
                dpi=200,
            )

            # ========== 6) 生成 2x2 布局版本 ==========
            visualize_mask_guided_comparison_v2(
                img0=img0_np,
                img1=img1_np,
                mask0=mask0_gt_np,
                mask1=mask1_gt_np,
                mkpts0_raw=mkpts0_raw,
                mkpts1_raw=mkpts1_raw,
                mconf_raw=mconf_raw,
                mkpts0_masked=mkpts0_gt_masked,
                mkpts1_masked=mkpts1_gt_masked,
                mconf_masked=mconf_gt_masked,
                save_path=f"{sample_dir}/mask_guided_comparison_v2_GT.png",
                max_lines=80,
                dpi=200,
            )

            visualize_mask_guided_comparison_v2(
                img0=img0_np,
                img1=img1_np,
                mask0=mask0_pred_np,
                mask1=mask1_pred_np,
                mkpts0_raw=mkpts0_raw,
                mkpts1_raw=mkpts1_raw,
                mconf_raw=mconf_raw,
                mkpts0_masked=mkpts0_pred_masked,
                mkpts1_masked=mkpts1_pred_masked,
                mconf_masked=mconf_pred_masked,
                save_path=f"{sample_dir}/mask_guided_comparison_v2_Pred.png",
                max_lines=80,
                dpi=200,
            )

            # ========== 7) 保存各阶段图像 ==========
            # 保存原图
            vutils.save_image(img0, f"{sample_dir}/image0_raw.png", normalize=True)
            vutils.save_image(img1, f"{sample_dir}/image1_raw.png", normalize=True)

            # 保存 GT mask 过滤后的图
            vutils.save_image(img0_filtered_gt, f"{sample_dir}/image0_masked_GT.png", normalize=True)
            vutils.save_image(img1_filtered_gt, f"{sample_dir}/image1_masked_GT.png", normalize=True)

            # 保存 Pred mask 过滤后的图
            vutils.save_image(img0_filtered_pred, f"{sample_dir}/image0_masked_Pred.png", normalize=True)
            vutils.save_image(img1_filtered_pred, f"{sample_dir}/image1_masked_Pred.png", normalize=True)

            # 保存 GT mask
            Image.fromarray((mask0_gt_np * 255).astype(np.uint8)).save(f"{sample_dir}/mask0_GT.png")
            Image.fromarray((mask1_gt_np * 255).astype(np.uint8)).save(f"{sample_dir}/mask1_GT.png")

            # 保存 Pred mask
            Image.fromarray((mask0_pred_np * 255).astype(np.uint8)).save(f"{sample_dir}/mask0_Pred.png")
            Image.fromarray((mask1_pred_np * 255).astype(np.uint8)).save(f"{sample_dir}/mask1_Pred.png")

            # ========== 8) 生成高质量论文图 (SuperGlue 风格) ==========
            # GT Mask 版本
            self._draw_superglue_style_comparison(
                img0_np, img1_np, mask0_gt_np, mask1_gt_np,
                mkpts0_raw, mkpts1_raw, mconf_raw,
                mkpts0_gt_masked, mkpts1_gt_masked, mconf_gt_masked,
                save_path=f"{sample_dir}/superglue_style_comparison_GT.png",
                title_ours="(b) GT Mask-Guided Matching (Oracle)"
            )

            # Pred Mask 版本
            self._draw_superglue_style_comparison(
                img0_np, img1_np, mask0_pred_np, mask1_pred_np,
                mkpts0_raw, mkpts1_raw, mconf_raw,
                mkpts0_pred_masked, mkpts1_pred_masked, mconf_pred_masked,
                save_path=f"{sample_dir}/superglue_style_comparison_Pred.png",
                title_ours="(b) Pred Mask-Guided Matching (Ours)"
            )

            # ========== 9) 三方对比图 (Baseline vs GT vs Pred) ==========
            self._draw_three_way_comparison(
                img0_np, img1_np,
                mask0_gt_np, mask1_gt_np,
                mask0_pred_np, mask1_pred_np,
                mkpts0_raw, mkpts1_raw, mconf_raw,
                mkpts0_gt_masked, mkpts1_gt_masked, mconf_gt_masked,
                mkpts0_pred_masked, mkpts1_pred_masked, mconf_pred_masked,
                save_path=f"{sample_dir}/three_way_comparison.png"
            )

            print(f"✅ Sample {i} saved to {sample_dir}/")

        print(f"\n🎉 All {num_save} samples saved to {save_dir}/")

    # =========================================================================
    #   [NEW] 2.4 Point Clouds Registration & Alignment Visualization
    #   展示从 2D 反投影形成视锥，到 SVD 刚体变换对齐的过程，含误差热力图
    # =========================================================================
    def debug_registration_alignment(self, src_pts, tgt_pts, R_pred, t_pred, sample_idx=0, instance_id=None,
                                     image0=None, image1=None):
        """
        src_pts: [N, 3] source points in Camera 0 frame
        tgt_pts: [N, 3] target points in Camera 1 frame (matched)
        R_pred:  [3, 3] rotation matrix (src -> tgt)
        t_pred:  [3] translation vector (src -> tgt)
        sample_idx: sample index for saving
        instance_id: instance id from batch for folder naming
        image0: [3, H, W] tensor, original image 0
        image1: [3, H, W] tensor, original image 1
        """
        import matplotlib
        matplotlib.use('Agg')  # 确保在服务器无头模式下运行
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import torchvision.utils as vutils
        from PIL import Image

        # 创建主目录和样本子目录（使用 instance_id 命名）
        save_dir = "debug_registration_vis"
        # 如果 instance_id 是 tensor，转换为 python 类型
        if instance_id is not None:
            if hasattr(instance_id, 'item'):
                instance_id = instance_id.item()
            sample_dir = os.path.join(save_dir, f"sample_{instance_id}")
        else:
            sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # ========== 保存原图 image0 和 image1 ==========
        if image0 is not None:
            vutils.save_image(image0.unsqueeze(0), os.path.join(sample_dir, "image0.png"), normalize=True)
            print(f"[Visualizer] Saved image0 to: {sample_dir}/image0.png")

        if image1 is not None:
            vutils.save_image(image1.unsqueeze(0), os.path.join(sample_dir, "image1.png"), normalize=True)
            print(f"[Visualizer] Saved image1 to: {sample_dir}/image1.png")

        # ========== 保存拼接的原图对 ==========
        if image0 is not None and image1 is not None:
            try:
                # 转换 tensor 到 numpy
                img0_np = image0.detach().cpu().permute(1, 2, 0).numpy()
                img1_np = image1.detach().cpu().permute(1, 2, 0).numpy()

                # 归一化到 0-255
                img0_np = (img0_np - img0_np.min()) / (img0_np.max() - img0_np.min() + 1e-8) * 255
                img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min() + 1e-8) * 255
                img0_np = img0_np.astype(np.uint8)
                img1_np = img1_np.astype(np.uint8)

                # 水平拼接
                concat_img = np.concatenate([img0_np, img1_np], axis=1)
                Image.fromarray(concat_img).save(os.path.join(sample_dir, "image_pair.png"))
                print(f"[Visualizer] Image pair saved to: {sample_dir}/image_pair.png")
            except Exception as e:
                print(f"[Visualizer] Failed to save image pair: {e}")

        # 1. 数据降采样 (防止点太多导致绘图卡顿，论文图通常不需要太多点)
        if src_pts.shape[0] > 500:
            idx = np.random.choice(src_pts.shape[0], 500, replace=False)
            src_p = src_pts[idx]
            tgt_p = tgt_pts[idx]
        else:
            src_p = src_pts
            tgt_p = tgt_pts

        # 2. 计算变换后的 Source 点云 (Aligned)
        # Formula: P_aligned = R * P_src + t
        # 注意 numpy 形状: (N,3) @ (3,3).T + (3,)
        src_p_aligned = (src_p @ R_pred.T) + t_pred

        # 3. 计算误差 (Error Map)
        # Euclidean distance per point
        errors = np.linalg.norm(src_p_aligned - tgt_p, axis=1)
        # 归一化误差用于颜色映射 (例如 0mm 到 50mm)
        max_err_vis = 50.0  # mm，根据物体尺度调整，通常 2cm-5cm 误差就很大了
        if src_p.max() < 10.0:  # 如果单位是米
            max_err_vis = 0.05

            # 颜色映射: Blue (Low Error) -> Red (High Error)
        norm_err = np.clip(errors / max_err_vis, 0, 1)
        cmap = plt.get_cmap('jet')  # 或者 'coolwarm'
        colors_error = cmap(norm_err)  # [N, 4] RGBA

        # ================== 绘图 ==================
        fig = plt.figure(figsize=(18, 8))

        # --- 子图 1: 2D Back-projection & Frustum (未对齐状态) ---
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        # 绘制 Source (Red)
        ax1.scatter(src_p[:, 0], src_p[:, 1], src_p[:, 2], s=5, c='r', alpha=0.6, label='Source (Cam 0)')

        # 绘制 Target (Green) - 这里的 Target 是在它自己的坐标系下，
        # 为了展示“未对齐”，我们将它们画在同一个坐标系里，它们此时应该是分离的。
        ax1.scatter(tgt_p[:, 0], tgt_p[:, 1], tgt_p[:, 2], s=5, c='g', alpha=0.6, label='Target (Cam 1)')

        # [可视化的精髓] 绘制视锥 (Frustum Lines)
        # 从原点 (0,0,0) 射向点云边缘的线，模拟反投影光线
        # 选取 Source 点云的 4 个角落或随机点
        origin = np.array([0, 0, 0])
        num_rays = 8
        step = max(1, len(src_p) // num_rays)
        for p in src_p[::step]:
            # 画线: Origin -> Point
            ax1.plot([origin[0], p[0]], [origin[1], p[1]], [origin[2], p[2]],
                     color='r', linestyle='--', linewidth=0.3, alpha=0.3)

        ax1.set_title("(a) 2D Back-projection & Frustum Generation\n(Misaligned Input Clouds)", fontsize=12,
                      fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend(loc='upper right')

        # 调整视角让视锥更明显
        ax1.view_init(elev=20, azim=45)

        # --- 子图 2: Rigid Transformation (SVD 优化结果) ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # 绘制 Target (Green, 作为 Ground Truth 参考)
        # 使用空心圈或者浅绿色，突出 Source 的覆盖
        ax2.scatter(tgt_p[:, 0], tgt_p[:, 1], tgt_p[:, 2], s=15, c='g', marker='o', alpha=0.2, label='Target (GT)')

        # 绘制 Aligned Source (Color-coded by Error)
        # 这就是 PVN3D/PointDSC 风格的 Error Map
        sc = ax2.scatter(src_p_aligned[:, 0], src_p_aligned[:, 1], src_p_aligned[:, 2],
                         s=8, c=colors_error, alpha=0.9, label='Aligned Source (Ours)')

        # 添加 Colorbar
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=max_err_vis), cmap=cmap),
            ax=ax2, shrink=0.6)
        cbar.set_label(f'Alignment Error (0-{max_err_vis} m/mm)', fontsize=10)

        # 绘制 SVD 变换轨迹示意 (可选)
        # 画一条线连接 变换前的重心 -> 变换后的重心，表示位移 t
        centroid_before = np.mean(src_p, axis=0)
        centroid_after = np.mean(src_p_aligned, axis=0)
        ax2.plot([centroid_before[0], centroid_after[0]],
                 [centroid_before[1], centroid_after[1]],
                 [centroid_before[2], centroid_after[2]],
                 c='k', linewidth=2, linestyle='-', label='Rigid Transform $R,t$')

        # 在轨迹上加个箭头 (简单用文字代替)
        mid_point = (centroid_before + centroid_after) / 2
        ax2.text(mid_point[0], mid_point[1], mid_point[2], "SVD Opt.", color='black', fontsize=10, fontweight='bold')

        ax2.set_title("(b) Rigid Transformation & Error Map\n(Perfect Alignment via SVD)", fontsize=12,
                      fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend(loc='lower left')
        ax2.view_init(elev=20, azim=45)

        # 保存配准可视化图到样本子目录
        save_path = os.path.join(sample_dir, "registration_vis.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Visualizer] 2D-3D Registration plot saved to: {save_path}")

    # =========================================================================
    #   [NEW] 3D Bounding Box Visualization on Image Pair
    #   展示 anchor 图和 query 图，在 query 图上绘制 GT (绿色) 和 Pred (蓝色) 的 3D bounding box
    # =========================================================================
    # def debug_boundingbox_visualization(self, image0, image1, R_pred, t_pred, R_gt, t_gt,
    #                                     K, obj_id, sample_idx=0, instance_id=None):
    #     """
    #     可视化 3D bounding box 投影到 query 图像上
    #     绿色: GT pose 的 bounding box
    #     红色: Predicted pose 的 bounding box

    #     Args:
    #         image0: [3, H, W] tensor, anchor image
    #         image1: [3, H, W] tensor, query image
    #         R_pred: [3, 3] numpy array, predicted rotation (absolute pose)
    #         t_pred: [3] numpy array, predicted translation in meters (absolute pose)
    #         R_gt: [3, 3] numpy array, GT rotation (absolute pose)
    #         t_gt: [3] numpy array, GT translation in meters (absolute pose)
    #         K: [3, 3] numpy array, camera intrinsics for image1
    #         obj_id: object ID for getting 3D model
    #         sample_idx: sample index
    #         instance_id: instance id from batch
    #     """
    #     import cv2
    #     import numpy as np
    #     import torchvision.utils as vutils
    #     from PIL import Image

    #     # 创建保存目录
    #     save_dir = "debug_boundingbox_vis"
    #     if instance_id is not None:
    #         if hasattr(instance_id, 'item'):
    #             instance_id = instance_id.item()
    #         sample_dir = os.path.join(save_dir, f"sample_{instance_id}")
    #     else:
    #         sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
    #     os.makedirs(sample_dir, exist_ok=True)

    #     # 转换图像为 numpy 格式 [H, W, 3]
    #     img0_np = image0.detach().cpu().permute(1, 2, 0).numpy()
    #     img1_np = image1.detach().cpu().permute(1, 2, 0).numpy()

    #     # 归一化到 0-255
    #     img0_np = (img0_np - img0_np.min()) / (img0_np.max() - img0_np.min() + 1e-8) * 255
    #     img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min() + 1e-8) * 255
    #     img0_np = img0_np.astype(np.uint8).copy()
    #     img1_np = img1_np.astype(np.uint8).copy()

    #     # 获取物体的 3D bounding box 角点
    #     bbox_3d = self._get_3d_bbox(obj_id)

    #     if bbox_3d is None:
    #         print(f"[Warning] Cannot get 3D bbox for obj_id={obj_id}")
    #         # 仍然保存图像对
    #         concat_img = np.concatenate([img0_np, img1_np], axis=1)
    #         Image.fromarray(concat_img).save(os.path.join(sample_dir, "bbox_vis.png"))
    #         return

    #     # 将 3D bbox 转换到相机坐标系并投影到图像
    #     # bbox_3d: [8, 3] in mm, t_pred/t_gt in meters
    #     bbox_3d_m = bbox_3d / 1000.0  # 转换为米

    #     # ========== 绘制 GT Pose (绿色) ==========
    #     bbox_cam_gt = (R_gt @ bbox_3d_m.T).T + t_gt  # [8, 3]
    #     bbox_2d_gt = self._project_3d_to_2d(bbox_cam_gt, K)  # [8, 2]

    #     # 在 image1 上绘制 GT bounding box (绿色)
    #     img1_with_gt = self._draw_3d_bbox(img1_np.copy(), bbox_2d_gt, color=(0, 255, 0), thickness=2)

    #     # ========== 绘制 Pred Pose (红色) ==========
    #     bbox_cam_pred = (R_pred @ bbox_3d_m.T).T + t_pred  # [8, 3]
    #     bbox_2d_pred = self._project_3d_to_2d(bbox_cam_pred, K)  # [8, 2]

    #     # 在 image1 上绘制 Pred bounding box (红色)
    #     img1_with_pred = self._draw_3d_bbox(img1_np.copy(), bbox_2d_pred, color=(0, 0, 255), thickness=2)

    #     # ========== 绘制 GT + Pred 叠加 ==========
    #     img1_with_both = img1_np.copy()
    #     # 先绘制 GT (绿色)
    #     img1_with_both = self._draw_3d_bbox(img1_with_both, bbox_2d_gt, color=(0, 255, 0), thickness=2)
    #     # 再绘制 Pred (红色)
    #     img1_with_both = self._draw_3d_bbox(img1_with_both, bbox_2d_pred, color=(0, 0, 255), thickness=2)

    #     # ========== 创建多面板可视化（带白色标题区域）==========
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.8
    #     font_thickness = 2
    #     title_height = 50  # 白色标题区域高度

    #     # 辅助函数：添加白色标题区域
    #     def add_title_bar(img_left, img_right, title_left, title_right, color_right=(255, 255, 255)):
    #         """
    #         在拼接图像上方添加白色标题栏

    #         Args:
    #             img_left: 左侧图像
    #             img_right: 右侧图像
    #             title_left: 左侧标题文字
    #             title_right: 右侧标题文字
    #             color_right: 右侧标题颜色 (B, G, R)
    #         """
    #         H, W = img_left.shape[:2]

    #         # 创建白色标题栏
    #         title_bar = np.ones((title_height, W * 2, 3), dtype=np.uint8) * 255

    #         # 添加左侧标题（黑色文字）
    #         text_size_left = cv2.getTextSize(title_left, font, font_scale, font_thickness)[0]
    #         text_x_left = (W - text_size_left[0]) // 2
    #         text_y_left = (title_height + text_size_left[1]) // 2
    #         cv2.putText(title_bar, title_left, (text_x_left, text_y_left),
    #                    font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    #         # 添加右侧标题（彩色文字）
    #         text_size_right = cv2.getTextSize(title_right, font, font_scale, font_thickness)[0]
    #         text_x_right = W + (W - text_size_right[0]) // 2
    #         text_y_right = (title_height + text_size_right[1]) // 2
    #         cv2.putText(title_bar, title_right, (text_x_right, text_y_right),
    #                    font, font_scale, color_right, font_thickness, cv2.LINE_AA)

    #         # 拼接图像
    #         concat_img = np.concatenate([img_left, img_right], axis=1)

    #         # 垂直拼接标题栏和图像
    #         result = np.concatenate([title_bar, concat_img], axis=0)

    #         return result

    #     # 1. Anchor + Query with GT (绿色)
    #     concat_gt = add_title_bar(
    #         img0_np, img1_with_gt,
    #         "Anchor (Image0)",
    #         "Query + GT BBox (Green)",
    #         color_right=(0, 180, 0)  # 深绿色
    #     )

    #     # 2. Anchor + Query with Pred (红色)
    #     concat_pred = add_title_bar(
    #         img0_np, img1_with_pred,
    #         "Anchor (Image0)",
    #         "Query + Pred BBox (Red)",
    #         color_right=(0, 0, 200)  # 深红色
    #     )

    #     # 3. Anchor + Query with Both (绿色 GT + 红色 Pred)
    #     concat_both = add_title_bar(
    #         img0_np, img1_with_both,
    #         "Anchor (Image0)",
    #         "Query: GT (Green) vs Pred (Red)",
    #         color_right=(100, 100, 100)  # 深灰色
    #     )

    #     # ========== 保存所有版本 ==========
    #     Image.fromarray(concat_gt).save(os.path.join(sample_dir, "bbox_vis_gt.png"))
    #     Image.fromarray(concat_pred).save(os.path.join(sample_dir, "bbox_vis_pred.png"))
    #     Image.fromarray(concat_both).save(os.path.join(sample_dir, "bbox_vis_comparison.png"))

    #     # 保存单独的 image1 版本
    #     Image.fromarray(img1_with_gt).save(os.path.join(sample_dir, "image1_with_bbox_gt.png"))
    #     Image.fromarray(img1_with_pred).save(os.path.join(sample_dir, "image1_with_bbox_pred.png"))
    #     Image.fromarray(img1_with_both).save(os.path.join(sample_dir, "image1_with_bbox_both.png"))

    #     # 保存原图
    #     vutils.save_image(image0.unsqueeze(0), os.path.join(sample_dir, "image0.png"), normalize=True)
    #     vutils.save_image(image1.unsqueeze(0), os.path.join(sample_dir, "image1.png"), normalize=True)

    #     print(f"[Visualizer] 3D BBox visualization saved to: {sample_dir}/")
    #     print(f"  - bbox_vis_gt.png (GT only)")
    #     print(f"  - bbox_vis_pred.png (Pred only)")
    #     print(f"  - bbox_vis_comparison.png (GT + Pred overlay)")
    def debug_boundingbox_visualization(self, image0, image1, R_pred, t_pred, R_gt, t_gt,
                                        K, obj_id, sample_idx=0, instance_id=None):
        """
        [终极修复版]
        1. 修复了 R 矩阵自带缩放导致物体变小的问题 (SVD 归一化)。
        2. 修复了 BBox(mm) 和 t(m) 单位不统一的问题 (统一转为米)。
        """
        import cv2
        import numpy as np
        import os
        from PIL import Image

        # --- 0. 路径与图像预处理 ---
        save_dir = "debug_boundingbox_vis_ycbv"
        if instance_id is not None:
            if hasattr(instance_id, 'item'): instance_id = instance_id.item()
            sample_dir = os.path.join(save_dir, f"sample_{instance_id}")
        else:
            sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # 转换图像 [C,H,W] -> [H,W,C] (0-255 uint8)
        def to_numpy_img(tensor_img):
            img = tensor_img.detach().cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            return img.astype(np.uint8).copy()

        img0_np = to_numpy_img(image0)
        img1_np = to_numpy_img(image1)

        # --- 1. 获取原始 BBox ---
        bbox_3d = self._get_3d_bbox(obj_id)
        if bbox_3d is None:
            return

        # --- 2. [逻辑核心A] 单位统一化 -> 全部转为米 (Meters) ---
        # 判定规则：如果数值绝对值大于 10.0，认为是毫米(mm)，除以 1000

        # 处理物体尺寸
        bbox_3d_m = bbox_3d / 1000.0 if np.abs(bbox_3d).max() > 10.0 else bbox_3d

        # 处理 GT 平移
        t_gt_m = t_gt.copy()
        if np.abs(t_gt_m).max() > 10.0:
            t_gt_m = t_gt_m / 1000.0

        # 处理 Pred 平移
        t_pred_m = t_pred.copy()
        if np.abs(t_pred_m).max() > 10.0:
            t_pred_m = t_pred_m / 1000.0

        # --- 3. [逻辑核心B] 旋转矩阵归一化 (修复变小的主因) ---
        # 你的数据 R 模长只有 0.16，必须用 SVD 强制还原为单位旋转矩阵
        def normalize_rotation(R):
            U, _, Vt = np.linalg.svd(R)
            return U @ Vt

        R_gt_norm = normalize_rotation(R_gt)
        R_pred_norm = normalize_rotation(R_pred)

        # --- 4. 投影与绘制 ---
        # 公式: P_img = K * (R_norm * P_obj_m + t_m)

        # 绘制 GT (绿色)
        bbox_cam_gt = (R_gt_norm @ bbox_3d_m.T).T + t_gt_m
        bbox_2d_gt = self._project_3d_to_2d(bbox_cam_gt, K)
        img1_with_gt = self._draw_3d_bbox(img1_np.copy(), bbox_2d_gt, color=(0, 255, 0), thickness=2)

        # 绘制 Pred (红色)
        bbox_cam_pred = (R_pred_norm @ bbox_3d_m.T).T + t_pred_m
        bbox_2d_pred = self._project_3d_to_2d(bbox_cam_pred, K)
        img1_with_pred = self._draw_3d_bbox(img1_np.copy(), bbox_2d_pred, color=(0, 0, 255), thickness=2)

        # 绘制 叠加 (两者)
        img1_with_both = self._draw_3d_bbox(img1_np.copy(), bbox_2d_gt, color=(0, 255, 0), thickness=2)
        img1_with_both = self._draw_3d_bbox(img1_with_both, bbox_2d_pred, color=(0, 0, 255), thickness=2)

        # --- 5. 拼接与保存 ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_h = 40

        def stack_imgs(imgA, imgB, txtA, txtB, colB=(255, 255, 255)):
            h, w = imgA.shape[:2]
            bar = np.ones((title_h, w * 2, 3), dtype=np.uint8) * 50  # 灰色背景

            # 文字居中计算
            scale = 0.6
            thick = 2
            szA = cv2.getTextSize(txtA, font, scale, thick)[0]
            szB = cv2.getTextSize(txtB, font, scale, thick)[0]

            cv2.putText(bar, txtA, ((w - szA[0]) // 2, 28), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            cv2.putText(bar, txtB, (w + (w - szB[0]) // 2, 28), font, scale, colB, thick, cv2.LINE_AA)

            return np.vstack([bar, np.hstack([imgA, imgB])])

        # 生成对比图
        vis_gt = stack_imgs(img0_np, img1_with_gt, "Anchor Image", "Query + GT (Green)", (0, 255, 0))
        vis_pred = stack_imgs(img0_np, img1_with_pred, "Anchor Image", "Query + Pred (Red)", (0, 0, 255))
        vis_both = stack_imgs(img0_np, img1_with_both, "Anchor Image", "GT(Grn) vs Pred(Red)")

        # 保存
        try:
            Image.fromarray(vis_gt).save(os.path.join(sample_dir, "bbox_vis_gt.png"))
            Image.fromarray(vis_pred).save(os.path.join(sample_dir, "bbox_vis_pred.png"))
            Image.fromarray(vis_both).save(os.path.join(sample_dir, "bbox_vis_comparison.png"))
            print(f"[Visualizer] Saved corrected bbox (R_norm + Meters) to {sample_dir}")
        except Exception as e:
            print(f"[Error] Failed to save bbox visualization: {e}")

    # def _get_3d_bbox(self, obj_id):
    #     """
    #     获取物体的 3D bounding box 8个角点

    #     Returns:
    #         bbox_3d: [8, 3] numpy array in mm (scaled), or None if not available
    #     """
    #     if not hasattr(self, 'obj_models') or obj_id not in self.obj_models:
    #         return None

    #     # 从物体模型点云计算 bounding box
    #     obj_pts = self.obj_models[obj_id]['pts']  # [N, 3] normalized canonical (mm)

    #     print(f"\n[_get_3d_bbox Debug]")
    #     print(f"  obj_pts original range (mm):")
    #     print(f"    X: [{obj_pts[:, 0].min():.4f}, {obj_pts[:, 0].max():.4f}]")
    #     print(f"    Y: [{obj_pts[:, 1].min():.4f}, {obj_pts[:, 1].max():.4f}]")
    #     print(f"    Z: [{obj_pts[:, 2].min():.4f}, {obj_pts[:, 2].max():.4f}]")

    #     # ⚠️ 应用与 CAD 渲染相同的缩放
    #     if hasattr(self, 'obj_diams') and obj_id in self.obj_diams:
    #         diameter_mm = self.obj_diams[obj_id]
    #         cad_bbox = obj_pts.max(axis=0) - obj_pts.min(axis=0)
    #         cad_diameter_mm = np.linalg.norm(cad_bbox)

    #         print(f"  CAD diameter: {cad_diameter_mm:.4f} mm")
    #         print(f"  obj_diams diameter: {diameter_mm:.4f} mm")

    #         # ✅ 修复：始终信任 obj_diams，不做归一化猜测
    #         scale_factor = diameter_mm / (cad_diameter_mm + 1e-8)
    #         print(f"  [Fix] Force Scaling: CAD({cad_diameter_mm:.1f}) -> Target({diameter_mm:.1f}) | Factor: {scale_factor:.4f}")
    #         obj_pts = obj_pts * scale_factor

    #         print(f"  After scaling:")
    #         print(f"    X: [{obj_pts[:, 0].min():.4f}, {obj_pts[:, 0].max():.4f}]")
    #         print(f"    Y: [{obj_pts[:, 1].min():.4f}, {obj_pts[:, 1].max():.4f}]")
    #         print(f"    Z: [{obj_pts[:, 2].min():.4f}, {obj_pts[:, 2].max():.4f}]")

    #     # 计算 axis-aligned bounding box
    #     min_pt = obj_pts.min(axis=0)
    #     max_pt = obj_pts.max(axis=0)

    #     # 8个角点 (按标准顺序)
    #     bbox_3d = np.array([
    #         [min_pt[0], min_pt[1], min_pt[2]],  # 0: front-bottom-left
    #         [max_pt[0], min_pt[1], min_pt[2]],  # 1: front-bottom-right
    #         [max_pt[0], max_pt[1], min_pt[2]],  # 2: front-top-right
    #         [min_pt[0], max_pt[1], min_pt[2]],  # 3: front-top-left
    #         [min_pt[0], min_pt[1], max_pt[2]],  # 4: back-bottom-left
    #         [max_pt[0], min_pt[1], max_pt[2]],  # 5: back-bottom-right
    #         [max_pt[0], max_pt[1], max_pt[2]],  # 6: back-top-right
    #         [min_pt[0], max_pt[1], max_pt[2]],  # 7: back-top-left
    #     ], dtype=np.float32)

    #     print(f"  BBox corners range (mm):")
    #     print(f"    X: [{bbox_3d[:, 0].min():.4f}, {bbox_3d[:, 0].max():.4f}]")
    #     print(f"    Y: [{bbox_3d[:, 1].min():.4f}, {bbox_3d[:, 1].max():.4f}]")
    #     print(f"    Z: [{bbox_3d[:, 2].min():.4f}, {bbox_3d[:, 2].max():.4f}]")

    #     return bbox_3d
    def _get_3d_bbox(self, obj_id):
        """
        获取物体的 3D bounding box 8个角点
        [逻辑]: 只负责读取原始点云并计算极值，不进行任何单位转换或缩放。
        """
        if not hasattr(self, 'obj_models') or obj_id not in self.obj_models:
            print(f"[Warning] _get_3d_bbox: obj_id {obj_id} not found in obj_models")
            return None

        # 1. 获取原始点云 (假设是 BOP 格式，单位通常是 mm)
        obj_pts = self.obj_models[obj_id]['pts'].copy()

        # 2. 计算 AABB (Axis-Aligned Bounding Box)
        min_pt = obj_pts.min(axis=0)
        max_pt = obj_pts.max(axis=0)

        # 3. 生成 8 个角点
        bbox_3d = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ], dtype=np.float32)

        return bbox_3d

    def _project_3d_to_2d(self, pts_3d, K):
        """
        将 3D 点投影到 2D 图像平面

        Args:
            pts_3d: [N, 3] numpy array, 3D points in camera frame (meters)
            K: [3, 3] numpy array, camera intrinsics

        Returns:
            pts_2d: [N, 2] numpy array, 2D pixel coordinates
        """
        # pts_3d: [N, 3], K: [3, 3]
        # [u, v, w]^T = K @ [X, Y, Z]^T
        pts_homo = K @ pts_3d.T  # [3, N]

        print(f"      [_project_3d_to_2d] pts_homo before normalization:")
        print(f"        u: [{pts_homo[0, :].min():.2f}, {pts_homo[0, :].max():.2f}]")
        print(f"        v: [{pts_homo[1, :].min():.2f}, {pts_homo[1, :].max():.2f}]")
        print(f"        w: [{pts_homo[2, :].min():.6f}, {pts_homo[2, :].max():.6f}]")

        # 归一化
        pts_2d = pts_homo[:2, :] / (pts_homo[2, :] + 1e-8)  # [2, N]
        pts_2d = pts_2d.T  # [N, 2]

        print(f"      [_project_3d_to_2d] pts_2d after normalization:")
        print(f"        u: [{pts_2d[:, 0].min():.2f}, {pts_2d[:, 0].max():.2f}]")
        print(f"        v: [{pts_2d[:, 1].min():.2f}, {pts_2d[:, 1].max():.2f}]")

        return pts_2d.astype(np.float32)

    def _draw_3d_bbox(self, img, bbox_2d, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制 3D bounding box

        Args:
            img: [H, W, 3] numpy array
            bbox_2d: [8, 2] numpy array, 2D projected corners
            color: (B, G, R) tuple, default green
            thickness: line thickness

        Returns:
            img_with_bbox: [H, W, 3] numpy array
        """
        import cv2
        img_draw = img.copy()

        # 定义 bounding box 的 12 条边 (连接哪些角点)
        edges = [
            # Front face (z_min)
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Back face (z_max)
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Connecting edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # 绘制边
        for i, j in edges:
            pt1 = tuple(bbox_2d[i].astype(int))
            pt2 = tuple(bbox_2d[j].astype(int))

            # 检查点是否在图像范围内
            H, W = img.shape[:2]
            if (0 <= pt1[0] < W * 2 and 0 <= pt1[1] < H * 2 and
                    0 <= pt2[0] < W * 2 and 0 <= pt2[1] < H * 2):
                cv2.line(img_draw, pt1, pt2, color, thickness, cv2.LINE_AA)

        # # 绘制角点
        # for i in range(8):
        #     pt = tuple(bbox_2d[i].astype(int))
        #     H, W = img.shape[:2]
        #     if 0 <= pt[0] < W * 2 and 0 <= pt[1] < H * 2:
        #         # 使用相同颜色绘制角点，但稍微加深
        #         point_color = tuple(int(c * 0.7) for c in color)
        #         cv2.circle(img_draw, pt, 5, point_color, -1, cv2.LINE_AA)
        #         # 添加白色边框使角点更明显
        #         cv2.circle(img_draw, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

        return img_draw

    # =========================================================================
    #   [NEW] CAD Model Rendering Visualization
    #   在 query 图上渲染完整 CAD 模型（实线、线框、GT vs Pred 对比）
    # =========================================================================
    def debug_cad_rendering_visualization(self, image0, image1, R_pred, t_pred, R_gt, t_gt,
                                          K, obj_id, sample_idx=0, instance_id=None):
        """
        CAD 模型渲染可视化，包括：
        1. 实线渲染（RGB / 半透明）
        2. 线框渲染（wireframe）
        3. GT vs Pred 叠加对比（绿色 GT / 红色 Pred）

        Args:
            image0: [3, H, W] tensor, anchor image
            image1: [3, H, W] tensor, query image
            R_pred: [3, 3] numpy array, predicted rotation
            t_pred: [3] numpy array, predicted translation (meters)
            R_gt: [3, 3] numpy array, GT rotation
            t_gt: [3] numpy array, GT translation (meters)
            K: [3, 3] numpy array, camera intrinsics
            obj_id: object ID
            sample_idx: sample index
            instance_id: instance id from batch
        """
        import cv2
        import numpy as np
        import torchvision.utils as vutils
        from PIL import Image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\n" + "=" * 80)
        print(f"[CAD RENDERING DEBUG] Sample {sample_idx} (instance_id={instance_id})")
        print("=" * 80)

        # 创建保存目录
        save_dir = "debug_cad_rendering_vis"
        if instance_id is not None:
            if hasattr(instance_id, 'item'):
                instance_id = instance_id.item()
            sample_dir = os.path.join(save_dir, f"sample_{instance_id}")
        else:
            sample_dir = os.path.join(save_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # 转换图像为 numpy 格式
        img0_np = image0.detach().cpu().permute(1, 2, 0).numpy()
        img1_np = image1.detach().cpu().permute(1, 2, 0).numpy()

        # 归一化到 0-255
        img0_np = (img0_np - img0_np.min()) / (img0_np.max() - img0_np.min() + 1e-8) * 255
        img1_np = (img1_np - img1_np.min()) / (img1_np.max() - img1_np.min() + 1e-8) * 255
        img0_np = img0_np.astype(np.uint8).copy()
        img1_np = img1_np.astype(np.uint8).copy()

        print(f"\n[1] Image Info:")
        print(f"  Image0 shape: {img0_np.shape}")
        print(f"  Image1 shape: {img1_np.shape}")

        # 获取物体模型点云
        if not hasattr(self, 'obj_models') or obj_id not in self.obj_models:
            print(f"[Warning] Cannot get model for obj_id={obj_id}")
            return

        obj_pts = self.obj_models[obj_id]['pts']  # [N, 3] normalized canonical (mm)

        print(f"\n[2] CAD Model Info:")
        print(f"  obj_id: {obj_id}")
        print(f"  obj_pts shape: {obj_pts.shape}")
        print(f"  obj_pts dtype: {obj_pts.dtype}")
        print(f"  obj_pts original range (mm):")
        print(f"    X: [{obj_pts[:, 0].min():.4f}, {obj_pts[:, 0].max():.4f}]")
        print(f"    Y: [{obj_pts[:, 1].min():.4f}, {obj_pts[:, 1].max():.4f}]")
        print(f"    Z: [{obj_pts[:, 2].min():.4f}, {obj_pts[:, 2].max():.4f}]")

        # ⚠️ 关键：使用 diameter 进行尺度对齐
        # CAD 是 normalized canonical，需要用真实直径来缩放
        if hasattr(self, 'obj_diams') and obj_id in self.obj_diams:
            diameter_mm = self.obj_diams[obj_id]  # 真实物体直径（mm）

            # 计算 CAD 当前的直径
            cad_bbox = obj_pts.max(axis=0) - obj_pts.min(axis=0)
            cad_diameter_mm = np.linalg.norm(cad_bbox)

            print(f"\n[3] Scale Calculation:")
            print(f"  CAD bbox (mm): {cad_bbox}")
            print(f"  CAD diameter (normalized, mm): {cad_diameter_mm:.4f}")
            print(f"  Real diameter from obj_diams (mm): {diameter_mm:.4f}")

            # ✅ 修复：始终信任 obj_diams，不做归一化猜测
            # 只要分母不为0，始终计算缩放比例
            scale_factor = diameter_mm / (cad_diameter_mm + 1e-8)
            print(
                f"  [Fix] Force Scaling: CAD({cad_diameter_mm:.1f}) -> Target({diameter_mm:.1f}) | Factor: {scale_factor:.4f}")

            # 应用缩放并转换为米
            obj_pts_scaled_mm = obj_pts * scale_factor
            obj_pts_m = obj_pts_scaled_mm / 1000.0

            print(f"\n[4] After Scaling:")
            print(f"  obj_pts_scaled (mm) range:")
            print(f"    X: [{obj_pts_scaled_mm[:, 0].min():.4f}, {obj_pts_scaled_mm[:, 0].max():.4f}]")
            print(f"    Y: [{obj_pts_scaled_mm[:, 1].min():.4f}, {obj_pts_scaled_mm[:, 1].max():.4f}]")
            print(f"    Z: [{obj_pts_scaled_mm[:, 2].min():.4f}, {obj_pts_scaled_mm[:, 2].max():.4f}]")
            print(f"  obj_pts_m (meters) range:")
            print(f"    X: [{obj_pts_m[:, 0].min():.6f}, {obj_pts_m[:, 0].max():.6f}]")
            print(f"    Y: [{obj_pts_m[:, 1].min():.6f}, {obj_pts_m[:, 1].max():.6f}]")
            print(f"    Z: [{obj_pts_m[:, 2].min():.6f}, {obj_pts_m[:, 2].max():.6f}]")

            # 计算缩放后的直径验证
            scaled_bbox = obj_pts_m.max(axis=0) - obj_pts_m.min(axis=0)
            scaled_diameter_m = np.linalg.norm(scaled_bbox)
            print(f"  Final diameter (m): {scaled_diameter_m:.6f}")
            print(f"  Expected diameter (m): {diameter_mm / 1000.0:.6f}")
            print(f"  Diameter match: {np.isclose(scaled_diameter_m, diameter_mm / 1000.0, rtol=0.01)}")
        else:
            # 降级方案：假设已经是正确尺度
            print(f"\n[Warning] No diameter info for {obj_id}, assuming correct scale")
            obj_pts_m = obj_pts / 1000.0

        print(f"\n[5] Pose Info:")
        print(f"  R_pred shape: {R_pred.shape}, dtype: {R_pred.dtype}")
        print(f"  R_pred:\n{R_pred}")
        print(f"  t_pred (m): {t_pred} (shape: {t_pred.shape})")
        print(f"  R_gt shape: {R_gt.shape}, dtype: {R_gt.dtype}")
        print(f"  R_gt:\n{R_gt}")
        print(f"  t_gt (m): {t_gt} (shape: {t_gt.shape})")

        print(f"\n[6] Camera Intrinsics:")
        print(f"  K shape: {K.shape}, dtype: {K.dtype}")
        print(f"  K:\n{K}")
        print(f"  fx: {K[0, 0]:.2f}, fy: {K[1, 1]:.2f}")
        print(f"  cx: {K[0, 2]:.2f}, cy: {K[1, 2]:.2f}")

        # 降采样点云（加速渲染）
        original_num_pts = obj_pts_m.shape[0]
        if obj_pts_m.shape[0] > 5000:
            idx = np.random.choice(obj_pts_m.shape[0], 5000, replace=False)
            obj_pts_m = obj_pts_m[idx]
            print(f"\n[7] Downsampling:")
            print(f"  Original points: {original_num_pts}")
            print(f"  Downsampled to: {len(obj_pts_m)} points")
        else:
            print(f"\n[7] No downsampling needed ({original_num_pts} points)")

        # ========== 1. 实线渲染（半透明点云投影）==========
        print(f"\n[8] Rendering Pred Pose (Red)...")
        img1_solid_pred = self._render_solid_model(img1_np.copy(), obj_pts_m, R_pred, t_pred, K,
                                                   color=(0, 0, 255), alpha=0.6)

        print(f"\n[9] Rendering GT Pose (Green)...")
        img1_solid_gt = self._render_solid_model(img1_np.copy(), obj_pts_m, R_gt, t_gt, K,
                                                 color=(0, 255, 0), alpha=0.6)

        # ========== 2. 线框渲染（wireframe）==========
        img1_wireframe_pred = self._render_wireframe_model(img1_np.copy(), obj_pts_m, R_pred, t_pred, K,
                                                           color=(0, 0, 255), thickness=1)
        img1_wireframe_gt = self._render_wireframe_model(img1_np.copy(), obj_pts_m, R_gt, t_gt, K,
                                                         color=(0, 255, 0), thickness=1)

        # ========== 3. GT vs Pred 叠加对比 ==========
        img1_overlay = img1_np.copy()
        # 先渲染 GT (绿色)
        img1_overlay = self._render_solid_model(img1_overlay, obj_pts_m, R_gt, t_gt, K,
                                                color=(0, 255, 0), alpha=0.4)
        # 再渲染 Pred (红色)
        img1_overlay = self._render_solid_model(img1_overlay, obj_pts_m, R_pred, t_pred, K,
                                                color=(255, 0, 0), alpha=0.4)

        # 线框叠加版本
        img1_wireframe_overlay = img1_np.copy()
        img1_wireframe_overlay = self._render_wireframe_model(img1_wireframe_overlay, obj_pts_m,
                                                              R_gt, t_gt, K, color=(0, 255, 0), thickness=2)
        img1_wireframe_overlay = self._render_wireframe_model(img1_wireframe_overlay, obj_pts_m,
                                                              R_pred, t_pred, K, color=(255, 0, 0), thickness=2)

        # ========== 创建多面板可视化 ==========
        fig = plt.figure(figsize=(20, 12))

        # 2x3 布局
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img0_np)
        ax1.set_title("(a) Anchor Image (Image0)", fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(img1_solid_pred)
        ax2.set_title("(b) Pred Pose - Solid Rendering", fontsize=12, fontweight='bold', color='red')
        ax2.axis('off')

        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(img1_solid_gt)
        ax3.set_title("(c) GT Pose - Solid Rendering", fontsize=12, fontweight='bold', color='green')
        ax3.axis('off')

        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(img1_wireframe_pred)
        ax4.set_title("(d) Pred Pose - Wireframe", fontsize=12, fontweight='bold', color='red')
        ax4.axis('off')

        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(img1_wireframe_gt)
        ax5.set_title("(e) GT Pose - Wireframe", fontsize=12, fontweight='bold', color='green')
        ax5.axis('off')

        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(img1_overlay)
        ax6.set_title("(f) GT (Green) vs Pred (Red) Overlay", fontsize=12, fontweight='bold')
        ax6.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "cad_rendering_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # ========== 保存单独的高质量图像 ==========
        Image.fromarray(img1_solid_pred).save(os.path.join(sample_dir, "solid_pred.png"))
        Image.fromarray(img1_solid_gt).save(os.path.join(sample_dir, "solid_gt.png"))
        Image.fromarray(img1_wireframe_pred).save(os.path.join(sample_dir, "wireframe_pred.png"))
        Image.fromarray(img1_wireframe_gt).save(os.path.join(sample_dir, "wireframe_gt.png"))
        Image.fromarray(img1_overlay).save(os.path.join(sample_dir, "overlay_solid.png"))
        Image.fromarray(img1_wireframe_overlay).save(os.path.join(sample_dir, "overlay_wireframe.png"))

        print(f"\n[10] Saved individual images to: {sample_dir}/")

        # ========== 创建左右对比图（论文风格）==========
        # 左: Anchor, 右: Query with GT (green) + Pred (red) overlay
        concat_paper = np.concatenate([img0_np, img1_wireframe_overlay], axis=1)

        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(concat_paper, "Anchor (Image0)", (10, 30), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(concat_paper, "Query: GT (Green) vs Pred (Red)", (img0_np.shape[1] + 10, 30),
                    font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        Image.fromarray(concat_paper).save(os.path.join(sample_dir, "paper_style_comparison.png"))

        print(f"\n[11] Summary:")
        print(f"  ✓ All visualizations saved to: {sample_dir}/")
        print(f"  ✓ Check the following files:")
        print(f"    - cad_rendering_comparison.png (multi-panel)")
        print(f"    - solid_pred.png / solid_gt.png")
        print(f"    - wireframe_pred.png / wireframe_gt.png")
        print(f"    - overlay_solid.png / overlay_wireframe.png")
        print(f"    - paper_style_comparison.png")
        print("=" * 80 + "\n")

    def _render_solid_model(self, img, obj_pts, R, t, K, color=(0, 255, 0), alpha=0.5):
        """
        实线渲染：将 3D 模型点云投影到图像并以半透明方式叠加

        Args:
            img: [H, W, 3] numpy array
            obj_pts: [N, 3] numpy array, 3D points in meters
            R: [3, 3] rotation matrix
            t: [3] translation vector in meters
            K: [3, 3] camera intrinsics
            color: (B, G, R) tuple
            alpha: transparency (0-1)
        """
        import cv2
        import numpy as np

        print(f"\n  [_render_solid_model] Starting...")
        print(f"    Input obj_pts shape: {obj_pts.shape}")
        print(f"    Input obj_pts range (m):")
        print(f"      X: [{obj_pts[:, 0].min():.6f}, {obj_pts[:, 0].max():.6f}]")
        print(f"      Y: [{obj_pts[:, 1].min():.6f}, {obj_pts[:, 1].max():.6f}]")
        print(f"      Z: [{obj_pts[:, 2].min():.6f}, {obj_pts[:, 2].max():.6f}]")
        print(f"    R shape: {R.shape}")
        print(f"    t shape: {t.shape}, value: {t}")
        print(f"    K:\n{K}")
        print(f"    Color: {color}, Alpha: {alpha}")

        # 变换点云到相机坐标系
        # P_cam = R * P_obj + t
        pts_cam = (R @ obj_pts.T).T + t  # [N, 3]

        print(f"    After transformation (pts_cam):")
        print(f"      X: [{pts_cam[:, 0].min():.6f}, {pts_cam[:, 0].max():.6f}]")
        print(f"      Y: [{pts_cam[:, 1].min():.6f}, {pts_cam[:, 1].max():.6f}]")
        print(f"      Z: [{pts_cam[:, 2].min():.6f}, {pts_cam[:, 2].max():.6f}]")

        # 过滤掉相机后面的点
        valid_mask = pts_cam[:, 2] > 0
        num_valid = valid_mask.sum()
        print(f"    Valid points (Z>0): {num_valid} / {len(valid_mask)} ({100 * num_valid / len(valid_mask):.1f}%)")

        pts_cam = pts_cam[valid_mask]

        if len(pts_cam) == 0:
            print(f"    ❌ No valid points after Z filtering!")
            return img

        # 投影到图像平面
        pts_2d = self._project_3d_to_2d(pts_cam, K)

        print(f"    After projection (pts_2d):")
        print(f"      X: [{pts_2d[:, 0].min():.1f}, {pts_2d[:, 0].max():.1f}]")
        print(f"      Y: [{pts_2d[:, 1].min():.1f}, {pts_2d[:, 1].max():.1f}]")

        # 创建叠加层
        overlay = img.copy()
        H, W = img.shape[:2]
        print(f"    Image size: {W}x{H}")

        # 绘制点
        n_drawn = 0
        n_out_of_bounds = 0
        for pt in pts_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (x, y), 2, color, -1, cv2.LINE_AA)
                n_drawn += 1
            else:
                n_out_of_bounds += 1

        print(f"    ✓ Drew {n_drawn} points (color={color})")
        print(f"    ✗ Out of bounds: {n_out_of_bounds} points")
        print(f"    Drawing success rate: {100 * n_drawn / (n_drawn + n_out_of_bounds):.1f}%")

        # 半透明叠加
        img_result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return img_result

    def _render_wireframe_model(self, img, obj_pts, R, t, K, color=(0, 255, 0), thickness=1):
        """
        线框渲染：使用 Delaunay 三角化或凸包绘制线框

        Args:
            img: [H, W, 3] numpy array
            obj_pts: [N, 3] numpy array, 3D points in meters
            R: [3, 3] rotation matrix
            t: [3] translation vector in meters
            K: [3, 3] camera intrinsics
            color: (B, G, R) tuple
            thickness: line thickness
        """
        import cv2
        import numpy as np
        from scipy.spatial import ConvexHull

        # 变换点云到相机坐标系
        pts_cam = (R @ obj_pts.T).T + t  # [N, 3]

        # 过滤掉相机后面的点
        valid_mask = pts_cam[:, 2] > 0
        pts_cam = pts_cam[valid_mask]

        if len(pts_cam) < 4:
            return img

        # 投影到图像平面
        pts_2d = self._project_3d_to_2d(pts_cam, K)

        H, W = img.shape[:2]

        # 过滤图像范围外的点
        valid_2d = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & \
                   (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)

        if valid_2d.sum() < 4:
            return img

        pts_2d_valid = pts_2d[valid_2d]

        try:
            # 使用凸包绘制外轮廓
            hull = ConvexHull(pts_2d_valid)

            # 绘制凸包边
            for simplex in hull.simplices:
                pt1 = tuple(pts_2d_valid[simplex[0]].astype(int))
                pt2 = tuple(pts_2d_valid[simplex[1]].astype(int))
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

            # 额外绘制一些内部连接（增强线框效果）
            # 随机连接一些点对
            n_extra_lines = min(50, len(pts_2d_valid) // 10)
            if n_extra_lines > 0:
                indices = np.random.choice(len(pts_2d_valid), size=(n_extra_lines, 2), replace=True)
                for i, j in indices:
                    if i != j:
                        pt1 = tuple(pts_2d_valid[i].astype(int))
                        pt2 = tuple(pts_2d_valid[j].astype(int))
                        # 只绘制距离不太远的点对
                        dist = np.linalg.norm(pts_2d_valid[i] - pts_2d_valid[j])
                        if dist < 50:  # 像素距离阈值
                            cv2.line(img, pt1, pt2, color, max(1, thickness - 1), cv2.LINE_AA)

        except Exception as e:
            print(f"[Warning] Wireframe rendering failed: {e}")
            # 降级方案：只绘制点
            for pt in pts_2d_valid:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)

        return img

    def _draw_superglue_style_comparison(
            self, img0, img1, mask0, mask1,
            mkpts0_raw, mkpts1_raw, mconf_raw,
            mkpts0_masked, mkpts1_masked, mconf_masked,
            save_path, max_lines=60,
            title_ours="(b) Ours: Mask-Guided Matching (Foreground-Focused)"
    ):
        """
        SuperGlue 论文风格的匹配可视化
        上下两行对比: Baseline vs Ours
        """
        import cv2
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        H, W = img0.shape[:2]

        # 归一化 mask
        if mask0.max() > 1:
            mask0 = mask0 / 255.0
        if mask1.max() > 1:
            mask1 = mask1 / 255.0

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # ===== 上排: Baseline (无 Mask) =====
        ax_top = axes[0]
        concat_raw = np.concatenate([img0, img1], axis=1)
        ax_top.imshow(concat_raw)
        ax_top.set_title("(a) Baseline: Direct Matching without Mask Guidance",
                         fontsize=14, fontweight='bold', color='darkred', pad=10)
        ax_top.axis('off')

        # 绘制匹配线
        n_raw = min(len(mkpts0_raw), max_lines)
        n_bg_matches = 0
        n_fg_matches = 0

        if n_raw > 0:
            idx_sort = np.argsort(-mconf_raw)[:n_raw]
            for idx in idx_sort:
                p0 = mkpts0_raw[idx]
                p1 = mkpts1_raw[idx].copy()
                p1[0] += W

                y0, x0 = int(round(mkpts0_raw[idx][1])), int(round(mkpts0_raw[idx][0]))
                y1, x1 = int(round(mkpts1_raw[idx][1])), int(round(mkpts1_raw[idx][0]))

                in0 = (0 <= y0 < H and 0 <= x0 < W and mask0[y0, x0] > 0.5)
                in1 = (0 <= y1 < H and 0 <= x1 < W and mask1[y1, x1] > 0.5)

                if in0 and in1:
                    color = '#FFA500'  # orange for foreground
                    alpha = 0.7
                    lw = 1.2
                    n_fg_matches += 1
                else:
                    color = '#FF4444'  # red for background
                    alpha = 0.5
                    lw = 0.8
                    n_bg_matches += 1

                ax_top.plot([p0[0], p1[0]], [p0[1], p1[1]],
                            color=color, alpha=alpha, linewidth=lw)
                ax_top.scatter([p0[0]], [p0[1]], c=color, s=15, alpha=alpha, zorder=5)
                ax_top.scatter([p1[0]], [p1[1]], c=color, s=15, alpha=alpha, zorder=5)

        # 添加图例和统计
        ax_top.text(10, 30, f"Total: {len(mkpts0_raw)} matches", fontsize=11, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        ax_top.text(10, 60, f"❌ Background outliers: {n_bg_matches} ({100 * n_bg_matches / max(n_raw, 1):.1f}%)",
                    fontsize=11, color='#FF6666',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

        # ===== 下排: Ours (Mask-Guided) =====
        ax_bot = axes[1]

        # 创建背景变暗的图像
        img0_dark = img0.copy().astype(np.float32)
        img1_dark = img1.copy().astype(np.float32)

        darken = 0.25
        for c in range(3):
            img0_dark[:, :, c][mask0 < 0.5] *= darken
            img1_dark[:, :, c][mask1 < 0.5] *= darken

        img0_dark = np.clip(img0_dark, 0, 255).astype(np.uint8)
        img1_dark = np.clip(img1_dark, 0, 255).astype(np.uint8)

        # 添加 mask 轮廓
        mask0_uint8 = (mask0 * 255).astype(np.uint8)
        mask1_uint8 = (mask1 * 255).astype(np.uint8)
        contours0, _ = cv2.findContours(mask0_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(mask1_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img0_dark, contours0, -1, (0, 255, 255), 2)
        cv2.drawContours(img1_dark, contours1, -1, (0, 255, 255), 2)

        concat_masked = np.concatenate([img0_dark, img1_dark], axis=1)
        ax_bot.imshow(concat_masked)
        ax_bot.set_title(title_ours,
                         fontsize=14, fontweight='bold', color='darkgreen', pad=10)
        ax_bot.axis('off')

        # 绘制匹配线 (全部绿色)
        n_masked = min(len(mkpts0_masked), max_lines)
        if n_masked > 0:
            idx_sort = np.argsort(-mconf_masked)[:n_masked]
            for idx in idx_sort:
                p0 = mkpts0_masked[idx]
                p1 = mkpts1_masked[idx].copy()
                p1[0] += W
                conf = mconf_masked[idx]

                # 根据置信度调整颜色深浅
                green_val = int(180 + 75 * conf)
                color = f'#{0:02x}{green_val:02x}{0:02x}'
                alpha = 0.6 + 0.4 * conf
                lw = 1.0 + 1.5 * conf

                ax_bot.plot([p0[0], p1[0]], [p0[1], p1[1]],
                            color='lime', alpha=alpha, linewidth=lw)
                ax_bot.scatter([p0[0]], [p0[1]], c='lime', s=20, alpha=alpha,
                               edgecolors='darkgreen', linewidths=0.5, zorder=5)
                ax_bot.scatter([p1[0]], [p1[1]], c='lime', s=20, alpha=alpha,
                               edgecolors='darkgreen', linewidths=0.5, zorder=5)

        # 添加统计
        ax_bot.text(10, 30, f"Total: {len(mkpts0_masked)} matches", fontsize=11, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        ax_bot.text(10, 60, f"✓ All foreground (0% outliers)", fontsize=11, color='#66FF66',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

        # 底部总结
        fig.text(0.5, 0.02,
                 "Mask guidance significantly reduces background outliers and focuses matching on the target object.",
                 ha='center', fontsize=12, style='italic', color='#333333')

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Saved SuperGlue-style comparison to: {save_path}")

    def _draw_three_way_comparison(
            self, img0, img1,
            mask0_gt, mask1_gt,
            mask0_pred, mask1_pred,
            mkpts0_raw, mkpts1_raw, mconf_raw,
            mkpts0_gt_masked, mkpts1_gt_masked, mconf_gt_masked,
            mkpts0_pred_masked, mkpts1_pred_masked, mconf_pred_masked,
            save_path, max_lines=50
    ):
        """
        三方对比图: Baseline vs GT Mask vs Pred Mask
        三行布局，展示不同掩码策略的效果
        """
        import cv2
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        H, W = img0.shape[:2]

        # 归一化 mask
        if mask0_gt.max() > 1:
            mask0_gt = mask0_gt / 255.0
        if mask1_gt.max() > 1:
            mask1_gt = mask1_gt / 255.0
        if mask0_pred.max() > 1:
            mask0_pred = mask0_pred / 255.0
        if mask1_pred.max() > 1:
            mask1_pred = mask1_pred / 255.0

        fig, axes = plt.subplots(3, 1, figsize=(16, 16))

        # ===== 第一行: Baseline (无 Mask) =====
        ax_top = axes[0]
        concat_raw = np.concatenate([img0, img1], axis=1)
        ax_top.imshow(concat_raw)
        ax_top.set_title("(a) Baseline: Direct Matching without Mask Guidance",
                         fontsize=13, fontweight='bold', color='darkred', pad=8)
        ax_top.axis('off')

        n_raw = min(len(mkpts0_raw), max_lines)
        n_bg_raw = 0
        if n_raw > 0:
            idx_sort = np.argsort(-mconf_raw)[:n_raw]
            for idx in idx_sort:
                p0 = mkpts0_raw[idx]
                p1 = mkpts1_raw[idx].copy()
                p1[0] += W

                y0, x0 = int(round(mkpts0_raw[idx][1])), int(round(mkpts0_raw[idx][0]))
                y1, x1 = int(round(mkpts1_raw[idx][1])), int(round(mkpts1_raw[idx][0]))

                in0 = (0 <= y0 < H and 0 <= x0 < W and mask0_gt[y0, x0] > 0.5)
                in1 = (0 <= y1 < H and 0 <= x1 < W and mask1_gt[y1, x1] > 0.5)

                if in0 and in1:
                    color = '#FFA500'
                    alpha = 0.7
                else:
                    color = '#FF4444'
                    alpha = 0.5
                    n_bg_raw += 1

                ax_top.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, alpha=alpha, linewidth=0.8)
                ax_top.scatter([p0[0]], [p0[1]], c=color, s=10, alpha=alpha, zorder=5)
                ax_top.scatter([p1[0]], [p1[1]], c=color, s=10, alpha=alpha, zorder=5)

        ax_top.text(10, 25,
                    f"Total: {len(mkpts0_raw)} | BG outliers: {n_bg_raw} ({100 * n_bg_raw / max(len(mkpts0_raw), 1):.1f}%)",
                    fontsize=10, color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

        # ===== 第二行: GT Mask =====
        ax_mid = axes[1]

        img0_dark_gt = img0.copy().astype(np.float32)
        img1_dark_gt = img1.copy().astype(np.float32)
        darken = 0.25
        for c in range(3):
            img0_dark_gt[:, :, c][mask0_gt < 0.5] *= darken
            img1_dark_gt[:, :, c][mask1_gt < 0.5] *= darken
        img0_dark_gt = np.clip(img0_dark_gt, 0, 255).astype(np.uint8)
        img1_dark_gt = np.clip(img1_dark_gt, 0, 255).astype(np.uint8)

        # 添加轮廓
        mask0_gt_uint8 = (mask0_gt * 255).astype(np.uint8)
        mask1_gt_uint8 = (mask1_gt * 255).astype(np.uint8)
        contours0, _ = cv2.findContours(mask0_gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(mask1_gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img0_dark_gt, contours0, -1, (0, 255, 255), 2)
        cv2.drawContours(img1_dark_gt, contours1, -1, (0, 255, 255), 2)

        concat_gt = np.concatenate([img0_dark_gt, img1_dark_gt], axis=1)
        ax_mid.imshow(concat_gt)
        ax_mid.set_title("(b) GT Mask-Guided Matching (Oracle Upper Bound)",
                         fontsize=13, fontweight='bold', color='darkblue', pad=8)
        ax_mid.axis('off')

        n_gt = min(len(mkpts0_gt_masked), max_lines)
        if n_gt > 0:
            idx_sort = np.argsort(-mconf_gt_masked)[:n_gt]
            for idx in idx_sort:
                p0 = mkpts0_gt_masked[idx]
                p1 = mkpts1_gt_masked[idx].copy()
                p1[0] += W
                conf = mconf_gt_masked[idx]
                alpha = 0.6 + 0.4 * conf
                ax_mid.plot([p0[0], p1[0]], [p0[1], p1[1]], color='cyan', alpha=alpha, linewidth=1.0)
                ax_mid.scatter([p0[0]], [p0[1]], c='cyan', s=15, alpha=alpha, edgecolors='darkblue', linewidths=0.5,
                               zorder=5)
                ax_mid.scatter([p1[0]], [p1[1]], c='cyan', s=15, alpha=alpha, edgecolors='darkblue', linewidths=0.5,
                               zorder=5)

        ax_mid.text(10, 25, f"Total: {len(mkpts0_gt_masked)} | All foreground (0% outliers)",
                    fontsize=10, color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

        # ===== 第三行: Pred Mask =====
        ax_bot = axes[2]

        img0_dark_pred = img0.copy().astype(np.float32)
        img1_dark_pred = img1.copy().astype(np.float32)
        for c in range(3):
            img0_dark_pred[:, :, c][mask0_pred < 0.5] *= darken
            img1_dark_pred[:, :, c][mask1_pred < 0.5] *= darken
        img0_dark_pred = np.clip(img0_dark_pred, 0, 255).astype(np.uint8)
        img1_dark_pred = np.clip(img1_dark_pred, 0, 255).astype(np.uint8)

        # 添加轮廓
        mask0_pred_uint8 = (mask0_pred * 255).astype(np.uint8)
        mask1_pred_uint8 = (mask1_pred * 255).astype(np.uint8)
        contours0_pred, _ = cv2.findContours(mask0_pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1_pred, _ = cv2.findContours(mask1_pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img0_dark_pred, contours0_pred, -1, (0, 255, 0), 2)
        cv2.drawContours(img1_dark_pred, contours1_pred, -1, (0, 255, 0), 2)

        concat_pred = np.concatenate([img0_dark_pred, img1_dark_pred], axis=1)
        ax_bot.imshow(concat_pred)
        ax_bot.set_title("(c) Pred Mask-Guided Matching (Ours)",
                         fontsize=13, fontweight='bold', color='darkgreen', pad=8)
        ax_bot.axis('off')

        n_pred = min(len(mkpts0_pred_masked), max_lines)
        if n_pred > 0:
            idx_sort = np.argsort(-mconf_pred_masked)[:n_pred]
            for idx in idx_sort:
                p0 = mkpts0_pred_masked[idx]
                p1 = mkpts1_pred_masked[idx].copy()
                p1[0] += W
                conf = mconf_pred_masked[idx]
                alpha = 0.6 + 0.4 * conf
                ax_bot.plot([p0[0], p1[0]], [p0[1], p1[1]], color='lime', alpha=alpha, linewidth=1.0)
                ax_bot.scatter([p0[0]], [p0[1]], c='lime', s=15, alpha=alpha, edgecolors='darkgreen', linewidths=0.5,
                               zorder=5)
                ax_bot.scatter([p1[0]], [p1[1]], c='lime', s=15, alpha=alpha, edgecolors='darkgreen', linewidths=0.5,
                               zorder=5)

        ax_bot.text(10, 25, f"Total: {len(mkpts0_pred_masked)} | All foreground (0% outliers)",
                    fontsize=10, color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

        # 底部总结
        fig.text(0.5, 0.01,
                 "Comparison: (a) Baseline has many background outliers → (b) GT Mask provides oracle upper bound → (c) Pred Mask achieves comparable quality",
                 ha='center', fontsize=11, style='italic', color='#333333')

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Saved three-way comparison to: {save_path}")
