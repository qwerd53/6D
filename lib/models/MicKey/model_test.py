import torch
import pytorch_lightning as pl

from lib.models.MicKey.modules.loss.loss_class import MetricPoseLoss
from lib.models.MicKey.modules.compute_correspondences import ComputeCorrespondences
from lib.models.MicKey.modules.utils.training_utils import log_image_matches, debug_reward_matches_log, vis_inliers,log_mask_images
from lib.models.MicKey.modules.utils.probabilisticProcrustes import e2eProbabilisticProcrustesSolver

from lib.utils.metrics import pose_error_torch, vcre_torch
from lib.benchmarks.utils import precision_recall
from lib.models.Oryon.oryon import Oryon
#metric
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
# =========================
#   MicKeyTrainingModel
# =========================
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
        #self._mask_loss = nn.BCEWithLogitsLoss(reduction='mean')  # 用 logits
        self._mask_loss =DiceLoss(weight=torch.tensor([0.5, 0.5]))
        self.mask_th = getattr(getattr(cfg, 'LOSS', {}), 'MASK_TH', 0.5)
        self.soft_clip = getattr(getattr(cfg, 'LOSS', {}), 'SOFT_CLIP', True)

        # ---------- 训练控制 ----------
        self.automatic_optimization = True  #  Lightning 自动优化
        self.multi_gpu = True
        self.validation_step_outputs = []
        self.log_interval = getattr(cfg.TRAINING, 'LOG_INTERVAL', 50)

        # 半 epoch 评估控制
        self._ran_half_eval_for_epoch = False
        self._half_epoch_batch_idx = None  # 每个 epoch 开头计算

        # VSD渲染器
        self.compute_vsd=True
        self.vsd_renderer = RendererVispy(640, 480, mode='depth')
        self.vsd_taus = list(np.arange(0.05, 0.51, 0.05))
        self.vsd_rec = np.arange(0.05, 0.51, 0.05)
        self.vsd_delta = 15.
        self._vsd_objects_loaded = False

        # 初始化指标记录
        self.validation_step_outputs = []
        self.test_step_outputs = []




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
        gt_shape = gt.shape[1:]
        pred_shape = pred_logits.shape[2:]
        #print("pred_shape", pred_shape)
        #print("gt_shape", gt_shape)
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
        loss_rot, _ = self.rot_angle_loss(R, Rgt_i)          # [B,1]
        loss_trans = self.trans_l1_loss(t, tgt_i)            # [B,1,3] -> [B,1]

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
        mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, mask0_gt)
        mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, mask1_gt)
        mask_loss_all = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2.

        # 3) logits -> 概率
        pred_mask0_prob = torch.sigmoid(pred_mask0_logits).squeeze(1)  # [B,Hm,Wm]
        pred_mask1_prob = torch.sigmoid(pred_mask1_logits).squeeze(1)  # [B,Hm,Wm]

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

        # 6) 灰度图过滤
        img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
        img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)

        # 7) LoFTR 匹配
        R_preds, t_preds = [], []
        for i in range(B):
            match_batch = {'image0': img0_gray[i:i + 1], 'image1': img1_gray[i:i + 1]}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()
            mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()

            m0 = pred_mask0_bin[i].detach().cpu().numpy()
            m1 = pred_mask1_bin[i].detach().cpu().numpy()

            if len(mkpts0) == 0:
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))
                continue

            # 按掩码过滤关键点
            in_mask = (m0[mkpts0[:, 1].round().astype(int),
            mkpts0[:, 0].round().astype(int)] > 0) & \
                      (m1[mkpts1[:, 1].round().astype(int),
                      mkpts1[:, 0].round().astype(int)] > 0)
            mkpts0 = mkpts0[in_mask]
            mkpts1 = mkpts1[in_mask]

            if len(mkpts0) < 3:
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))
                continue

            # 8) 回投影到 3D
            depth0 = batch['depth0'][i].detach().cpu().numpy()
            depth1 = batch['depth1'][i].detach().cpu().numpy()
            K0 = batch['K_color0'][i].detach().cpu().numpy()
            K1 = batch['K_color1'][i].detach().cpu().numpy()


            current_h, current_w = batch['image0'][i].shape[1:]
            K0 = self.adjust_intrinsics_for_resize(K0, original_size=(640, 480),
                                                   current_size=(current_w, current_h))
            K1 = self.adjust_intrinsics_for_resize(K1, original_size=(640, 480),
                                                   current_size=(current_w, current_h))

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
            R_t = torch.from_numpy(R_np).float().to(device)
            t_t = torch.from_numpy(t_np).float().to(device).unsqueeze(0) / 1000.0

            # 10) 转到绝对位姿
            T_a = batch['item_a_pose'][i].detach().cpu().numpy()
            T_rel = np.eye(4, dtype=np.float32)
            T_rel[:3, :3] = R_np
            T_rel[:3, 3] = (t_np / 1000.0)
            T_q_pred = T_rel @ T_a
            R_q = torch.from_numpy(T_q_pred[:3, :3]).float().to(device)
            t_q = torch.from_numpy(T_q_pred[:3, 3]).float().to(device).unsqueeze(0)

            R_preds.append(R_q)
            t_preds.append(t_q)

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

                #print("gt_pose",gt_pose)
                obj_sym = self.obj_symms.get(obj_id, None)
                #处理好的 ndarray
                bop_sym = obj_sym

                # MSSD/MSPD
                # mssd_err = my_mssd(pred_R, pred_t, gt_R, gt_t, obj_model['pts'], bop_sym)
                # mspd_err = my_mspd(pred_R, pred_t, gt_R, gt_t, K, obj_model['pts'], bop_sym)
                mssd_err = my_mssd(pred_R, pred_t, gt_R, gt_t, obj_model['pts'], bop_sym)
                mspd_err = my_mspd(pred_R, pred_t, gt_R, gt_t, K, obj_model['pts'], bop_sym)

                # VSD
                #print(depth.shape)
                #
                #depth_proc = depth.copy()
                #depth = depth.astype(np.int32)
                #depth_proc[depth_proc == 0] = depth_proc.max()
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
                #test
                #print("obj_diam:",obj_diam)


                #print("K.shape",K.shape)
                # # 处理 depth  0
                # depth_proc = depth.copy()
                # depth_proc[depth_proc == 0] = depth_proc.max()
                # vsd_err = vsd(pred_R, pred_t, gt_R, gt_t, depth_proc, K,
                #               self.vsd_delta, self.vsd_taus, True, obj_diam,
                #               self.vsd_renderer, obj_id)

            # 计算 recall 分数
            mssd_rec = np.arange(0.05, 0.51, 0.05) * obj_diam
            mssd_scores.append((mssd_err < mssd_rec).mean())#if 'mssd_err' in locals() else 0.0)

            mspd_rec = np.arange(5, 51, 5)
            mspd_scores.append((mspd_err < mspd_rec).mean()) #if 'mspd_err' in locals() else 0.0)

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
            'vsd': torch.tensor(mean_vsd)#torch.tensor(np.mean(vsd_scores) if vsd_scores else 0.0),
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
            R_pred = out['R_pred']           # [B,3,3]
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

    def normalize_pose(self,R, t):
        # R: 3x3 旋转矩阵
        # t: 3x1 平移向量
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:  # 处理反射
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho, t

