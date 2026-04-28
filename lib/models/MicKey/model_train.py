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

# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from omegaconf import OmegaConf

# ==== 澶栭儴渚濊禆锛堜繚鎸佷笌浣犲伐绋嬬殑涓€鑷达級 ====
from lib.models.Oryon.oryon import Oryon

# FeatureMatcher
from MATCHER_CORE.src.matcher import FeatureMatcher, default_cfg

from lib.utils.metrics import pose_error_torch  # 浠呯敤浜庡彲閫夊榻愭鏌ワ紙鏈敤浜巐oss锛?
from lib.benchmarks.utils import precision_recall  # 鏃ュ織
from Oryon.utils.metrics import compute_add, compute_adds  # 鐢ㄤ簬 ADD/ADD-S
from Oryon.utils.geo6d import best_fit_transform_with_RANSAC  # 鍙€夌殑RANSAC
# from Oryon.utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver  # PointDSC
from Oryon.utils.losses import DiceLoss, LovaszLoss, FocalLoss
from Oryon.utils.metrics import mask_iou
from Oryon.losses import FeatureLoss

from filesOfcatseg.cat_seg.cat_seg_model import CATSeg

# For CATSeg config
from detectron2.config import get_cfg
from filesOfcatseg.cat_seg.config import add_cat_seg_config


# =========================
#   MicKeyTrainingModel
# =========================
class MicKeyTrainingModel(pl.LightningModule):
    """
    - 淇濈暀 Oryon 浜х敓 mask 鐨勬ā鍧?
    - FeatureMatcher 杩涜鍖归厤 + 鎺╃爜杩囨护 + 鍥炴姇褰?鈫?Kabsch 姹傚Э鎬?
    - 鎹熷け锛歮ask_loss锛圔CEWithLogits锛?+ compute_pose_loss锛坮ot_angle + trans_l1锛屽彲閫塼anh clipping锛?
    - 姣忓崐涓?epoch 璺戜竴娆?ADD(S)-0.1D 璇勪及骞惰褰曞埌 TensorBoard
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg

        # ---------- Oryon ----------
        self.oryon_model = Oryon(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')
        # catseg
        # # ---------- CATSeg ----------
        # # 鏋勫缓閰嶇疆瀵硅薄
        # catseg_cfg = get_cfg()
        # add_cat_seg_config(catseg_cfg)
        # # 鍔犺浇閰嶇疆鏂囦欢
        # catseg_cfg.merge_from_file("filesOfcatseg/configs/config.yaml")
        # catseg_cfg.freeze()
        #
        # # 浣跨敤 from_config 鏂规硶鍒濆鍖?CATSeg 妯″瀷
        # catseg_kwargs = CATSeg.from_config(catseg_cfg)
        # self.CATSeg = CATSeg(**catseg_kwargs)
        # ---------- FeatureMatcher ----------
        # # 浣犲彲鍦?cfg.LOFTR.WEIGHTS 鎸囧畾鏉冮噸璺緞锛涘惁鍒欏洖閫€鍒伴粯璁よ矾寰?
        # matcher_weights = getattr(getattr(cfg, 'LOFTR', {}), 'WEIGHTS', 'MATCHER_CORE/weights/outdoor_ds.ckpt')
        # default_cfg['coarse']['temp_bug_fix'] = False
        # self.matcher = FeatureMatcher(config=default_cfg)
        # state = torch.load(matcher_weights, map_location='cpu')
        # self.matcher.load_state_dict(state['state_dict'])
        # self.matcher = self.matcher.eval()  # FeatureMatcher 鎺ㄧ悊妯″紡

        default_cfg['coarse']['temp_bug_fix'] = False
        self.matcher = FeatureMatcher(config=default_cfg)
        self.matcher.load_state_dict(torch.load("MATCHER_CORE/weights/outdoor_ds.ckpt")['state_dict'])
        self.matcher = self.matcher.eval().cuda()
        # self.matcher = self.matcher.train().cuda()
        # ---------- 鎹熷け ----------
        # ---------- 鎹熷け ----------
        self._mask_loss = DiceLoss(weight=torch.tensor([0.5, 0.5]))
        self.mask_th = 0.5
        self.soft_clip = True

        # --- 鍙€夌殑 feature loss ---
        self.use_feature_loss = False
        if self.use_feature_loss:
            self.feature_loss = FeatureLoss(device='cuda' if torch.cuda.is_available() else 'cpu')

        # ---------- 璁粌鎺у埗 ----------
        self.automatic_optimization = True  # Lightning 鑷姩浼樺寲
        self.multi_gpu = True
        self.validation_step_outputs = []
        self.log_interval = getattr(cfg.TRAINING, 'LOG_INTERVAL', 50)

        # 鍗?epoch 璇勪及鎺у埗
        self._ran_half_eval_for_epoch = False
        self._half_epoch_batch_idx = None  # 姣忎釜 epoch 寮€澶磋绠?

    def forward(self, batch):
        return self.forward_once(batch)

    # -------------------------
    #   = = = 鍏抽敭 Loss = = =
    # -------------------------
    # def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor):
    #     """
    #     pred_logits: [B,1,H_pred,W_pred] 鈥?鎺╃爜 logits
    #     gt:          [B,H_gt,W_gt]       鈥?ground truth binary mask
    #
    #     杩斿洖: loss, pred_mask(0/1), pred_logits, IoU
    #     """
    #     gt_shape = gt.shape[1:]
    #     pred_shape = pred_logits.shape[2:]
    #
    #     gt_c = gt.clone().to(torch.float32)
    #     if gt_shape != pred_shape:
    #         gt_c = F.interpolate(gt.unsqueeze(1), size=pred_shape, mode='nearest').squeeze(1)
    #
    #     if gt_c.max() > 1.0:
    #         gt_c = gt_c / 255.0
    #
    #     logits = pred_logits.squeeze(1)  # [B, H, W]
    #     loss = self._mask_loss(logits, gt_c.to(torch.float32))
    #
    #     with torch.no_grad():
    #         pred_mask = (torch.sigmoid(logits) > self.mask_th).float()
    #         intersection = (pred_mask * gt_c).sum(dim=(1, 2))
    #         union = (pred_mask + gt_c - pred_mask * gt_c).sum(dim=(1, 2)) + 1e-6
    #         iou = (intersection / union).mean()
    #
    #     return loss, pred_mask, logits, iou
    def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        '''
        Prediction will probably be lower then ground truth in resolution.
        Ground truth is downsampled if this happens
        pred: [B,N,H1,W1]
        gt:   [B,H2,W2]
        '''

        gt_shape = gt.shape[1:]
        pred_shape = pred_logits.shape[2:]
        gt_c = gt.clone()
        # print(gt.shape, pred.shape)
        # reduce gt dimension if necessary
        if gt_shape != pred_shape:
            gt_c = F.interpolate(gt.unsqueeze(1), pred_shape, mode='nearest').squeeze(1)

        pred_logits = pred_logits.squeeze(1)
        loss = self._mask_loss(pred_logits, gt_c.to(torch.float))
        with torch.no_grad():
            pred_mask = torch.where(torch.sigmoid(pred_logits) > self.mask_th, 1, 0)
            iou = mask_iou(gt_c, pred_mask)

        return loss, pred_mask, pred_logits, iou.mean()

    def compute_pose_loss(self, R, t, Rgt_i, tgt_i, soft_clipping=True):
        """
        涓巆ompute_pose_loss 涓€鑷达細rot_angle_loss + trans_l1_loss锛堝彲 tanh soft clipping锛?
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

    # ---- 宸ュ叿鍑芥暟 ----
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
    #     FeatureMatcher + 鍚庣姹傚Э鎬?
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
        return: pts3d_full:[N,3] (NaN濉厖), valid_mask:[N]
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
    #         鍓嶅悜閮ㄥ垎
    # -------------------------
    def forward_once(self, batch):
        """
        鍗曟鍓嶅悜锛堢敤浜?train/val/eval锛?
        """
        device = batch['image0'].device
        B, _, H, W = batch['image0'].shape

        #1) Oryon 杈撳嚭锛堝亣瀹氳繑鍥?logits 褰㈡€侊級
        oryon_out = self.oryon_model.forward(batch)  # 鍖呭惈 'mask_a', 'mask_q'
        pred_mask0_logits = oryon_out['mask_a']  # [B,1,Hm,Wm]
        pred_mask1_logits = oryon_out['mask_q']  # [B,1,Hm,Wm]
        # catseg_out = self.oryon_model.forward(batch)  # 鍖呭惈 'mask_a', 'mask_q'
        # pred_mask0_logits = catseg_out['mask_a']  # [B,1,Hm,Wm]
        # pred_mask1_logits = catseg_out['mask_q']  # [B,1,Hm,Wm]

        # 2) mask loss
        mask0_gt = batch['mask0_gt']  # [B,H,W]
        mask1_gt = batch['mask1_gt']  # [B,H,W]
        mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, mask0_gt)
        mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, mask1_gt)
        mask_loss_all = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2.

        # 3) logits -> 姒傜巼
        pred_mask0_prob = torch.sigmoid(pred_mask0_logits).squeeze(1)  # [B,Hm,Wm]
        pred_mask1_prob = torch.sigmoid(pred_mask1_logits).squeeze(1)  # [B,Hm,Wm]

        # 4) resize 鍒拌緭鍏ュ浘鍍忓ぇ灏?
        pred_mask0_prob = F.interpolate(pred_mask0_prob.unsqueeze(1),
                                        size=(H, W), mode='bilinear',
                                        align_corners=False).squeeze(1)
        pred_mask1_prob = F.interpolate(pred_mask1_prob.unsqueeze(1),
                                        size=(H, W), mode='bilinear',
                                        align_corners=False).squeeze(1)

        # 5) 浜屽€煎寲
        pred_mask0_bin = (pred_mask0_prob > 0.5).float()
        pred_mask1_bin = (pred_mask1_prob > 0.5).float()

        # 6) 鐏板害鍥捐繃婊?
        img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
        img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)

        # 7) FeatureMatcher 鍖归厤
        R_preds, t_preds = [], []
        for i in range(B):
            match_batch = {'image0': img0_gray[i:i + 1], 'image1': img1_gray[i:i + 1]}
            with torch.no_grad():
                self.matcher.eval()
                # self.matcher.train()
                self.matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()
            mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()

            m0 = pred_mask0_bin[i].detach().cpu().numpy()
            m1 = pred_mask1_bin[i].detach().cpu().numpy()

            if len(mkpts0) == 0:
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))
                continue

            # 鎸夋帺鐮佽繃婊ゅ叧閿偣
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

            # 8) 鍥炴姇褰卞埌 3D
            depth0 = batch['depth0'][i].detach().cpu().numpy()
            depth1 = batch['depth1'][i].detach().cpu().numpy()
            K0 = batch['K_color0'][i].detach().cpu().numpy()
            K1 = batch['K_color1'][i].detach().cpu().numpy()

            # current_h, current_w = batch['image0'][i].shape[1:]
            # K0 = self.adjust_intrinsics_for_resize(K0, original_size=(640, 480),
            #                                        current_size=(current_w, current_h))
            # K1 = self.adjust_intrinsics_for_resize(K1, original_size=(640, 480),
            #                                        current_size=(current_w, current_h))

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

            # 10) 杞埌缁濆浣嶅Э
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
        # 鍓嶅悜
        out = self.forward_once(batch)

        # GT 缁濆浣嶅Э锛坬uery锛?
        T_q_gt = batch['item_q_pose']
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)

        if self.use_feature_loss:
            # ====== 璋冪敤 feature loss ======
            feat_losses, feat_results = self.feature_loss(batch, out)
            feat_loss = feat_losses['pos'] + feat_losses['neg'] + feat_losses['mask']
            total_loss = feat_loss

            # ---- 鏃ュ織 ----
            self.log('train/feature_pos_loss', feat_losses['pos'], prog_bar=False, on_step=True, on_epoch=True)
            self.log('train/feature_neg_loss', feat_losses['neg'], prog_bar=False, on_step=True, on_epoch=True)
            self.log('train/feature_mask_loss', feat_losses['mask'], prog_bar=True, on_step=True, on_epoch=True)
            self.log('train/feature_total_loss', feat_loss, prog_bar=True, on_step=True, on_epoch=True)

            # 缃浂锛岄槻姝㈡棩蹇楁姤閿?
            pose_loss = torch.tensor(0.0, device=feat_loss.device)
            pose_rot_loss = torch.tensor(0.0, device=feat_loss.device)
            pose_trans_loss = torch.tensor(0.0, device=feat_loss.device)

        else:
            # ====== 璁＄畻 pose loss ======
            pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
                out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
            )
            total_loss = out['mask_loss'] + pose_loss

            # ---- 鏃ュ織 ----
            self.log('train/mask_loss', out['mask_loss'], prog_bar=True, on_step=True, on_epoch=True)
            self.log('train/pose_loss', pose_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train/pose_rot_loss', pose_rot_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log('train/pose_trans_loss', pose_trans_loss, prog_bar=False, on_step=True, on_epoch=True)

        # 鎬?loss 鏃ュ織
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    # def on_train_epoch_start(self):
    #     """鍦ㄦ瘡涓?epoch 寮€澶寸‘瀹氣€滃崐涓?epoch鈥濈殑 batch 绱㈠紩锛屽苟閲嶇疆寮€鍏炽€?""
    #     self._ran_half_eval_for_epoch = False
    #     try:
    #         # 浼拌鏈?epoch 鐨?train batch 鏁?
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
        out = self.forward_once(batch)

        T_q_gt = batch['item_q_pose']
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)

        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )
        total_loss = out['mask_loss'] + pose_loss

        logs = {
            'loss': total_loss.detach(),
            'pose_loss': pose_loss.detach(),
            'pose_rot_loss': pose_rot_loss.detach(),
            'pose_trans_loss': pose_trans_loss.detach(),
            'mask_loss': out['mask_loss'].detach(),
            'mask_iou': out['mask_iou'].detach(),
        }

        # 淇濆瓨鍒板垪琛紝渚?epoch_end 鑱氬悎
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(logs)

        # 鍙互鐩存帴 log batch 绾у埆鐨?loss
        for k, v in logs.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=(k == 'loss'))

        return total_loss

    # -------------------------
    #   Validation Epoch End
    # -------------------------
    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or len(self.validation_step_outputs) == 0:
            return

        # 鑱氬悎鎵€鏈?batch 鐨勬寚鏍?
        agg = {k: torch.stack([x[k] for x in self.validation_step_outputs]).mean()
               for k in self.validation_step_outputs[0].keys()}

        # log epoch-level metrics
        for k, v in agg.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=(k == 'loss'))

        # ADD-0.1D 璇勪及锛堝彧鍦?epoch 鏈畻涓€娆★級
        add_acc = self.run_add01d_eval()
        add_acc = float(add_acc)
        if add_acc is not None:
            self.log('val/add01d_acc', add_acc, prog_bar=True, on_epoch=True, sync_dist=self.multi_gpu)

        self.validation_step_outputs.clear()

    # -------------------------
    #   Optim / Scheduler
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.TRAINING.LR,
            weight_decay=getattr(self.cfg.TRAINING, "WEIGHT_DECAY", 0.0),
            eps=1e-6
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.TRAINING.EPOCHS,
            eta_min=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # -------------------------
    #   ADD(S)-0.1D 璇勪及
    # -------------------------
    @torch.no_grad()
    def run_add01d_eval(self):
        """
         FeatureMatcher 娴嬭瘯璇勪及閫昏緫锛堟牳蹇冭绠楋級锛?
        - 闇€瑕?val_dataloader().dataset 鎻愪緵 dataset.get_obj_info(obj_id)
        - 閫?batch 鐢?forward_once 鑾峰彇 R_pred,t_pred锛堝凡鏄?Query 鐨勭粷瀵瑰Э鎬侊級
        - 璁＄畻鐗╀綋 ADD/ADD-S锛屽苟鐢?0.1D 鍒ゅ畾鎴愬姛
        杩斿洖锛氭垚鍔熺巼锛堢櫨鍒嗘瘮锛夛紝鑻ュけ璐ヨ繑鍥?None
        """
        try:
            dm = self.trainer.datamodule
            vloader = dm.val_dataloader()
        except Exception as e:
            print(f"[WARN] failed to get val_dataloader: {e}")
            return None

        # dataset 蹇呴』瀹炵幇 get_obj_info(obj_id) -> (model, diameter, sym)
        dataset = getattr(vloader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'get_obj_info'):
            print("[WARN] val dataset 鏈彁渚?get_obj_info(obj_id)锛岃烦杩?ADD-0.1D 璇勪及")
            return None

        total, success = 0, 0
        device = self.device

        for batch in vloader:
            # 绉诲姩鍒拌澶?
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 棰勬祴缁濆濮挎€?
            out = self.forward_once(batch)
            R_pred = out['R_pred']  # [B,3,3]
            t_pred = out['t_pred'].squeeze(1)  # [B,3] (m)

            # GT 缁濆濮挎€?
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
                    print(f"[WARN] get_obj_info failed: {e}")
                    continue

                # 鏋勯€?4x4 濮挎€?
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
            print("[WARN] ADD(S)-0.1D 璇勪及娌℃湁鏈夋晥鏍锋湰")
            return None

        acc = 100.0 * success / total
        return acc

