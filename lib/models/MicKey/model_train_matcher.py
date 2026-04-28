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
from MATCHER_CORE.src.losses.matcher_loss import MatcherLoss  # 鏂板锛氬鍏eatureMatcher鎹熷け
from MATCHER_CORE.src.matcher.utils.supervision import compute_supervision_coarse, compute_supervision_fine  # 鏂板锛氬鍏ョ洃鐫ｈ绠?

from lib.utils.metrics import pose_error_torch  # 浠呯敤浜庡彲閫夊榻愭鏌ワ紙鏈敤浜巐oss锛?
from lib.benchmarks.utils import precision_recall  # 鏃ュ織
from Oryon.utils.metrics import compute_add, compute_adds  # 鐢ㄤ簬 ADD/ADD-S
from Oryon.utils.geo6d import best_fit_transform_with_RANSAC  # 鍙€夌殑RANSAC
# from Oryon.utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver  # PointDSC
from Oryon.utils.losses import DiceLoss, LovaszLoss, FocalLoss
from Oryon.utils.metrics import mask_iou

from MATCHER_CORE.src.utils.misc import lower_config, flattenList
from yacs.config import CfgNode as CN
# =========================
#   MicKeyTrainingModel
# =========================
def cfg_to_dict(cfg_node):
    """閫掑綊 yacs CfgNode 杞?dict锛岀粺涓€灏忓啓 key"""
    from yacs.config import CfgNode
    if isinstance(cfg_node, CfgNode):
        return {k.lower(): cfg_to_dict(v) for k, v in cfg_node.items()}
    elif isinstance(cfg_node, list):
        return [cfg_to_dict(x) for x in cfg_node]
    elif isinstance(cfg_node, tuple):
        return tuple(cfg_to_dict(x) for x in cfg_node)
    else:
        return cfg_node

class MicKeyTrainingModel(pl.LightningModule):
    """
    -
    - FeatureMatcher 鍖归厤 + 鎺╃爜杩囨护 + 鍥炴姇褰?鈫?Kabsch 姹傚Э鎬?
    - 鎹熷け锛歮ask_loss锛圔CEWithLogits锛?+ compute_pose_loss锛坮ot_angle + trans_l1锛屽彲閫塼anh clipping锛?
    - 姣忓崐涓?epoch 璺戜竴娆?ADD(S)-0.1D 璇勪及骞惰褰曞埌 TensorBoard
    """


    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg

        # ---------- Oryon ----------
        self.oryon_model = Oryon(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')

        # ---------- FeatureMatcher ----------
        #  YACS config 杞?dict 骞跺皬鍐?key
        _config = cfg_to_dict(cfg)

        # 閰嶇疆
        legacy_matcher_key = ''.join(['lo', 'ftr'])
        self.matcher_cfg = _config.get('matcher', _config[legacy_matcher_key])
        print("self.matcher_cfg:",self.matcher_cfg)
        # 鍒濆鍖?matcher
        #default_cfg['coarse']['temp_bug_fix'] = False
        self.matcher = FeatureMatcher(config=self.matcher_cfg)
        state_dict = torch.load("MATCHER_CORE/weights/outdoor_ds.ckpt")['state_dict']
        self.matcher.load_state_dict(state_dict)
        self.matcher = self.matcher.train().cuda()
        #self.matcher = self.matcher.eval().cuda()
        self.matcher_fine_tune_mode = 'conservative'  # 'conservative', 'minimal', 'none'
        self._setup_matcher_fine_tuning()
        # ---------- FeatureMatcher Loss ----------s
        self.matcher_loss = MatcherLoss(_config).train()
        #self.use_gt_mask=True #filter 2d keypoints
        self.use_gt_mask =True
        # ---------- Mask Loss ----------
        self._mask_loss = DiceLoss(weight=torch.tensor([0.5, 0.5]))
        self.mask_th = 0.5
        self.soft_clip = True

        # ---------- 璁粌 ----------
        self.automatic_optimization = True
        self.multi_gpu = True
        self.validation_step_outputs = []
        self.log_interval = getattr(cfg.TRAINING, 'LOG_INTERVAL', 50)

        # 鍗?epoch 璇勪及
        self._ran_half_eval_for_epoch = False
        self._half_epoch_batch_idx = None

    def _setup_matcher_fine_tuning(self):
        """
        璁剧疆FeatureMatcher鐨勬瀬灏忓井璋冪瓥鐣ワ細
         - 'none': 瀹屽叏鍐荤粨FeatureMatcher
        - 'minimal': 鍙缁冩俯搴﹀弬鏁?
        - 'conservative': 璁粌鏈€鍚?灞倀ransformer + 娓╁害鍙傛暟
        """
        if self.matcher_fine_tune_mode == 'none':
            # 瀹屽叏鍐荤粨FeatureMatcher
            for param in self.matcher.parameters():
                param.requires_grad = False
            print("瀹屽叏鍐荤粨FeatureMatcher鍙傛暟")

        elif self.matcher_fine_tune_mode == 'minimal':
            # 鍙缁冩俯搴﹀弬鏁?
            for param in self.matcher.parameters():
                param.requires_grad = False

            # 鍙缁僣oarse matching鐨勬俯搴﹀弬鏁?
            if hasattr(self.matcher.coarse_matching, 'temperature'):
                self.matcher.coarse_matching.temperature.requires_grad = True
                print("鍙缁僣oarse matching娓╁害鍙傛暟")
            else:
                print("璀﹀憡锛氭壘涓嶅埌娓╁害鍙傛暟锛屽畬鍏ㄥ喕缁揊eatureMatcher")

        elif self.matcher_fine_tune_mode == 'conservative':
            # 鍐荤粨backbone
            for param in self.matcher.backbone.parameters():
                param.requires_grad = False
            print("鍐荤粨FeatureMatcher backbone鍙傛暟")

            # 鍐荤粨浣嶇疆缂栫爜
            for param in self.matcher.pos_encoding.parameters():
                param.requires_grad = False
            print("鍐荤粨FeatureMatcher浣嶇疆缂栫爜鍙傛暟")

            # 鍙缁冩渶鍚?灞俢oarse transformer
            coarse_layers = self.matcher.coarse_transformer.layers
            for i, layer in enumerate(coarse_layers):
                if i < len(coarse_layers) - 1:  # 鍐荤粨鍓嶅嚑灞?
                    for param in layer.parameters():
                        param.requires_grad = False
                else:  # 鍙缁冩渶鍚庝竴灞?
                    print(f"Train coarse transformer layer {i + 1}")

            # 鍐荤粨fine transformer
            for param in self.matcher.fine_transformer.parameters():
                param.requires_grad = False
            print("鍐荤粨FeatureMatcher fine transformer鍙傛暟")

            # 鍐荤粨fine preprocessing
            for param in self.matcher.fine_preprocess.parameters():
                param.requires_grad = False
            print("鍐荤粨FeatureMatcher fine preprocessing鍙傛暟")

            # 鍐荤粨fine matching
            for param in self.matcher.fine_matching.parameters():
                param.requires_grad = False
            print("鍐荤粨FeatureMatcher fine matching鍙傛暟")

            # 璁粌coarse matching鐨勬俯搴﹀弬鏁?
            if hasattr(self.matcher.coarse_matching, 'temperature'):
                print("璁粌coarse matching娓╁害鍙傛暟")
            else:
                for param in self.matcher.coarse_matching.parameters():
                    param.requires_grad = False
                print("鍐荤粨coarse matching鍙傛暟")

        # 缁熻鍙缁冨弬鏁?
        total_params = sum(p.numel() for p in self.matcher.parameters())
        trainable_params = sum(p.numel() for p in self.matcher.parameters() if p.requires_grad)
        print(f"FeatureMatcher鎬诲弬鏁? {total_params / 1e6:.1f}M")
        print(f"FeatureMatcher鍙缁冨弬鏁? {trainable_params / 1e6:.1f}M ({trainable_params / total_params * 100:.1f}%)")

    def forward(self, batch):
        return self.forward_once(batch)

    # -------------------------
    #   = = = Loss = = =
    # -------------------------
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
        rot_angle_loss + trans_l1_loss锛堝彲 tanh soft clipping锛?
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
    #     FeatureMatcher + 姹傚Э鎬?
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
    # def forward_once(self, batch, training=True):
    #     """
    #     鍗曟鍓嶅悜锛堢敤浜?train/val/eval锛?
    #     """
    #     device = batch['image0'].device
    #     B, _, H, W = batch['image0'].shape
    #
    #     # 1) Oryon 杈撳嚭锛堝亣瀹氳繑鍥?logits 褰㈡€侊級
    #     oryon_out = self.oryon_model.forward(batch)  # 鍖呭惈 'mask_a', 'mask_q'
    #     pred_mask0_logits = oryon_out['mask_a']  # [B,1,Hm,Wm]
    #     pred_mask1_logits = oryon_out['mask_q']  # [B,1,Hm,Wm]
    #
    #     # 2) mask loss
    #     mask0_gt = batch['mask0_gt']  # [B,H,W]
    #     mask1_gt = batch['mask1_gt']  # [B,H,W]
    #     mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, mask0_gt)
    #     mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, mask1_gt)
    #     mask_loss_all = mask0_loss + mask1_loss
    #     mask_iou_mean = (mask0_iou + mask1_iou) / 2.
    #
    #     # 3) logits -> 姒傜巼
    #     pred_mask0_prob = torch.sigmoid(pred_mask0_logits).squeeze(1)  # [B,Hm,Wm]
    #     pred_mask1_prob = torch.sigmoid(pred_mask1_logits).squeeze(1)  # [B,Hm,Wm]
    #
    #     # 4) resize 鍒拌緭鍏ュ浘鍍忓ぇ灏?
    #     pred_mask0_prob = F.interpolate(pred_mask0_prob.unsqueeze(1),
    #                                     size=(H, W), mode='bilinear',
    #                                     align_corners=False).squeeze(1)
    #     pred_mask1_prob = F.interpolate(pred_mask1_prob.unsqueeze(1),
    #                                     size=(H, W), mode='bilinear',
    #                                     align_corners=False).squeeze(1)
    #
    #     # 5) 浜屽€煎寲
    #     pred_mask0_bin = (pred_mask0_prob > 0.5).float()
    #     pred_mask1_bin = (pred_mask1_prob > 0.5).float()
    #
    #     # 6) 鐏板害鍥捐繃婊?
    #     img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
    #     img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)
    #
    #     # 7) FeatureMatcher 鍖归厤 - 淇敼锛氱幇鍦ㄥ弬涓庤缁?
    #     R_preds, t_preds = [], []
    #     matcher_losses = []
    #
    #     for i in range(B):
    #         # match_batch = {
    #         #     'image0': img0_gray[i:i + 1],
    #         #     'image1': img1_gray[i:i + 1]
    #         # }
    #
    #         # 鏂板锛氳绠桭eatureMatcher鐩戠潱淇″彿
    #         if training:
    #             # 鍒涘缓鐩戠潱璁＄畻鎵€闇€鐨刡atch缁撴瀯
    #             # 鍋囪 i 鏄?batch 鍐呮牱鏈储寮?
    #             # 1) 鑾峰彇 coarse feature map 灏哄
    #             B, _, H, W = batch['image0'].shape
    #             #feat_H, feat_W = pred_mask0_logits.shape[2:]  # [B,1,Hf,Wf]
    #
    #             # 2) 涓嬮噰鏍?mask 鍒?coarse 灏哄
    #             mask0 = F.interpolate(batch['mask0_gt'].float().unsqueeze(1), size=(60, 80),# h #w
    #                                   mode='nearest').squeeze(1)
    #             mask1 = F.interpolate(batch['mask1_gt'].float().unsqueeze(1), size=(60,80),
    #                                   mode='nearest').squeeze(1)
    #
    #             match_batch = {
    #                 # 'image0': batch['image0'][i:i + 1],  # [1,3,H,W]
    #                 # 'image1': batch['image1'][i:i + 1],
    #                 'image0': img0_gray[i:i + 1],
    #                 'image1': img1_gray[i:i + 1],
    #                 'mask0': mask0[i:i + 1],  #
    #                 'mask1': mask1[i:i + 1],
    #                 'depth0': batch['depth0'][i:i + 1],  # [1,H,W]
    #                 'depth1': batch['depth1'][i:i + 1],
    #                 'K0': batch['K_color0'][i:i + 1],  # [1,3,3]
    #                 'K1': batch['K_color1'][i:i + 1],
    #                 'T_0to1': batch['pose'][i:i + 1],  # [1,4,4]锛屼綘鍘?batch_dict 涓?'pose' 鏄?item_a -> item_q 鐩稿浣嶅Э
    #                 'T_1to0': torch.inverse(batch['pose'][i:i + 1]),  # 濡傛灉闇€瑕侀€?
    #                 'scale0': torch.tensor([[1.0, 1.0]], device=batch['image0'].device),
    #                 'scale1': torch.tensor([[1.0, 1.0]], device=batch['image0'].device),
    #                 #'dataset_name': 'Shapenet6D',  # 鎴栦綘瀹為檯鏁版嵁闆嗗悕
    #                 'scene_id': batch['instance_id_a'][i],  # 娌℃湁 scene_id 灏辩敤 instance_id 鎴?0
    #                 'pair_id': i,
    #                 'pair_names': (batch['instance_id_a'][i], batch['instance_id_q'][i])  # 濡傛灉娌℃湁璺緞锛屽彲浠ユ殏鏃剁敤 instance_id
    #             }
    #
    #             # 璁＄畻绮楃洃鐫?
    #             compute_supervision_coarse(match_batch, self.cfg)
    #
    #         # FeatureMatcher鍓嶅悜浼犳挱
    #         self.matcher(match_batch)
    #
    #         # 鏂板锛氳绠桭eatureMatcher鎹熷け
    #         if training:
    #             # 璁＄畻绮剧粏鐩戠潱
    #             compute_supervision_fine(match_batch, self.cfg)
    #             # 璁＄畻FeatureMatcher鎹熷け
    #             matcher_loss = self.matcher_loss(match_batch)
    #             matcher_losses.append(matcher_loss)
    #
    #         mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()
    #         mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()
    #
    #         m0 = pred_mask0_bin[i].detach().cpu().numpy()
    #         m1 = pred_mask1_bin[i].detach().cpu().numpy()
    #
    #         if len(mkpts0) == 0:
    #             R_preds.append(torch.eye(3, device=device))
    #             t_preds.append(torch.zeros(1, 3, device=device))
    #             continue
    #
    #         # 鎸夋帺鐮佽繃婊ゅ叧閿偣
    #         in_mask = (m0[mkpts0[:, 1].round().astype(int), mkpts0[:, 0].round().astype(int)] > 0) & \
    #                   (m1[mkpts1[:, 1].round().astype(int), mkpts1[:, 0].round().astype(int)] > 0)
    #         mkpts0 = mkpts0[in_mask]
    #         mkpts1 = mkpts1[in_mask]
    #
    #         if len(mkpts0) < 3:
    #             R_preds.append(torch.eye(3, device=device))
    #             t_preds.append(torch.zeros(1, 3, device=device))
    #             continue
    #
    #         # 8) 鍥炴姇褰卞埌 3D
    #         depth0 = batch['depth0'][i].detach().cpu().numpy()
    #         depth1 = batch['depth1'][i].detach().cpu().numpy()
    #         K0 = batch['K_color0'][i].detach().cpu().numpy()
    #         K1 = batch['K_color1'][i].detach().cpu().numpy()
    #
    #         pts3d_0, valid0 = self.backproject(mkpts0, depth0, K0)
    #         pts3d_1, valid1 = self.backproject(mkpts1, depth1, K1)
    #         valid = valid0 & valid1
    #         if np.count_nonzero(valid) < 3:
    #             R_preds.append(torch.eye(3, device=device))
    #             t_preds.append(torch.zeros(1, 3, device=device))
    #             continue
    #
    #         A = pts3d_0[valid]
    #         Bp = pts3d_1[valid]
    #
    #         # 9) Kabsch
    #         R_np, t_np = self.kabsch_umeyama(A, Bp)
    #         R_t = torch.from_numpy(R_np).float().to(device)
    #         t_t = torch.from_numpy(t_np).float().to(device).unsqueeze(0) / 1000.0
    #
    #         # 10) 杞埌缁濆浣嶅Э
    #         T_a = batch['item_a_pose'][i].detach().cpu().numpy()
    #         T_rel = np.eye(4, dtype=np.float32)
    #         T_rel[:3, :3] = R_np
    #         T_rel[:3, 3] = (t_np / 1000.0)
    #         T_q_pred = T_rel @ T_a
    #         R_q = torch.from_numpy(T_q_pred[:3, :3]).float().to(device)
    #         t_q = torch.from_numpy(T_q_pred[:3, 3]).float().to(device).unsqueeze(0)
    #
    #         R_preds.append(R_q)
    #         t_preds.append(t_q)
    #
    #         R_pred = torch.stack(R_preds, dim=0)
    #         t_pred = torch.stack(t_preds, dim=0)
    #
    #         # 璁＄畻骞冲潎FeatureMatcher鎹熷け
    #         matcher_loss_mean = torch.stack(matcher_losses).mean() if matcher_losses else torch.tensor(0.0, device=device)
    #
    #     return {
    #         'R_pred': R_pred,
    #         't_pred': t_pred,
    #         'mask_loss': mask_loss_all,
    #         'mask_iou': mask_iou_mean,
    #         'mask0_loss': mask0_loss,
    #         'mask1_loss': mask1_loss,
    #         'matcher_loss': matcher_loss_mean  # 鏂板锛欶eatureMatcher鎹熷け
    #     }
    def forward_once(self, batch, training=True):
        """
        鍓嶅悜锛孎eatureMatcher + mask + 3D + Batch Kabsch
        keypoints padding + mask杩囨护
        """
        #print("self.training:", self.training, "training param:", training)

        device = batch['image0'].device
        B, _, H, W = batch['image0'].shape

        # Oryon 杈撳嚭锛坙ogits锛?
        oryon_out = self.oryon_model.forward(batch)
        pred_mask0_logits = oryon_out['mask_a']  # [B,1,Hf,Wf]
        pred_mask1_logits = oryon_out['mask_q']

        # mask loss
        mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, batch['mask0_gt'])
        mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, batch['mask1_gt'])
        mask_loss_all = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2.

        # mask姒傜巼 & resize
        pred_mask0_prob = F.interpolate(torch.sigmoid(pred_mask0_logits), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        pred_mask1_prob = F.interpolate(torch.sigmoid(pred_mask1_logits), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        pred_mask0_bin = (pred_mask0_prob > self.mask_th).float()
        pred_mask1_bin = (pred_mask1_prob > self.mask_th).float()

        # 鐏板害鍥?& mask
        img0_gray = self.rgb_to_gray(batch['image0']) #* pred_mask0_bin.unsqueeze(1)
        img1_gray = self.rgb_to_gray(batch['image1']) #* pred_mask1_bin.unsqueeze(1)

        # GT mask锛岃浆鎴?float 骞跺姞涓?channel 缁村害浠ヤ究骞挎挱
        mask0_gt = batch['mask0_gt'].unsqueeze(1).float()  # [B,1,H,W]
        mask1_gt = batch['mask1_gt'].unsqueeze(1).float()
        #print("mask0_gt",mask0_gt)
        # 鐢?GT mask 杩囨护鐏板害鍥撅紝闈炵洰鏍囧尯鍩熺疆 0
        img0_gray_masked = img0_gray * mask0_gt
        img1_gray_masked = img1_gray * mask1_gt
        item_a_pose = batch['item_a_pose']  # 鐗╀綋鍦╝nchor涓嬬殑浣嶅Э
        item_q_pose = batch['item_q_pose']  # 鐗╀綋鍦╭uery涓嬬殑浣嶅Э

        # 鐩告満闂寸浉瀵逛綅濮?(0->1 琛ㄧず anchor鐩告満 鈫?query鐩告満)
        T_0to1 = item_q_pose @ torch.inverse(item_a_pose)
        T_1to0 = item_a_pose @ torch.inverse(item_q_pose)
        # FeatureMatcher 杈撳叆 batch
        match_batch = {
             #'image0': img0_gray,
             #'image1': img1_gray,
            'image0': img0_gray_masked,
            'image1': img1_gray_masked,
            'depth0': batch['depth0'],
            'depth1': batch['depth1'],
            'K0': batch['K_color0'],
            'K1': batch['K_color1'],
            # 'T_0to1': batch['pose'],
            # 'T_1to0': torch.inverse(batch['pose']),
            'T_0to1': T_0to1,
            'T_1to0': T_1to0,
            'scale0': torch.ones(B, 2, device=device),
            'scale1': torch.ones(B, 2, device=device),
            'pair_names': batch['instance_id']
        }

        if training:
            # coarse supervision
            compute_supervision_coarse(match_batch, self.cfg)
            # FeatureMatcher matcher
            #with torch.no_grad():
                #self.matcher.eval().cuda()
            self.matcher(match_batch)
            # fine supervision
            compute_supervision_fine(match_batch, self.cfg)
            # FeatureMatcher loss
            #matcher_loss = self.matcher_loss(match_batch)
            matcher_loss_out = self.matcher_loss(match_batch)  #  dict锛屽寘鍚?loss / loss_scalars
            matcher_loss = matcher_loss_out['loss']
        else:
            with torch.no_grad():
                self.matcher(match_batch)
            # 楠岃瘉鏃朵笉璁＄畻FeatureMatcher loss
            matcher_loss = torch.tensor(0., device=device)

            # keypoints padding + mask杩囨护 瀹屽叏 vectorized
        mkpts0_f = match_batch['mkpts0_f']  # [M, 2]
        mkpts1_f = match_batch['mkpts1_f']  # [M, 2]
        m_bids = match_batch['m_bids']      # [M]

        # 鎸夋壒娆″垎缁勫叧閿偣
        mkpts0_list, mkpts1_list = [], []
        for b in range(B):
            mask = (m_bids == b)
            mkpts0_list.append(mkpts0_f[mask].cpu().numpy())
            mkpts1_list.append(mkpts1_f[mask].cpu().numpy())

        max_pts = max([len(k) for k in mkpts0_list]) if mkpts0_list else 0
        print("length_pts:", [len(k) for k in mkpts0_list])

        if max_pts == 0:
            print(" max_pts=0,batch got no keypoints")
            #exit()

        pts0_batch = torch.zeros(B, max_pts, 2, device=device)
        pts1_batch = torch.zeros(B, max_pts, 2, device=device)

        lengths = torch.tensor([len(k) for k in mkpts0_list], device=device)
        for b in range(B):
            if lengths[b] > 0:
                pts0_batch[b, :lengths[b]] = torch.tensor(mkpts0_list[b], device=device)
                pts1_batch[b, :lengths[b]] = torch.tensor(mkpts1_list[b], device=device)

        # keypoints mask杩囨护
        x0 = pts0_batch[..., 0].long()
        y0 = pts0_batch[..., 1].long()
        x1 = pts1_batch[..., 0].long()
        y1 = pts1_batch[..., 1].long()




        # 閫夋嫨浣跨敤鐨勬帺鐮?
        if self.use_gt_mask:  #
            # GT 鎺╃爜flat
            gt_mask0_flat = batch['mask0_gt'].view(B, -1).float()
            gt_mask1_flat = batch['mask1_gt'].view(B, -1).float()
            mask0_flat = gt_mask0_flat
            mask1_flat = gt_mask1_flat
        else:
            pred_mask0_flat = pred_mask0_bin.view(B, -1)
            pred_mask1_flat = pred_mask1_bin.view(B, -1)
            mask0_flat = pred_mask0_flat
            mask1_flat = pred_mask1_flat

        idx0 = y0 * W + x0  # 涓€缁寸储寮?
        idx1 = y1 * W + x1

        m0_vals = torch.gather(mask0_flat, 1, idx0)
        m1_vals = torch.gather(mask1_flat, 1, idx1)
        valid_mask_batch = (m0_vals > 0) & (m1_vals > 0)

        arange_pts = torch.arange(max_pts, device=device)[None, :].expand(B, -1)
        valid_mask_batch = valid_mask_batch & (arange_pts < lengths[:, None])

        # 鍥炴姇褰卞埌3D 瀹屽叏 batch
        depth0 = batch['depth0'].unsqueeze(1)
        depth1 = batch['depth1'].unsqueeze(1)
        K0_inv = torch.linalg.inv(batch['K_color0'])
        K1_inv = torch.linalg.inv(batch['K_color1'])

        z0 = depth0[torch.arange(B)[:, None], 0, y0, x0]
        z1 = depth1[torch.arange(B)[:, None], 0, y1, x1]
        # 娣卞害鏈夋晥鎬ц繃婊わ紙z>0锛?
        valid_mask_batch = valid_mask_batch & (z0 > 0) & (z1 > 0)

        pts3d0 = torch.stack([x0 * z0, y0 * z0, z0], dim=-1) @ K0_inv.transpose(1, 2)
        pts3d1 = torch.stack([x1 * z1, y1 * z1, z1], dim=-1) @ K1_inv.transpose(1, 2)

        pts3d0 = pts3d0 * valid_mask_batch.unsqueeze(-1)
        pts3d1 = pts3d1 * valid_mask_batch.unsqueeze(-1)

        num_valid = valid_mask_batch.sum(dim=1).unsqueeze(-1).clamp(min=1)  # [B, 1]

        # batch Kabsch
        centroid0 = pts3d0.sum(1) / num_valid  # [B, 3]
        centroid1 = pts3d1.sum(1) / num_valid  # [B, 3]
        A_centered = (pts3d0 - centroid0.unsqueeze(1)) * valid_mask_batch.unsqueeze(-1)
        B_centered = (pts3d1 - centroid1.unsqueeze(1)) * valid_mask_batch.unsqueeze(-1)

        H = torch.einsum('bni,bnj->bij', A_centered, B_centered)  # [B,3,3]
        U, S, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-2, -1)
        R = V @ U.transpose(-2, -1)
        det = torch.linalg.det(R)
        mask_neg = det < 0
        if mask_neg.any():
            V[mask_neg, -1, :] *= -1
            R[mask_neg] = V[mask_neg] @ U[mask_neg].transpose(-2, -1)
        t = centroid1 - torch.einsum('bij,bj->bi', R, centroid0)

        # 鏈夋晥鐐规暟 < 3 鐨勬牱鏈洖閫€鍒板崟浣嶅Э鎬佷笌闆跺钩绉?
        has_valid = (valid_mask_batch.sum(dim=1) >= 3)
        if (~has_valid).any():
            R_fallback = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
            t_fallback = torch.zeros(B, 3, device=device)
            R = torch.where(has_valid.view(B, 1, 1), R, R_fallback)
            t = torch.where(has_valid.view(B, 1), t, t_fallback)

        # 杞崲鐩稿浣嶅Э涓虹粷瀵逛綅濮?
        T_rel = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_rel[:, :3, :3] = R
        T_rel[:, :3, 3] = t / 1000.0  # 杞崲涓虹背

        T_a = batch['item_a_pose']  # [B, 4, 4]
        T_q_pred = T_rel @ T_a

        R_pred = T_q_pred[:, :3, :3]
        t_pred = T_q_pred[:, :3, 3].unsqueeze(1)  # [B, 1, 3]

         # === 淇锛氭棤鏁堢偣澶勭悊 ===
        # for b in range(B):
        #     if num_valid[b] < 3:
        #         R_pred[b] = torch.eye(3, device=device)
        #         t_pred[b] = torch.zeros(1, 3, device=device)
        # print("R_pred",R_pred)
        # print("t_pred",t_pred)

        return {
            'R_pred': R_pred,
            't_pred': t_pred,
            'mask_loss': mask_loss_all,
            'mask_iou': mask_iou_mean,
            'mask0_loss': mask0_loss,
            'mask1_loss': mask1_loss,
            'matcher_loss': matcher_loss
        }



    # -------------------------
    #     Lightning Hooks
    # -------------------------
    def training_step(self, batch, batch_idx):
        # 鍏煎 collate
        # if 'pose' in batch and 'T_0to1' not in batch:
        #     batch['T_0to1'] = batch['pose']

        # 鍓嶅悜 + 寰楀埌棰勬祴浣嶅Э
        out = self.forward_once(batch, training=True)

        # GT 缁濆浣嶅Э锛坬uery锛?
        T_q_gt = batch['item_q_pose']  # [B,4,4]
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)  # [B,1,3]锛屽崟浣?m

        # 濮挎€佹崯澶?
        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )

        # print("mask_loss:", out['mask_loss'])
        # print("pose_loss:", pose_loss)
        # print("matcher_loss:", out['matcher_loss'])

        # totalloss
        total_loss =out['mask_loss'] + pose_loss + 0*out['matcher_loss']

        # ---- 鏃ュ織 ----
        self.log('train/mask_loss', out['mask_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mask_iou', out['mask_iou'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_loss', pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/pose_rot_loss', pose_rot_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_trans_loss', pose_trans_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/matcher_loss', out['matcher_loss'], prog_bar=True, on_step=True, on_epoch=True)  # 鏂板
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    # -------------------------
    #   Validation Step
    # -------------------------
    def validation_step(self, batch, batch_idx):
        # 楠岃瘉鏃朵笉璁＄畻FeatureMatcher鎹熷け
        out = self.forward_once(batch, training=False)

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

        # epoch_end 鑱氬悎
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(logs)

        #  batch level  loss
        for k, v in logs.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=(k == 'loss'))

        return total_loss

    # -------------------------
    #   Validation Epoch End
    # -------------------------
    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or len(self.validation_step_outputs) == 0:
            return

        # batch 鐨勬寚鏍?
        agg = {k: torch.stack([x[k] for x in self.validation_step_outputs]).mean()
               for k in self.validation_step_outputs[0].keys()}

        # log epoch-level metrics
        for k, v in agg.items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=(k == 'loss'))

        # ADD-0.1D 璇勪及epoch 鏈?
        add_acc = self.run_add01d_eval()
        add_acc = float(add_acc)
        if add_acc is not None:
            self.log('val/add01d_acc', add_acc, prog_bar=True, on_epoch=True, sync_dist=self.multi_gpu)

        self.validation_step_outputs.clear()

    # -------------------------
    #   Optim / Scheduler
    # -------------------------
    def configure_optimizers(self):
        #
        optimizer = torch.optim.Adam(
            self.parameters(),  # FeatureMatcher鐨勫弬鏁?
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
        -  val_dataloader().dataset 鎻愪緵 dataset.get_obj_info(obj_id)
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
            out = self.forward_once(batch, training=False)
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


