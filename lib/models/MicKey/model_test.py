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
from Oryon.bop_toolkit_lib.pose_error import my_mssd, my_mspd, vsd
from Oryon.bop_toolkit_lib.renderer_vispy import RendererVispy
from Oryon.utils.pcd import get_diameter

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
from Oryon.bop_toolkit_lib.misc import format_sym_set
import numpy as np
from omegaconf import OmegaConf

# ==== 澶栭儴渚濊禆 ====
from lib.models.Oryon.oryon import Oryon

# FeatureMatcher
from MATCHER_CORE.src.matcher import FeatureMatcher, default_cfg

# 浣犲伐绋嬪唴宸叉湁鐨勫伐鍏?
from lib.utils.metrics import pose_error_torch  # 浠呯敤浜庡彲閫夊榻愭鏌ワ紙鏈敤浜巐oss锛?
from lib.benchmarks.utils import precision_recall  # 鏃ュ織闇€瑕佺殑璇濆彲浠ョ户缁敤
from Oryon.utils.metrics import compute_add, compute_adds  # 鐢ㄤ簬 ADD/ADD-S
from Oryon.utils.geo6d import best_fit_transform_with_RANSAC  # 鍙€夌殑RANSAC
from Oryon.utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver  # 濡傞渶PointDSC灏辨墦寮€
from Oryon.utils.losses import DiceLoss, LovaszLoss, FocalLoss

#from lib.models.MicKey.debug_matcher import debug_matcher
# =========================
#   MicKeyTrainingModel
# =========================
def sharpen_binary_mask(mask_bin, alpha=1.0):
    """
    mask_bin: [B,H,W] float 0/1 浜屽€煎浘
    alpha: 杈圭紭鎵╁紶寮哄害锛?.5~2.0锛?
    """
    B, H, W = mask_bin.shape[0], mask_bin.shape[1], mask_bin.shape[2]

    # 1) Sobel 鎴?Laplacian 鑾峰彇浜屽€?mask 鐨勮竟缂?
    lap_kernel = torch.tensor([[0, 1, 0],
                               [1,-4, 1],
                               [0, 1, 0]], dtype=torch.float32,
                               device=mask_bin.device).unsqueeze(0).unsqueeze(0)

    edge = F.conv2d(mask_bin.unsqueeze(1), lap_kernel, padding=1).abs()  # [B,1,H,W]
    edge = (edge > 0.1).float()  # 浜屽€煎寲杈圭紭

    # 2) 鎵╁紶杈圭紭锛堝寮洪攼鍖栨晥鏋滐級
    dilate_kernel = torch.ones((1,1,3,3), device=mask_bin.device)
    edge_dilate = F.conv2d(edge, dilate_kernel, padding=1)
    edge_dilate = (edge_dilate > 0).float()

    # 3) 灏嗚竟缂樺姞鍥?mask锛堥攼鍖栵級
    sharpened = mask_bin.unsqueeze(1) + alpha * edge_dilate
    sharpened = (sharpened > 0.5).float()  # 鍐嶆浜屽€煎寲淇濇寔骞插噣

    return sharpened.squeeze(1)

class MicKeyTrainingModel(pl.LightningModule):
    """
    绮剧畝鍚庣殑璁粌妯″瀷锛?
    - 浠呬繚鐣?Oryon 浜х敓 mask 鐨勬ā鍧?
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

        # ---------- 鎹熷け ----------
        #self._mask_loss = nn.BCEWithLogitsLoss(reduction='mean')  # 鐢?logits
        self._mask_loss =DiceLoss(weight=torch.tensor([0.5, 0.5]))
        self.mask_th = getattr(getattr(cfg, 'LOSS', {}), 'MASK_TH', 0.5)
        self.soft_clip = getattr(getattr(cfg, 'LOSS', {}), 'SOFT_CLIP', True)

        # ---------- 璁粌鎺у埗 ----------
        self.automatic_optimization = True  #  Lightning 鑷姩浼樺寲
        self.multi_gpu = True
        self.validation_step_outputs = []
        self.log_interval = getattr(cfg.TRAINING, 'LOG_INTERVAL', 50)

        # 鍗?epoch 璇勪及鎺у埗
        self._ran_half_eval_for_epoch = False
        self._half_epoch_batch_idx = None  # 姣忎釜 epoch 寮€澶磋绠?

        # VSD娓叉煋鍣?
        self.compute_vsd=True
        self.vsd_renderer = RendererVispy(640, 480, mode='depth')
        self.vsd_taus = list(np.arange(0.05, 0.51, 0.05))
        self.vsd_rec = np.arange(0.05, 0.51, 0.05)
        self.vsd_delta = 15.
        self._vsd_objects_loaded = False

        # 鍒濆鍖栨寚鏍囪褰?
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.useGTmask=False

        self.debug_matcher_flag =False # 璋冭瘯寮€鍏?

        self.FirstMask=False
        self.SecondMask=False
        self.top_k_matches=0  #     =8 for lm and ycbv

        if(self.top_k_matches==0):
            print("all valid points to 3d")
        else:
            print("top conf_valid points to 3d num:",self.top_k_matches)


    def _load_vsd_objects(self):
        if self._vsd_objects_loaded:
            return

        # 浠?datamodule 鑾峰彇鎵€鏈?obj 淇℃伅
        obj_models, obj_diams, obj_symms = self.trainer.datamodule.val_dataloader().dataset.get_object_info()
        self.add_object_info(obj_models, obj_diams, obj_symms)
        self._vsd_objects_loaded = True

        print("Loaded VSD objects:", list(self.vsd_renderer.model_bbox_corners.keys()))

    def forward(self, batch):
        return self.forward_once(batch)

    # -------------------------
    #   = = = 鍏抽敭 Loss = = =
    # -------------------------
    def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor):
        """
        pred_logits: [B,1,H_pred,W_pred] 鈥?鎺╃爜 logits
        gt:          [B,H_gt,W_gt]       鈥?ground truth binary mask

        杩斿洖: loss, pred_mask(0/1), pred_logits, IoU
        """
        # print("(pred.shape:",pred_logits.shape)
        # print("(gt.shape:",gt.shape)
        # if pred_logits.dim() == 3:
        #     pred =pred_logits.unsqueeze(1)  # [B, 1, H, W]
        # if gt.dim() == 3:
        #     gt = gt.unsqueeze(1)  # [B, 1, H, W]

        #pred_logits.shape)
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
        涓庝綘缁欑殑 compute_pose_loss 涓€鑷达細rot_angle_loss + trans_l1_loss锛堝彲 tanh soft clipping锛?
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

        if self.debug_matcher_flag:
            self.debug_matcher(batch)
            print("debug finished...")
            exit()

        """
        鍗曟鍓嶅悜锛堢敤浜?train/val/eval锛?
        """
        device = batch['image0'].device
        B, _, H, W = batch['image0'].shape

        # 1) Oryon 杈撳嚭锛堝亣瀹氳繑鍥?logits 褰㈡€侊級
        oryon_out = self.oryon_model.forward(batch)  # 鍖呭惈 'mask_a', 'mask_q'
        pred_mask0_logits = oryon_out['mask_a']  # [B,1,Hm,Wm]
        pred_mask1_logits = oryon_out['mask_q']  # [B,1,Hm,Wm]

        # 2) mask loss
        mask0_gt = batch['mask0_gt']  # [B,H,W]
        mask1_gt = batch['mask1_gt']  # [B,H,W]

        # 2) mask loss
        mask0_gt = batch['mask0_gt']  # [B,H,W]
        mask1_gt = batch['mask1_gt']  # [B,H,W]

        # # -------------------------------
        # # 鉁?璋冭瘯淇濆瓨锛氬墠3缁勬牱鏈埌 debug 鐩綍
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
        # # 鍙繚瀛樺墠涓夌粍锛岄槻姝㈠お澶?
        # num_save = min(3, B)
        #
        # # 鎶婇娴?logits 閫氳繃 sigmoid 鍙樻垚 0~1 姒傜巼
        # pred_mask0 = torch.sigmoid(pred_mask0_logits)
        # pred_mask1 = torch.sigmoid(pred_mask1_logits)
        #
        # # 濡傛灉鍒嗚鲸鐜囦笌GT涓嶅悓锛屽彲鍏堟彃鍊?
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
        #     # 杞垚 [0,255] uint8 鎺╃爜
        #     gt0_img = Image.fromarray((gt0 * 255).astype(np.uint8))
        #     gt1_img = Image.fromarray((gt1 * 255).astype(np.uint8))
        #     pm0_img = Image.fromarray((pm0 * 255).astype(np.uint8))
        #     pm1_img = Image.fromarray((pm1 * 255).astype(np.uint8))
        #
        #     # 淇濆瓨鍘熷浘鍜屾帺鐮?
        #     vutils.save_image(img0, f"{save_dir}/image0_{i}.png", normalize=True)
        #     vutils.save_image(img1, f"{save_dir}/image1_{i}.png", normalize=True)
        #     gt0_img.save(f"{save_dir}/mask0_gt_{i}.png")
        #     gt1_img.save(f"{save_dir}/mask1_gt_{i}.png")
        #     pm0_img.save(f"{save_dir}/mask0_pred_{i}.png")
        #     pm1_img.save(f"{save_dir}/mask1_pred_{i}.png")
        #
        #     # 猸?鐢?GT 鎺╃爜杩囨护鍘熷浘锛堢墿浣撳尯鍩熶繚鐣欙紝鍏朵綑鍏ㄩ粦锛?
        #     mask0_tensor = torch.from_numpy(gt0).float().unsqueeze(0)  # [1,H,W]
        #     mask1_tensor = torch.from_numpy(gt1).float().unsqueeze(0)
        #     filtered0_gt = img0 * mask0_tensor
        #     filtered1_gt = img1 * mask1_tensor
        #     vutils.save_image(filtered0_gt, f"{save_dir}/image0_filtered_gt_{i}.png", normalize=True)
        #     vutils.save_image(filtered1_gt, f"{save_dir}/image1_filtered_gt_{i}.png", normalize=True)
        #
        #     # # 猸?鐢ㄩ娴嬫帺鐮佽繃婊ゅ師鍥撅紙鏌ョ湅棰勬祴鍖哄煙鏁堟灉锛?
        #     # mask0_pred_tensor = torch.from_numpy(pm0).float().unsqueeze(0)
        #     # mask1_pred_tensor = torch.from_numpy(pm1).float().unsqueeze(0)
        #     # filtered0_pred = img0 * mask0_pred_tensor
        #     # filtered1_pred = img1 * mask1_pred_tensor
        #     # vutils.save_image(filtered0_pred, f"{save_dir}/image0_filtered_pred_{i}.png", normalize=True)
        #     # vutils.save_image(filtered1_pred, f"{save_dir}/image1_filtered_pred_{i}.png", normalize=True)
        #
        # print(f"宸蹭繚瀛?{num_save} 缁勫浘鍍忎笌鎺╃爜鍒?{save_dir}/")

        if(self.useGTmask):
            mask0_gt = mask0_gt.unsqueeze(1)  # [B,1,H,W]
            mask1_gt = mask1_gt.unsqueeze(1)  # [B,1,H,W]
            mask0_loss, _, _, mask0_iou = self.mask_loss(mask0_gt, batch['mask0_gt'])
            mask1_loss, _, _, mask1_iou = self.mask_loss(mask1_gt, batch['mask1_gt'])
        else:
            mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, mask0_gt)
            mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, mask1_gt)
        mask_loss_all = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2.

        # 3) logits -> 姒傜巼
        pred_mask0_prob = torch.sigmoid(pred_mask0_logits).squeeze(1)  # [B,Hm,Wm]
        pred_mask1_prob = torch.sigmoid(pred_mask1_logits).squeeze(1)  # [B,Hm,Wm]

        # # ----馃敟鍩轰簬鎷夋櫘鎷夋柉绠楀瓙鐨勮竟缂橀攼鍖?----
        # sharpen = LaplacianSharpen(alpha=1.0).to(pred_mask0_prob.device)
        # pred_mask0_prob = sharpen(pred_mask0_prob.unsqueeze(1)).squeeze(1)
        # pred_mask1_prob = sharpen(pred_mask1_prob.unsqueeze(1)).squeeze(1)

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
        #pred_mask0_bin = (pred_mask0_prob > 0.3).float()
        #pred_mask1_bin = (pred_mask1_prob > 0.3).float()

        # #Laplacian 妫€娴嬭竟缂?+ 鑶ㄨ儉锛屽緱鍒伴攼鍖栬竟鐣?
        # pred_mask0_bin = sharpen_binary_mask(pred_mask0_bin)
        # pred_mask1_bin = sharpen_binary_mask(pred_mask1_bin)

        # 6) pred鐏板害鍥捐繃婊?
        #img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
        #img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)

        if(self.useGTmask):

            ## 6) 鐏板害鍥捐繃婊?- GT 鎺╃爜
            img0_gray = self.rgb_to_gray(batch['image0']) * batch['mask0_gt'].unsqueeze(1)
            img1_gray = self.rgb_to_gray(batch['image1']) * batch['mask1_gt'].unsqueeze(1)
        else:
            # pred鐏板害鍥捐繃婊?1st
            if(self.FirstMask):
                img0_gray = self.rgb_to_gray(batch['image0']) * pred_mask0_bin.unsqueeze(1)
                img1_gray = self.rgb_to_gray(batch['image1']) * pred_mask1_bin.unsqueeze(1)
            #no 1st
            else:
                img0_gray = self.rgb_to_gray(batch['image0'])
                img1_gray = self.rgb_to_gray(batch['image1'])

        # 7) FeatureMatcher 鍖归厤
        R_preds, t_preds = [], []
        for i in range(B):
            match_batch = {'image0': img0_gray[i:i + 1], 'image1': img1_gray[i:i + 1]}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()  # [N, 2]
            mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()  # [N, 2]
            mconf = match_batch['mconf'].detach().cpu().numpy()  # [N]
            #print("len(mkpts0 after matcher", len(mkpts0))

            if (self.useGTmask):
                # if浣跨敤 GT 鎺╃爜
                m0 = batch['mask0_gt'][i].detach().cpu().numpy()  # 鈽?
                m1 = batch['mask1_gt'][i].detach().cpu().numpy()  # 鈽?
            else:
                #if pred mask
                m0 = pred_mask0_bin[i].detach().cpu().numpy()
                m1 = pred_mask1_bin[i].detach().cpu().numpy()


            if len(mkpts0) == 0:
                print("error,len(mkpts0)<0 after matcher")
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))
                continue

            # # 鎸夋帺鐮佽繃婊ゅ叧閿偣
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

            ### 鉁?鎸夋帺鐮佽繃婊ゅ叧閿偣 2st filter
            if (self.SecondMask):
                in_mask = (m0[mkpts0[:, 1].round().astype(int),
                mkpts0[:, 0].round().astype(int)] > 0) & \
                          (m1[mkpts1[:, 1].round().astype(int),
                          mkpts1[:, 0].round().astype(int)] > 0)

                mkpts0 = mkpts0[in_mask]
                mkpts1 = mkpts1[in_mask]
                mconf = mconf[in_mask]  # 鈽?鍚屾杩囨护缃俊搴?
            ###



            #print("len(mkpts0 after mask2d:", len(mkpts0))
            # # 鉁?鎸夌疆淇″害鎺掑簭骞跺彇 Top-8
            # if len(mkpts0) >= 8:
            #     idx = np.argsort(-mconf)[:8]  # 浠庡ぇ鍒板皬
            #     mkpts0 = mkpts0[idx]
            #     mkpts1 = mkpts1[idx]
            #     mconf = mconf[idx]  # 淇濇寔涓€鑷?
            # elif len(mkpts0) < 3:  # 浠嶄笉澶熺敤浜嶬absch
            #     print("error,len(mkpts0)<3 after 2nd filtering")
            #     R_preds.append(torch.eye(3, device=device))
            #     t_preds.append(torch.zeros(1, 3, device=device))
            #     continue

            # 鏍规嵁 self.top_k_matches 鎺у埗鏄惁鍙?Top-K 缃俊搴﹀尮閰嶇偣
            top_k =self.top_k_matches

            if top_k is not None and top_k > 0:
                # 缃俊搴︽帓搴?
                if len(mkpts0) >= top_k:
                    idx = np.argsort(-mconf)[:top_k]  # 鍙栫疆淇″害鏈€楂?top_k 鍧愭爣
                    mkpts0 = mkpts0[idx]
                    mkpts1 = mkpts1[idx]
                    mconf = mconf[idx]
                elif len(mkpts0) < 3:  # 浠嶇劧涓嶅鐢ㄤ簬 Kabsch锛堣嚦灏戦渶瑕?鐐癸級
                    print(f"error: len(mkpts0)<3 after filtering, got {len(mkpts0)}")
                    R_preds.append(torch.eye(3, device=device))
                    t_preds.append(torch.zeros(1, 3, device=device))
                    continue
            else:
                # 鉁?涓嶈繃婊?
                if len(mkpts0) < 3:
                    print(f"error: len(mkpts0)<3 (no filtering mode), got {len(mkpts0)}")
                    R_preds.append(torch.eye(3, device=device))
                    t_preds.append(torch.zeros(1, 3, device=device))
                    continue

            # 8) 鍥炴姇褰卞埌 3D
            depth0 = batch['depth0'][i].detach().cpu().numpy()#/10.
            depth1 = batch['depth1'][i].detach().cpu().numpy()#/10.
            #print("depth0:",depth0)
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
            #R_t = torch.from_numpy(R_np).float().to(device)
            #t_t = torch.from_numpy(t_np).float().to(device).unsqueeze(0) / 1000.0

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
        # 鍏煎浣犱箣鍓嶇殑鑷畾涔?collate 瀛楁
        if 'pose' in batch and 'T_0to1' not in batch:
            batch['T_0to1'] = batch['pose']

        # 鍓嶅悜 + 寰楀埌棰勬祴浣嶅Э
        out = self.forward_once(batch)

        # GT 缁濆浣嶅Э锛坬uery锛?
        T_q_gt = batch['item_q_pose']  # [B,4,4]
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)  # [B,1,3]锛屽崟浣?m

        # 濮挎€佹崯澶?
        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )

        total_loss = out['mask_loss'] + pose_loss

        # ---- 鏃ュ織 ----
        self.log('train/mask_loss', out['mask_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mask_iou', out['mask_iou'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_loss', pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/pose_rot_loss', pose_rot_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_trans_loss', pose_trans_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        # ---- 鍗婁釜 epoch 瑙﹀彂 ADD(S)-0.1D 璇勪及 ----
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
        # 纭繚 VSD 瀵硅薄宸茬粡鍔犺浇
        self._load_vsd_objects()

        out = self.forward_once(batch)

        T_q_gt = batch['item_q_pose']
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)

        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )
        total_loss = out['mask_loss'] + pose_loss

        # 璁＄畻ADD(S)-0.1d
        add01d_acc = self.compute_add01d(batch, out['R_pred'], out['t_pred'])

        # 璁＄畻MSSD, MSPD, VSD
        mssd_scores, mspd_scores, vsd_scores = [], [], []

        for i in range(len(batch['obj_id'])):
            obj_id = batch['obj_id'][i]

            #  鐢ㄥ垵濮嬪寲鏃朵繚瀛樼殑 obj 淇℃伅
            obj_model = self.obj_models[obj_id]
            obj_diam = self.obj_diams[obj_id]
            obj_sym = self.obj_symms[obj_id]

            # 瀹夊叏澶勭悊锛氬鏋?obj_id 涓嶅湪 renderer 鍐咃紝璺宠繃 VSD
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



                # 杞崲涓烘绫?
                pred_R = pred_pose[:3, :3].cpu().numpy()
                pred_t = (pred_pose[:3, 3] * 1000).cpu().numpy().reshape(3, 1)
                gt_R = gt_pose[:3, :3].cpu().numpy()
                gt_t = (gt_pose[:3, 3] * 1000).cpu().numpy().reshape(3, 1)

                #print("gt_pose",gt_pose)
                obj_sym = self.obj_symms.get(obj_id, None)
                #澶勭悊濂界殑 ndarray
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
                # # 澶勭悊 depth  0
                # depth_proc = depth.copy()
                # depth_proc[depth_proc == 0] = depth_proc.max()
                # vsd_err = vsd(pred_R, pred_t, gt_R, gt_t, depth_proc, K,
                #               self.vsd_delta, self.vsd_taus, True, obj_diam,
                #               self.vsd_renderer, obj_id)

            # 璁＄畻 recall 鍒嗘暟
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

        # 鑱氬悎鎵€鏈塨atch鐨勬寚鏍?
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

        # 璁板綍鎵€鏈夋寚鏍?
        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=('loss' in name or 'acc' in name))

        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # 鑱氬悎鎵€鏈塨atch鐨勬寚鏍?
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

        # 璁板綍鎵€鏈夋寚鏍?
        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=('loss' in name or 'acc' in name))

        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # 鑱氬悎鎵€鏈塨atch鐨勬寚鏍?
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

        # 璁板綍鎵€鏈夋寚鏍?
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
                'monitor': 'val/loss',  # 鉁?validation_step 鎴?epoch_end log  key 瀵瑰簲
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # -------------------------
    #   ADD(S)-0.1D 璇勪及
    # -------------------------
    def compute_add01d(self, batch, R_pred, t_pred):
        """璁＄畻ADD(S)-0.1d鎸囨爣"""
        if not hasattr(self, 'obj_models'):
            return torch.tensor(0.0)

        total, success = 0, 0
        for i in range(len(batch['obj_id'])):
            obj_id = batch['obj_id'][i]

            obj_model = self.obj_models[obj_id]
            obj_diam = self.obj_diams[obj_id]
            obj_sym = self.obj_symms.get(obj_id, None)

            # 鏋勯€犻娴嬪拰GT濮挎€?
            pred_pose = torch.eye(4, device=self.device)
            pred_pose[:3, :3] = R_pred[i]
            pred_pose[:3, 3] = t_pred[i].squeeze()

            gt_pose = batch['item_q_pose'][i]

            # 璁＄畻ADD(S)
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
            R_pred = out['R_pred']           # [B,3,3]
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

    def normalize_pose(self,R, t):
        # R: 3x3 鏃嬭浆鐭╅樀
        # t: 3x1 骞崇Щ鍚戦噺
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:  # 澶勭悊鍙嶅皠
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho, t

    # def debug_matcher(self, batch):
    #     """
    #     Debug FeatureMatcher + mask filtering + visualization + top-8 mconf filtering
    #     淇濆瓨鍘熷浘銆丟T mask杩囨护鍥俱€丗eatureMatcher鍖归厤鍙鍖栧浘
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
    #     save_dir = "debug_matcher"
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
    #         # 1) 淇濆瓨鍘熷浘
    #         # -------------------------
    #         vutils.save_image(img0, f"{save_dir}/image0_raw_{i}.png", normalize=True)
    #         vutils.save_image(img1, f"{save_dir}/image1_raw_{i}.png", normalize=True)
    #
    #         # -------------------------
    #         # 2) GT mask 杩囨护鍥?
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
    #             # NTSC鏍囧噯鏉冮噸锛屾洿璐磋繎瑙嗚鐗规€?
    #             r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    #             gray = 0.299 * r + 0.587 * g + 0.114 * b
    #             return gray.float()
    #
    #         img0_gray_filtered = rgb_to_gray(img0_filtered)
    #         img1_gray_filtered = rgb_to_gray(img1_filtered)
    #         # -------------------------
    #         # 3锛夐€佸叆 FeatureMatcher
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
    #         # if浣跨敤 GT 鎺╃爜
    #         m0 = batch['mask0_gt'][i].detach().cpu().numpy()  # 鈽?
    #         m1 = batch['mask1_gt'][i].detach().cpu().numpy()  # 鈽?
    #         # else:
    #         #     # if pred mask
    #         #     m0 = pred_mask0_bin[i].detach().cpu().numpy()
    #         #     m1 = pred_mask1_bin[i].detach().cpu().numpy()
    #
    #         if len(mkpts0) == 0:
    #             print("error,len(mkpts0)<0 after matcher")
    #             continue
    #
    #         # # 鎸夋帺鐮佽繃婊ゅ叧閿偣
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
    #         # 鉁?鎸夋帺鐮佽繃婊ゅ叧閿偣
    #         in_mask = (m0[mkpts0[:, 1].round().astype(int),
    #         mkpts0[:, 0].round().astype(int)] > 0) & \
    #                   (m1[mkpts1[:, 1].round().astype(int),
    #                   mkpts1[:, 0].round().astype(int)] > 0)
    #
    #         mkpts0 = mkpts0[in_mask]
    #         mkpts1 = mkpts1[in_mask]
    #         mconf = mconf[in_mask]  # 鈽?鍚屾杩囨护缃俊搴?
    #
    #         # -------------------------
    #         # 4锛夋寜缃俊搴︽帓搴忥紝鍙?top-8
    #         # -------------------------
    #         idx = np.argsort(-mconf)[:8]
    #         mkpts0 = mkpts0[idx]
    #         mkpts1 = mkpts1[idx]
    #         mconf = mconf[idx]
    #
    #         # -------------------------
    #         # 5锛夌敾鍖归厤鍙鍖栧浘
    #         # -------------------------
    #         img0_vis = (img0.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #         img1_vis = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #
    #         # 鉁?鎷兼帴骞剁‘淇?OpenCV-friendly 鏍煎紡
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
    def debug_matcher(self, batch):
        """
        Debug FeatureMatcher + mask filtering + visualization + top-8 mconf filtering
        淇濆瓨鍘熷浘銆丟T mask杩囨护鍥俱€丗eatureMatcher鍖归厤鍙鍖栧浘 +
        淇濆瓨GT pose涓嶱red pose + mask鐐逛簯 + 3D鍏抽敭鐐瑰彲瑙嗗寲
        """
        import os, torch, numpy as np, cv2
        import torchvision.utils as vutils
        from PIL import Image
        try:
            import open3d as o3d
        except Exception:
            o3d = None

        save_dir = "debug_matcher"
        os.makedirs(save_dir, exist_ok=True)

        # pose log
        pose_log = os.path.join(save_dir, "pose_log.txt")

        # 鉁?娓呯┖ pose_log 鏂囦欢锛堣鐩栨ā寮忥級
        with open(pose_log, "w") as f:
            f.write("=== Pose Debug Log ===\n")

        B = batch['image0'].shape[0]
        num_save = min(3, B)
        device = batch['image0'].device

        for i in range(num_save):

            # 鉁?sample_i 瀛愭枃浠跺す
            sample_dir = f"{save_dir}/sample_{i}"
            os.makedirs(sample_dir, exist_ok=True)

            img0 = batch['image0'][i:i + 1]  # [1,3,H,W]
            img1 = batch['image1'][i:i + 1]
            mask0 = batch['mask0_gt'][i]
            mask1 = batch['mask1_gt'][i]
            depth0 = batch['depth0'][i].detach().cpu().numpy()/10.
            depth1 = batch['depth1'][i].detach().cpu().numpy()/10.
            K0 = batch['K_color0'][i].detach().cpu().numpy()
            K1 = batch['K_color1'][i].detach().cpu().numpy()

            # 1) 淇濆瓨鍘熷浘
            vutils.save_image(img0, f"{sample_dir}/image0_raw.png", normalize=True)
            vutils.save_image(img1, f"{sample_dir}/image1_raw.png", normalize=True)

            # 2) 淇濆瓨 mask 杩囨护鍥?
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

            # 3) FeatureMatcher
            match_batch = {'image0': img0_gray_filtered, 'image1': img1_gray_filtered}
            with torch.no_grad():
                self.matcher.eval()
                self.matcher(match_batch)

            mkpts0 = match_batch['mkpts0_f'].cpu().numpy()
            mkpts1 = match_batch['mkpts1_f'].cpu().numpy()
            mconf = match_batch['mconf'].cpu().numpy()

            # mask filter (鐢℅T mask)
            m0 = mask0.detach().cpu().numpy()
            m1 = mask1.detach().cpu().numpy()
            in_mask = (m0[mkpts0[:, 1].round().astype(int), mkpts0[:, 0].round().astype(int)] > 0) & \
                      (m1[mkpts1[:, 1].round().astype(int), mkpts1[:, 0].round().astype(int)] > 0)

            mkpts0 = mkpts0[in_mask]
            mkpts1 = mkpts1[in_mask]
            mconf = mconf[in_mask]

            if len(mkpts0) < 3:
                print("鉂?less than 3 correspondences after mask filtering")
                continue

            # 4) Top-K filter (鍜?forward_once 淇濇寔涓€鑷?
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
                    print(f"鉂?less than 3 correspondences after top-k (got {len(mkpts0)})")
                    continue

            # 5) 璋冩暣鍐呭弬锛堝鏋滈渶瑕侊級涓庡洖鎶曞奖
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
                print("鉂?less than 3 valid 3D correspondences after backprojection")
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

                # 10) 杞埌缁濆浣嶅Э
            T_a = batch['item_a_pose'][i].detach().cpu().numpy()
            T_rel = np.eye(4, dtype=np.float32)
            T_rel[:3, :3] = R_np
            T_rel[:3, 3] = (t_np / 1000.0)
            T_q_pred = T_rel @ T_a
            R = torch.from_numpy(T_q_pred[:3, :3]).float().to(device)
            t = torch.from_numpy(T_q_pred[:3, 3]).float().to(device).unsqueeze(0)
            # R = R_np
            # t = t_np / 1000.

            # ---- 鉁?淇濆瓨GT & Pred pose ----
            R_gt = batch['item_q_pose'][i, :3, :3].cpu().numpy()
            t_gt = batch['item_q_pose'][i, :3, 3].cpu().numpy()
            with open(pose_log, "a") as f:
                f.write(f"\n==== Sample {i} ====\n")
                f.write("GT_R:\n" + str(R_gt) + "\nGT_t:" + str(t_gt) + "\n")
                f.write("Pred_R:\n" + str(R) + "\nPred_t:" + str(t) + "\n")

            # ==== 鉁?image0 mask 鈫?3D 鐐逛簯 ====
            mask_pts0 = np.argwhere(m0 > 0)
            if mask_pts0.shape[0] > 0:
                mask_uv0 = mask_pts0[:, [1, 0]]
                if hasattr(self, "backproject"):
                    pc_mask0, _ = self.backproject(mask_uv0, depth0, K0_adj)
                else:
                    pc_mask0, _ = _local_backproject(mask_uv0, depth0, K0_adj)
            else:
                pc_mask0 = np.zeros((0, 3))

            # ==== 鉁?image1 mask 鈫?3D 鐐逛簯锛堟柊鍔狅級 ====
            mask_pts1 = np.argwhere(m1 > 0)
            if mask_pts1.shape[0] > 0:
                mask_uv1 = mask_pts1[:, [1, 0]]
                if hasattr(self, "backproject"):
                    pc_mask1, _ = self.backproject(mask_uv1, depth1, K1_adj)
                else:
                    pc_mask1, _ = _local_backproject(mask_uv1, depth1, K1_adj)
            else:
                pc_mask1 = np.zeros((0, 3))

            # final FeatureMatcher 3D keypoints (image0 domain)
            pc_key0 = A.copy()
            pc_key1 = Bp.copy()
            # ==== 鉁?淇濆瓨Ply ====
            if o3d is not None:
                # image0 mask cloud
                if pc_mask0.shape[0] > 0:
                    pcd0 = o3d.geometry.PointCloud()
                    pcd0.points = o3d.utility.Vector3dVector(pc_mask0)
                    pcd0.paint_uniform_color([0.5, 0.5, 0.5])
                    o3d.io.write_point_cloud(f"{sample_dir}/pc_mask_img0.ply", pcd0)

                # image1 mask cloud锛堟柊鍔狅級
                if pc_mask1.shape[0] > 0:
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(pc_mask1)
                    pcd1.paint_uniform_color([0.0, 0.5, 1.0])  # 钃濊壊
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

                    # ==== 鉁?image1 娣峰悎鐐逛簯 ====
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


            # ==== 鉁?3D matplotlib 淇濆瓨 ====

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

                # ---- 鉁?draw match lines between keypoints in 3D ----
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
            #     # === 鐐逛簯 ===
            #     pcd_mask0 = make_pcd(pc_mask0, np.array([0.6, 0.6, 0.6]))  # gray
            #     pcd_mask1 = make_pcd(pc_mask1, np.array([0.2, 0.6, 1.0]))  # blue
            #     pcd_key0 = make_pcd(pc_key0, np.array([1.0, 0.2, 0.2]))  # red
            #     pcd_key1 = make_pcd(pc_key1, np.array([0.2, 1.0, 0.2]))  # green
            #
            #     for p in [pcd_mask0, pcd_mask1, pcd_key0, pcd_key1]:
            #         if p is not None:
            #             vis_pcd.append(p)
            #
            #     # === 杩炵嚎锛堝叧閿偣鍖归厤锛?==
            #     if pc_key0.shape[0] > 0 and pc_key1.shape[0] > 0:
            #         lines = []
            #         for i in range(min(pc_key0.shape[0], pc_key1.shape[0])):
            #             lines.append([i, i + pc_key0.shape[0]])
            #
            #         # 鍚堝苟鍏抽敭鐐?
            #         all_keys = np.vstack([pc_key0, pc_key1])
            #         line_set = o3d.geometry.LineSet()
            #         line_set.points = o3d.utility.Vector3dVector(all_keys)
            #         line_set.lines = o3d.utility.Vector2iVector(lines)
            #         line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in lines])  # black
            #
            #         vis_pcd.append(line_set)
            #
            #     # === 鍙氦浜掓煡鐪?===
            #     #o3d.visualization.draw_geometries(vis_pcd, window_name="3D Keypoint Match Viewer")
            #     o3d.io.write_point_cloud(f"{sample_dir}/vis_all_points.ply", pcd_mask0 + pcd_mask1 + pcd_key0 + pcd_key1)
            #
            # except Exception as e:
            #     print("Open3D visualization failed:", e)
            #     pass

            # 鉁?FeatureMatcher match vis
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


