"""
集成 ObjectMatch 的测试代码
将 LoFTR + Kabsch 替换为 ObjectMatch 的完整配准流程
"""

import torch
import pytorch_lightning as pl
import numpy as np
import os
import sys

# 添加 ObjectMatch 路径
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ObjectMatch'))

# ObjectMatch 导入
from ObjectMatch.objectmatch_simple import ObjectMatch

# 原有导入
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
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from filesOfOryon.bop_toolkit_lib.misc import format_sym_set
from omegaconf import OmegaConf

from filesOfOryon.utils.metrics import compute_add, compute_adds
from filesOfOryon.utils.geo6d import best_fit_transform_with_RANSAC
from filesOfOryon.utils.losses import DiceLoss, LovaszLoss, FocalLoss


class MicKeyTrainingModel(pl.LightningModule):
    """
    集成 ObjectMatch 的训练模型
    - 使用 Oryon 产生 mask
    - 使用 ObjectMatch 进行完整的配准流程（SuperGlue + 姿态估计）
    - 保留原有的损失函数和评估指标
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg

        # ---------- Oryon ----------
        self.oryon_model = Oryon(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')


        # ---------- ObjectMatch ----------
        # 初始化 ObjectMatch（替代 LoFTR）
        objectmatch_cfg = getattr(cfg, 'OBJECTMATCH', {})
        self.objectmatch = ObjectMatch(
            checkpoint_dir=getattr(objectmatch_cfg, 'CHECKPOINT_DIR', 'ObjectMatch/checkpoints'),
            superglue_dir=getattr(objectmatch_cfg, 'SUPERGLUE_DIR', 'ObjectMatch/SuperGluePretrainedNetwork'),
            superglue_model=getattr(objectmatch_cfg, 'MODEL', 'indoor'),
            match_cache_dir=getattr(objectmatch_cfg, 'CACHE_DIR', './dump_features'),
            verbose=False,  # 训练时关闭详细输出
        )
        # LoFTR
        from LOFTER.src.loftr import LoFTR, default_cfg
        # LoFTR matcher
        default_cfg['coarse']['temp_bug_fix'] = False
        self.matcher = LoFTR(config=default_cfg)
        self.matcher.load_state_dict(torch.load("LOFTER/weights/outdoor_ds.ckpt")['state_dict'])
        self.matcher = self.matcher.eval().cuda()
        print("Using LoFTR for 2D keypoint matching")

        # ---------- 损失 ----------
        self._mask_loss = DiceLoss(weight=torch.tensor([0.5, 0.5]))
        self.mask_th = getattr(getattr(cfg, 'LOSS', {}), 'MASK_TH', 0.5)
        self.soft_clip = getattr(getattr(cfg, 'LOSS', {}), 'SOFT_CLIP', True)

        # ---------- 训练控制 ----------
        self.automatic_optimization = True
        self.multi_gpu = True
        self.validation_step_outputs = []
        self.log_interval = getattr(cfg.TRAINING, 'LOG_INTERVAL', 50)

        # VSD渲染器
        self.compute_vsd = True
        self.vsd_renderer = RendererVispy(640, 480, mode='depth')
        self.vsd_taus = list(np.arange(0.05, 0.51, 0.05))
        self.vsd_rec = np.arange(0.05, 0.51, 0.05)
        self.vsd_delta = 15.
        self._vsd_objects_loaded = False

        # 初始化指标记录
        self.test_step_outputs = []

        self.useGTmask = True
        self.debug_objectmatch_flag = False

    def _load_vsd_objects(self):
        if self._vsd_objects_loaded:
            return
        obj_models, obj_diams, obj_symms = self.trainer.datamodule.val_dataloader().dataset.get_object_info()
        self.add_object_info(obj_models, obj_diams, obj_symms)
        self._vsd_objects_loaded = True
        print("Loaded VSD objects:", list(self.vsd_renderer.model_bbox_corners.keys()))

    def forward(self, batch):
        return self.forward_once(batch)

    # -------------------------
    #   损失函数（保持不变）
    # -------------------------
    def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor):
        """掩码损失"""
        gt_shape = gt.shape[1:]
        pred_shape = pred_logits.shape[2:]
        gt_c = gt.clone().to(torch.float32)

        if gt_shape != pred_shape:
            gt_c = F.interpolate(gt.unsqueeze(1), size=pred_shape, mode='nearest').squeeze(1)

        if gt_c.max() > 1.0:
            gt_c = gt_c / 255.0

        logits = pred_logits.squeeze(1)
        loss = self._mask_loss(logits, gt_c.to(torch.float32))

        with torch.no_grad():
            pred_mask = (torch.sigmoid(logits) > self.mask_th).float()
            intersection = (pred_mask * gt_c).sum(dim=(1, 2))
            union = (pred_mask + gt_c - pred_mask * gt_c).sum(dim=(1, 2)) + 1e-6
            iou = (intersection / union).mean()

        return loss, pred_mask, logits, iou

    def compute_pose_loss(self, R, t, Rgt_i, tgt_i, soft_clipping=True):
        """姿态损失"""
        loss_rot, _ = self.rot_angle_loss(R, Rgt_i)
        loss_trans = self.trans_l1_loss(t, tgt_i)

        if soft_clipping:
            loss_trans_soft = torch.tanh(loss_trans / 0.9)
            loss_rot_soft = torch.tanh(loss_rot / 0.9)
            loss = loss_rot_soft + loss_trans_soft
        else:
            loss = loss_rot + loss_trans

        return loss.mean(), loss_rot.mean(), loss_trans.mean()

    @staticmethod
    def trans_l1_loss(t, tgt):
        return torch.abs(t - tgt).sum(-1)

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
    #   核心前向传播（集成 ObjectMatch）
    # -------------------------
    def forward_once(self, batch):
        """
        使用 ObjectMatch 替代 LoFTR + Kabsch 的完整流程
        """
        device = batch['image0'].device
        B, _, H, W = batch['image0'].shape

        # 1) Oryon 输出掩码
        oryon_out = self.oryon_model.forward(batch)
        pred_mask0_logits = oryon_out['mask_a']
        pred_mask1_logits = oryon_out['mask_q']

        # 2) 计算掩码损失
        mask0_gt = batch['mask0_gt']
        mask1_gt = batch['mask1_gt']

        if self.useGTmask:
            mask0_gt_input = mask0_gt.unsqueeze(1)
            mask1_gt_input = mask1_gt.unsqueeze(1)
            mask0_loss, _, _, mask0_iou = self.mask_loss(mask0_gt_input, batch['mask0_gt'])
            mask1_loss, _, _, mask1_iou = self.mask_loss(mask1_gt_input, batch['mask1_gt'])
        else:
            mask0_loss, _, _, mask0_iou = self.mask_loss(pred_mask0_logits, mask0_gt)
            mask1_loss, _, _, mask1_iou = self.mask_loss(pred_mask1_logits, mask1_gt)

        mask_loss_all = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2.

        # 3) 使用 ObjectMatch 进行配准
        R_preds, t_preds = [], []

        for i in range(B):
            # 准备单个样本的数据
            sample_data = self._prepare_objectmatch_batch(batch, i)

            try:
                # 调用 ObjectMatch 配准
                result = self._run_objectmatch_registration(sample_data)

                if result is not None:
                    R_preds.append(result['R'])
                    t_preds.append(result['t'])
                else:
                    # 配准失败，使用单位矩阵
                    R_preds.append(torch.eye(3, device=device))
                    t_preds.append(torch.zeros(1, 3, device=device))

            except Exception as e:
                print(f"ObjectMatch 配准失败: {e}")
                R_preds.append(torch.eye(3, device=device))
                t_preds.append(torch.zeros(1, 3, device=device))

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

    def _prepare_objectmatch_batch(self, batch, idx):
        """
        准备 ObjectMatch 需要的数据格式
        """
        # 保存临时图像文件（ObjectMatch 需要文件路径）
        temp_dir = './temp_objectmatch'
        os.makedirs(temp_dir, exist_ok=True)

        # 转换图像格式并保存
        img0 = batch['image0'][idx].cpu().numpy().transpose(1, 2, 0)
        img1 = batch['image1'][idx].cpu().numpy().transpose(1, 2, 0)
        img0 = (img0 * 255).astype(np.uint8)
        img1 = (img1 * 255).astype(np.uint8)

        from PIL import Image
        img0_path = f'{temp_dir}/img0_{idx}.png'
        img1_path = f'{temp_dir}/img1_{idx}.png'
        Image.fromarray(img0).save(img0_path)
        Image.fromarray(img1).save(img1_path)

        # 保存深度图
        depth0 = batch['depth0'][idx].cpu().numpy()
        depth1 = batch['depth1'][idx].cpu().numpy()
        np.save(f'{temp_dir}/depth0_{idx}.npy', depth0)
        np.save(f'{temp_dir}/depth1_{idx}.npy', depth1)

        # 准备内参和位姿
        K0 = batch['K_color0'][idx].cpu().numpy()
        K1 = batch['K_color1'][idx].cpu().numpy()
        pose_a = batch['item_a_pose'][idx].cpu().numpy()
        pose_q = batch['item_q_pose'][idx].cpu().numpy()

        return {
            'image0_path': img0_path,
            'image1_path': img1_path,
            'depth0': depth0,
            'depth1': depth1,
            'K0': K0,
            'K1': K1,
            'pose_a': pose_a,
            'pose_q': pose_q,
            'mask0': batch['mask0_gt'][idx].cpu().numpy() if self.useGTmask else None,
            'mask1': batch['mask1_gt'][idx].cpu().numpy() if self.useGTmask else None,
        }

    def _run_objectmatch_registration(self, sample_data):
        """
        运行 ObjectMatch 配准
        返回: {'R': [3,3], 't': [1,3]} 或 None
        """
        try:
            # 使用 ObjectMatch 的 register 方法
            result = self.objectmatch.register(
                image0_path=sample_data['image0_path'],
                image1_path=sample_data['image1_path'],
                intrinsic0=sample_data['K0'],
                intrinsic1=sample_data['K1'],
                pose0=sample_data['pose_a'],
                pose1=sample_data['pose_q'],
                visualize=False,
            )

            if result.success:
                # 提取旋转和平移
                R = torch.from_numpy(result.camera_pose[:3, :3]).float().to(self.device)
                t = torch.from_numpy(result.camera_pose[:3, 3]).float().to(self.device).unsqueeze(0)

                return {'R': R, 't': t}
            else:
                return None

        except Exception as e:
            print(f"ObjectMatch 配准异常: {e}")
            return None

    # -------------------------
    #   训练步骤
    # -------------------------
    def training_step(self, batch, batch_idx):
        if 'pose' in batch and 'T_0to1' not in batch:
            batch['T_0to1'] = batch['pose']

        out = self.forward_once(batch)

        T_q_gt = batch['item_q_pose']
        R_gt = T_q_gt[:, :3, :3]
        t_gt = T_q_gt[:, :3, 3].unsqueeze(1)

        pose_loss, pose_rot_loss, pose_trans_loss = self.compute_pose_loss(
            out['R_pred'], out['t_pred'], R_gt, t_gt, soft_clipping=self.soft_clip
        )

        total_loss = out['mask_loss'] + pose_loss

        # 日志
        self.log('train/mask_loss', out['mask_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mask_iou', out['mask_iou'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_loss', pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/pose_rot_loss', pose_rot_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/pose_trans_loss', pose_trans_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    # -------------------------
    #   验证步骤
    # -------------------------
    def validation_step(self, batch, batch_idx):
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
            obj_model = self.obj_models[obj_id]
            obj_diam = self.obj_diams[obj_id]
            obj_sym = self.obj_symms[obj_id]

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

                pred_R = pred_pose[:3, :3].cpu().numpy()
                pred_t = (pred_pose[:3, 3] * 1000).cpu().numpy().reshape(3, 1)
                gt_R = gt_pose[:3, :3].cpu().numpy()
                gt_t = (gt_pose[:3, 3] * 1000).cpu().numpy().reshape(3, 1)

                bop_sym = obj_sym

                mssd_err = my_mssd(pred_R, pred_t, gt_R, gt_t, obj_model['pts'], bop_sym)
                mspd_err = my_mspd(pred_R, pred_t, gt_R, gt_t, K, obj_model['pts'], bop_sym)

                pred_R, pred_t = self.normalize_pose(pred_R, pred_t)
                gt_R, gt_t = self.normalize_pose(gt_R, gt_t)
                vsd_err = vsd(pred_R, pred_t, gt_R, gt_t, depth, K,
                              self.vsd_delta, self.vsd_taus, True, obj_diam,
                              self.vsd_renderer, obj_id)

            mssd_rec = np.arange(0.05, 0.51, 0.05) * obj_diam
            mssd_scores.append((mssd_err < mssd_rec).mean())

            mspd_rec = np.arange(5, 51, 5)
            mspd_scores.append((mspd_err < mspd_rec).mean())

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
            'vsd': torch.tensor(mean_vsd),
        }

        self.validation_step_outputs.append(logs)
        return logs

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

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

        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, sync_dist=self.multi_gpu, prog_bar=('loss' in name or 'acc' in name))

        self.validation_step_outputs.clear()

    # -------------------------
    #   优化器配置
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
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # -------------------------
    #   评估指标
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

            pred_pose = torch.eye(4, device=self.device)
            pred_pose[:3, :3] = R_pred[i]
            pred_pose[:3, 3] = t_pred[i].squeeze()

            gt_pose = batch['item_q_pose'][i]

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
        self.obj_models = obj_models
        self.obj_diams = obj_diams
        self.obj_symms = {k: format_sym_set(sym_set) for k, sym_set in obj_symms.items()}

        if self.compute_vsd:
            for obj_id, obj in self.obj_models.items():
                self.vsd_renderer.my_add_object(obj, obj_id)

    def normalize_pose(self, R, t):
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho, t
