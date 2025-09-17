# import pytorch_lightning as pl
# import torch
#
# from lib.models.MicKey.modules.compute_correspondences import ComputeCorrespondences
# from lib.models.MicKey.modules.utils.probabilisticProcrustes import e2eProbabilisticProcrustesSolver
# from lib.models.Oryon.oryon import Oryon
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LOFTER.src.loftr import LoFTR, default_cfg
from lib.models.Oryon.oryon import Oryon
import pytorch_lightning as pl

class MickeyRelativePose(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.oryon_model = Oryon(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')

        default_cfg['coarse']['temp_bug_fix'] = False
        self.matcher = LoFTR(config=default_cfg)
        self.matcher.load_state_dict(torch.load("LOFTER/weights/outdoor_ds.ckpt")['state_dict'])
        self.matcher = self.matcher.eval().cuda()

        # 直接用 BCEWithLogitsLoss 作为底层 mask loss
        self._mask_loss = nn.BCEWithLogitsLoss()
        self.mask_th = 0.5  # IoU 计算时的阈值
        self.pose_loss = nn.MSELoss()

    def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor):
        """
        pred_logits: [B, 1, H_pred, W_pred] — 掩码 logits
        gt:          [B, H_gt, W_gt]         — ground truth binary mask
        """
        gt_shape = gt.shape[1:]
        pred_shape = pred_logits.shape[2:]

        gt_c = gt.clone().float()

        # 如果 shape 不匹配，就 resize GT mask
        if gt_shape != pred_shape:
            gt_c = F.interpolate(gt.unsqueeze(1), size=pred_shape, mode='nearest').squeeze(1)

        # Normalization 到 [0,1]
        if gt_c.max() > 1.0:
            gt_c = gt_c / 255.0

        pred_logits_s = pred_logits.squeeze(1)  # [B, H, W]
        loss = self._mask_loss(pred_logits_s, gt_c)

        with torch.no_grad():
            pred_mask = (torch.sigmoid(pred_logits_s) > self.mask_th).float()
            intersection = (pred_mask * gt_c).sum(dim=(1, 2))
            union = (pred_mask + gt_c - pred_mask * gt_c).sum(dim=(1, 2)) + 1e-6
            iou = (intersection / union).mean()

        return loss, pred_mask, pred_logits_s, iou

    def rgb_to_gray(self, tensor_rgb):
        gray = (0.2989 * tensor_rgb[:, 0, :, :] +
                0.5870 * tensor_rgb[:, 1, :, :] +
                0.1140 * tensor_rgb[:, 2, :, :])
        return gray.unsqueeze(1)

    def forward(self, data, return_inliers=False):
        oryon_output = self.oryon_model.forward(data)
        mask0_pred = oryon_output['mask_a']
        mask1_pred = oryon_output['mask_q']

        img0_gray = self.rgb_to_gray(data['image0']) * torch.sigmoid(mask0_pred)
        img1_gray = self.rgb_to_gray(data['image1']) * torch.sigmoid(mask1_pred)

        match_batch = {
            'image0': img0_gray,
            'image1': img1_gray,
            'mask0': mask0_pred,
            'mask1': mask1_pred
        }

        with torch.set_grad_enabled(self.training):
            self.matcher(match_batch)

        mkpts0 = match_batch['mkpts0_f']
        mkpts1 = match_batch['mkpts1_f']

        depth0 = data['depth0'][0].cpu().numpy()
        depth1 = data['depth1'][0].cpu().numpy()
        K0 = data['K_color0'][0].cpu().numpy()
        K1 = data['K_color1'][0].cpu().numpy()

        pts3d_0, valid0 = backproject(mkpts0.cpu().numpy(), depth0, K0)
        pts3d_1, valid1 = backproject(mkpts1.cpu().numpy(), depth1, K1)
        valid = valid0 & valid1

        if np.count_nonzero(valid) < 3:
            return None if self.training else (None, None)

        R_pred, t_pred = kabsch_umeyama(pts3d_0[valid], pts3d_1[valid])

        if self.training:
            return {
                'R_pred': R_pred,
                't_pred': t_pred,
                'mask0_pred': mask0_pred,
                'mask1_pred': mask1_pred,
                'mask0_gt': data['mask0_gt'],
                'mask1_gt': data['mask1_gt'],
                'pose_gt': data['pose']
            }
        else:
            return R_pred, t_pred

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        if outputs is None:
            return None

        # 用新 mask_loss 计算
        mask0_loss, _, _, mask0_iou = self.mask_loss(outputs['mask0_pred'], outputs['mask0_gt'])
        mask1_loss, _, _, mask1_iou = self.mask_loss(outputs['mask1_pred'], outputs['mask1_gt'])
        mask_loss_total = mask0_loss + mask1_loss
        mask_iou_mean = (mask0_iou + mask1_iou) / 2

        # 姿态损失
        R_gt = outputs['pose_gt'][0, :3, :3].cpu().numpy()
        t_gt = outputs['pose_gt'][0, :3, 3].cpu().numpy()
        pose_loss_val = np.linalg.norm(outputs['R_pred'] - R_gt) + np.linalg.norm(outputs['t_pred'] - t_gt)

        total_loss = mask_loss_total + pose_loss_val

        # 日志
        self.log('train/mask_loss', mask_loss_total)
        self.log('train/mask_iou', mask_iou_mean)
        self.log('train/pose_loss', pose_loss_val)
        self.log('train/total_loss', total_loss)

        return total_loss
    def is_eval_model(self, is_eval):
        if is_eval:
            self.compute_matches.extractor.depth_head.eval()
            self.compute_matches.extractor.det_offset.eval()
            self.compute_matches.extractor.dsc_head.eval()
            self.compute_matches.extractor.det_head.eval()
        else:
            self.compute_matches.extractor.depth_head.train()
            self.compute_matches.extractor.det_offset.train()
            self.compute_matches.extractor.dsc_head.train()
            self.compute_matches.extractor.det_head.train()

# class MickeyRelativePose(pl.LightningModule):
#     # Compute the metric relative pose between two input images, with given intrinsics (for the pose solver).
#
#     def __init__(self, cfg):
#         super().__init__()
#
#         # Initialize Oryon model
#         self.oryon_model = Oryon(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')
#
#         # Define MicKey architecture and matching module:
#         self.compute_matches = ComputeCorrespondences(cfg)
#
#         # Metric solver
#         self.e2e_Procrustes = e2eProbabilisticProcrustesSolver(cfg)
#
#         self.is_eval_model(True)
#
#         self.step_counter = 0
#         self.mask_filter_start = 0  #开始使用掩码过滤
#
#     def forward(self, data, return_inliers=False):
#         # Get features from Oryon model
#         oryon_output = self.oryon_model.forward(data)
#         oryon_feats0 = oryon_output['featmap_a']
#         oryon_feats1 = oryon_output['featmap_q']
#
#     # Use Oryon features in Mickey
#         data['feats0'] = oryon_feats0
#         data['feats1'] = oryon_feats1
#         mask_a = oryon_output['mask_a']
#         mask_q = oryon_output['mask_q']
#
#         data['mask0'] = mask_a
#         data['mask1'] = mask_q
#
#         self.compute_matches(data)
#         print("scores shape:", data['scores'].shape)
#         print("kp_scores shape:", data['kp_scores'].shape)
#         # print("scores",data['scores'])
#         # print("kp_scores",data['kp_scores'])
#
#         data['final_scores'] = data['scores'] * data['kp_scores']
#
#
#
#         # #mask
#         # self.step_counter += 1  # 每次 forward 增加一次
#         # # 掩码过滤逻辑
#         # if self.step_counter >= self.mask_filter_start:
#         #     mask0 = torch.sigmoid(mask_a)
#         #     print(f"[Step {self.step_counter}] 掩码最大值/最小值：", mask0.max().item(), mask0.min().item())
#         #     mask0 = (mask0 >= 0.5).float()
#         #     mask1 = torch.sigmoid(mask_q)
#         #     print(f"[Step {self.step_counter}] 掩码最大值/最小值：", mask1.max().item(), mask1.min().item())
#         #     mask1 = (mask0 >= 0.5).float()
#         #
#         #
#         #     #  将掩码展开为一维 (B, 2304)
#         #     B, H, W = mask0.shape  # 48x48 = 2304
#         #     mask0_flat = mask0.view(B, -1)  # (B, 2304)
#         #     mask1_flat = mask1.view(B, -1)  # (B, 2304)
#         #
#         #     # 构造外积得到 mask_matrix，表示哪些匹配位置有效
#         #     # (B, 2304, 1) × (B, 1, 2304) => (B, 2304, 2304)
#         #     valid_mask_matrix = torch.bmm(mask0_flat.unsqueeze(2), mask1_flat.unsqueeze(1))  # binary mask for match pairs
#         #
#         #     # 将无效匹配位置设为 -1e9，防止参与 softmax 等操作
#         #     data['final_scores'] = data['final_scores'].masked_fill(valid_mask_matrix == 0, -1e9)
#         #
#         # else:
#         #     # 可选：打印提示当前还未启用掩码过滤
#         #     pass
#         # #mask0.shape=mask1.shape=(B,48,48)
#         # #print(data['scores'])
#
#         if return_inliers:
#             # Returns inliers list:
#             R, t, inliers, inliers_list = self.e2e_Procrustes.estimate_pose_vectorized(data, return_inliers=True)
#             data['inliers_list'] = inliers_list
#         else:
#             # If the inlier list is not needed:
#             R, t, inliers = self.e2e_Procrustes.estimate_pose_vectorized(data, return_inliers=False)
#
#         data['R'] = R
#         data['t'] = t
#         data['inliers'] = inliers
#
#         return R, t
#
#     def on_load_checkpoint(self, checkpoint):
#         # This function avoids loading DINOv2 which are not sotred in Mickey's checkpoint.
#         # This saves memory during training, since DINOv2 is frozen and not updated there is no need to store
#         # the weights in every checkpoint.
#
#         # Recover DINOv2 features from pretrained weights.
#         for param_tensor in self.compute_matches.state_dict():
#             if 'dinov2'in param_tensor:
#                 checkpoint['state_dict']['compute_matches.'+param_tensor] = \
#                     self.compute_matches.state_dict()[param_tensor]
#
#     def is_eval_model(self, is_eval):
#         if is_eval:
#             self.compute_matches.extractor.depth_head.eval()
#             self.compute_matches.extractor.det_offset.eval()
#             self.compute_matches.extractor.dsc_head.eval()
#             self.compute_matches.extractor.det_head.eval()
#         else:
#             self.compute_matches.extractor.depth_head.train()
#             self.compute_matches.extractor.det_offset.train()
#             self.compute_matches.extractor.dsc_head.train()
#             self.compute_matches.extractor.det_head.train()
