import torch
import torch.nn as nn
from lib.models.MicKey.modules.mickey_extractor import MicKey_Extractor
from lib.models.MicKey.modules.utils.feature_matcher import featureMatcher

class ComputeCorrespondences(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Feature extractor#input oryon feature ,output kpt,deps,dsc,ksc
        self.extractor = MicKey_Extractor(cfg['MICKEY'])

        self.dsc_dim = cfg['MICKEY']['DSC_HEAD']['LAST_DIM']

        # Feature matcher
        self.matcher = featureMatcher(cfg['FEATURE_MATCHER'])

        self.down_factor = cfg['MICKEY']['DINOV2']['DOWN_FACTOR']
        #self.down_factor =
        # self.training_step_count = 0  # 计数器
        # self.mask_filter_start_step =3000 #after step then mask

    def get_abs_kpts_coordinates(self, kpts):

        B, C, H, W = kpts.shape

        # Compute offset for every kp grid
        x_abs_pos = torch.arange(W).view(1, 1, W).tile([B, H, 1]).to(kpts.device)
        y_abs_pos = torch.arange(H).view(1, H, 1).tile([B, 1, W]).to(kpts.device)
        abs_pos = torch.concat([x_abs_pos.unsqueeze(1), y_abs_pos.unsqueeze(1)], dim=1)

        kpts_abs_pos = (kpts + abs_pos) * self.down_factor

        return kpts_abs_pos

    def prepare_kpts_dsc(self, kpt, depth, scr, dsc):

        B, _, H, W = kpt.shape
        num_kpts = (H * W)

        kpt = kpt.view(B, 2, num_kpts)
        depth = depth.view(B, 1, num_kpts)
        scr = scr.view(B, 1, num_kpts)
        dsc = dsc.view(B, self.dsc_dim, num_kpts)

        # print("scr:",scr)
        # print("dsc:",dsc)

        return kpt, depth, scr, dsc

    # Independent method to only combine matching and keypoint scores during training
    def kp_matrix_scores(self, sc0, sc1):


        # matrix with "probability" of sampling a correspondence based on keypoint scores only
        scores = torch.matmul(sc0.transpose(2, 1).contiguous(), sc1)
        return scores

    def filter_kpts_by_mask(self, kpts, dscs, depths, scores, masks, H, W, threshold=0.5):
        """
        同时筛选关键点、描述符、深度、得分，基于掩码
        """
        B, _, N = kpts.shape
        D = dscs.shape[1]
        device = kpts.device

        masks_bin = (torch.sigmoid(masks) > threshold).float()

        kpts_out, dscs_out, depths_out, scores_out = [], [], [], []

        for b in range(B):
            kp = kpts[b]  # [2, N]
            dsc = dscs[b]  # [D, N]
            dep = depths[b]  # [1, N]
            scr = scores[b]  # [1, N]
            mask = masks_bin[b, 0]  # [H, W]

            x = kp[0].round().long().clamp(0, W - 1)
            y = kp[1].round().long().clamp(0, H - 1)

            valid = mask[y, x] > 0
            print("valid_mask_pixel:",valid)
            if valid.sum() == 0:
                # fallback: use all
                valid = torch.ones_like(valid, dtype=torch.bool)

            kpts_out.append(kp[:, valid])
            dscs_out.append(dsc[:, valid])
            depths_out.append(dep[:, valid])
            scores_out.append(scr[:, valid])

        return (
            torch.stack(kpts_out),
            torch.stack(dscs_out),
            torch.stack(depths_out),
            torch.stack(scores_out),
        )

    def forward(self, data):

        # # Compute detection and descriptor maps
        # im0 = data['image0']
        # im1 = data['image1']
        #
        # # Extract independently features from im0 and im1
        # kps0, depth0, scr0, dsc0 = self.extractor(im0)
        # kps1, depth1, scr1, dsc1 = self.extractor(im1)

        # 使用特征，完全跳过原始图像
        if 'feats0' not in data or 'feats1' not in data:
            raise RuntimeError("必须传入特征(feats0/feats1)")

        print(f"Oryon特征形状: {data['feats0'].shape}")  # 应该在compute_correspondences.py中

        # 使用原始特征
        kps0, depth0, scr0, dsc0 = self.extractor(data['feats0'],data['mask0'])
        kps1, depth1, scr1, dsc1 = self.extractor(data['feats1'],data['mask1'])
        # print("after extract")
        # print("kps0:", kps0.shape)   #(B,2,48,48)
        # print("depth0:", depth0.shape) #(B,1,48,48)
        # print("scr0:", scr0.shape) #(B,1,48,48)
        # print("dsc0:", dsc0.shape) #(B,128,48,48)

        kps0 = self.get_abs_kpts_coordinates(kps0)
        kps1 = self.get_abs_kpts_coordinates(kps1)
        # print("after getabs")
        # print("kps0:", kps0.shape)  #(B,2,48,48)
        # Log shape for logging purposes
        _, _, H_kp0, W_kp0 = kps0.shape
        _, _, H_kp1, W_kp1 = kps1.shape
        data['kps0_shape'] = [H_kp0, W_kp0]
        data['kps1_shape'] = [H_kp1, W_kp1]
        data['depth0_map'] = depth0
        data['depth1_map'] = depth1
        data['down_factor'] = self.down_factor

        # Reshape kpts and descriptors to [B, num_kpts, dim]
        kps0, depth0, scr0, dsc0 = self.prepare_kpts_dsc(kps0, depth0, scr0, dsc0)
        kps1, depth1, scr1, dsc1 = self.prepare_kpts_dsc(kps1, depth1, scr1, dsc1)
        # print("after prepare")
        # print("kps0:", kps0.shape)
        # print("depth0:", depth0.shape)
        # print("scr0:", scr0.shape)
        # print("dsc0:", dsc0.shape)

        #mask filtering
        # if self.training:
        #     self.training_step_count += 1
        # B, _, H, W = data['mask0'].shape
        # #if not self.training or self.training_step_count >= self.mask_filter_start_step:
        #     # 训练中且达到阈值后，或者eval时才进行筛选
        # kps0, dsc0, depth0, scr0 = self.filter_kpts_by_mask(kps0, dsc0, depth0, scr0, data['mask0'], H, W)
        # kps1, dsc1, depth1, scr1 = self.filter_kpts_by_mask(kps1, dsc1, depth1, scr1, data['mask1'], H, W)
        # else:
        #     # 训练早期不筛选，保持原样
        #     pass

        # get correspondences
        scores = self.matcher(kps0, dsc0, kps1, dsc1)

        data['kps0'] = kps0
        data['depth_kp0'] = depth0
        data['scr0'] = scr0
        data['kps1'] = kps1
        data['depth_kp1'] = depth1
        data['scr1'] = scr1
        data['scores'] = scores
        data['dsc0'] = dsc0
        data['dsc1'] = dsc1
        data['kp_scores'] = self.kp_matrix_scores(scr0, scr1)

        print("kp_scores shape:", data['kp_scores'].shape)
        print("scores shape:", data['scores'].shape)
        print("scr1", scr0.shape)
        print("scr2", scr1.shape)

        # # 裁剪 scr0, scr1 与 scores 保持一致，只保留前 N=128 个关键点得分
        # B, N, M = scores.shape  # scores 是 [B, N, M]
        # scr0 = scr0[:, :, :N]  # [B, 1, N]
        # scr1 = scr1[:, :, :M]  # [B, 1, M]
        # data['scr0'] = scr0
        # data['scr1'] = scr1
        # data['kp_scores'] = self.kp_matrix_scores(scr0, scr1)

        # # 打印关键点数量
        # kps0 = data["kps0"]  # (B,N,2)
        # kps1 = data["kps1"]
        # print(f"\n关键点数量: Anchor={kps0.shape[1]}, Query={kps1.shape[1]}")
        #
        # # 打印3D关键点（如果有）
        # if "depth_kp0" in data:
        #     print(f"3D关键点示例(前5个):\n{data['depth_kp0'][0, :5]}")

        return kps0, dsc0, kps1, dsc1
