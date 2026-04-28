import torch
import torch.nn as nn
#from lib.models.MicKey.modules.DINO_modules.dinov2 import vit_large
from lib.models.MicKey.modules.att_layers.transformer import Transformer_self_att
from lib.models.MicKey.modules.utils.extractor_utils import desc_l2norm, BasicBlock

class MicKey_Extractor(nn.Module):
    def __init__(self, cfg, dinov2_weights=None):
        super().__init__()
        self.cfg = cfg
        self.debug_visualize = cfg.get('DEBUG_VISUALIZE',True)  # 从配置读取或手动设置
        # # 动态适配输入通道
        # self.input_conv = nn.Conv2d(
        #     in_channels=256,  # Oryon特征通道
        #     out_channels=cfg['KP_HEADS']['BLOCKS_DIM'][0],
        #     kernel_size=1
        # )
        # 修改1：将feat改为实际输出的通道数（32）
        self.step_counter =0  # 控制何时启用掩码过滤
        self.mask_filter_start =0  # 超过这个步数才启用掩码过滤

        self.input_proj = nn.Sequential(
            nn.Conv2d(32, cfg['KP_HEADS']['BLOCKS_DIM'][0], 1),  # 32
            nn.BatchNorm2d(cfg['KP_HEADS']['BLOCKS_DIM'][0]),
            nn.ReLU()
        )

        # 获取配置参数
        block_dims = cfg['KP_HEADS']['BLOCKS_DIM']

        # 修改各头部模块，传入正确的输入通道数
        self.depth_head = DeepResBlock_depth(cfg,in_channels=block_dims[0])
        self.det_offset = DeepResBlock_offset(cfg,in_channels=block_dims[0])
        self.dsc_head = DeepResBlock_desc(cfg,in_channels=block_dims[0])
        self.det_head = DeepResBlock_det(cfg,in_channels=block_dims[0])

        # # 配置参数
        # in_channels = 32  # Oryon输出特征通道数
        # block_dims = cfg['KP_HEADS']['BLOCKS_DIM']
        # self.down_factor = cfg['DINOV2']['DOWN_FACTOR']  # 保持与原始配置一致

        # # Define DINOv2 extractor
        # self.dino_channels = cfg['DINOV2']['CHANNEL_DIM']
        # self.dino_downfactor = cfg['DINOV2']['DOWN_FACTOR']
        # if dinov2_weights is None:
        #     dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/"
        #                                                         "dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        #                                                         map_location="cpu")
        # vit_kwargs = dict(img_size= 518,
        #     patch_size= 14,
        #     init_values = 1.0,
        #     ffn_layer = "mlp",
        #     block_chunks = 0,
        # )
        #
        # self.dinov2_vitl14 = vit_large(**vit_kwargs)
        # self.dinov2_vitl14.load_state_dict(dinov2_weights)
        # self.dinov2_vitl14.requires_grad_(False)
        # self.dinov2_vitl14.eval()
        #
        # # Define whether DINOv2 runs on float16 or float32
        # if cfg['DINOV2']['FLOAT16']:
        #     self.amp_dtype = torch.float16
        #     self.dinov2_vitl14.to(self.amp_dtype)
        # else:
        #     self.amp_dtype = torch.float32

        # # Define MicKey's heads
        # self.depth_head = DeepResBlock_depth(cfg)
        # self.det_offset = DeepResBlock_offset(cfg)
        # self.dsc_head = DeepResBlock_desc(cfg)
        # self.det_head = DeepResBlock_det(cfg)

        # # 输入特征适配层（特征→MicKey输入）
        # self.feature_adapter = nn.Sequential(
        #     nn.Conv2d(in_channels, block_dims[0], 3, padding=1),
        #     nn.GroupNorm(8, block_dims[0]),
        #     nn.GELU()
        # )

    def forward(self, x, mask: torch.Tensor):

        # B, C, H, W = x.shape
        # x = x[:, :, :self.dino_downfactor * (H//self.dino_downfactor), :self.dino_downfactor * (W//self.dino_downfactor)]
        #
        # print("DINOv2输入形状:", x.shape)
        #
        # with torch.no_grad():
        #     dinov2_features = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
        #     dinov2_features = dinov2_features['x_norm_patchtokens'].permute(0, 2, 1).\
        #         reshape(B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor).float()
        #
        # print("DINOv2输出形状:", dinov2_features['x_norm_patchtokens'].shape)
        #
        # scrs = self.det_head(dinov2_features)
        # kpts = self.det_offset(dinov2_features)
        # depths = self.depth_head(dinov2_features)
        # dscs = self.dsc_head(dinov2_features)
        #
        # return kpts, depths, scrs, dscs

        """输入x: [B,256,192,192] 来自Oryon的特征图"""

        if x.shape[1] == 3:  # 如果是原始图像
            raise ValueError("本提取器仅处理特征")


        x = self.input_proj(x)  #通道适配

        # 特征适配
        #x = self.feature_adapter(x)

        # 四个任务头
        #(B,1,48,48)
        scrs = self.det_head(x)
        #B,2,48,48
        kpts = self.det_offset(x)
        depths = self.depth_head(x)
        dscs = self.dsc_head(x)


        #print("mask:",mask.shape)


        #mask
        if mask is not None:
            self.step_counter += 1  # 每次 forward 增加一次

            if self.step_counter >= self.mask_filter_start:
                # Step 1: 将网络输出的掩码 logits 转为概率
                mask = torch.sigmoid(mask)  #(B,1,48,48)
                print(f"[Step {self.step_counter}] 掩码最大值/最小值：", mask.max().item(), mask.min().item())


                prob_mask = mask
                binary_mask = (mask >= 0.5).float()
                if prob_mask.shape != scrs.shape:
                    print(f"[Warning] 掩码 shape 不一致: {prob_mask.shape} vs {scrs.shape}")
                else:
                    print("scrs.shape:",scrs.shape)
                    #scrs = scrs * binary_mask
                    scrs= scrs.masked_fill(binary_mask == 0,1e-9)
                    binary_mask_expanded = binary_mask.expand(-1, 128, -1, -1)
                    dscs = scrs.masked_fill(binary_mask_expanded == 0, 1e-9)

        # if mask is not None:
        #     # Step 1: 将网络输出的掩码 logits 转为概率
        #     mask = torch.sigmoid(mask)  #
        #     #mask = mask.clamp(min=0.5)
        #     print("掩码最大值/最小值：", mask.max().item(),mask.min().item())
        #     prob_mask = mask
        #
        #     assert prob_mask.shape == scrs.shape, f"掩码 shape 不一致: {prob_mask.shape} vs {scrs.shape}"
        #     scrs = scrs * prob_mask

            # # Step 2: 二值化
            # binary_mask = (mask > 0.3).float()  # 阈值
            # print("掩码非零占比：", binary_mask.mean().item())

            # Step 3: 匹配形状并应用
           #assert binary_mask.shape == scrs.shape, f"掩码 shape 不一致: {binary_mask.shape} vs {scrs.shape}"
           #scrs = scrs * binary_mask
        # if mask is not None:
        #     mask_float = mask.float()
        #     nonzero_ratio = (mask_float > 0).float().mean()
        #     print("掩码非零占比：", nonzero_ratio.item())
        #     assert mask.shape == scrs.shape, f"掩码 shape 不一致: {mask.shape} vs {scrs.shape}"
        #     scrs = scrs * mask
        #     # kpts = kpts * mask
        #     # depths = depths * mask
        #     # dscs = dscs * mask

        # ##############################################
        # # 新增可视化调试代码
        # if self.debug_visualize and torch.rand(1).item() < 0.5:  # 10%概率采样
        #     self._visualize_detection_results(
        #         input_features=x,
        #         keypoints=kpts,
        #         scores=scrs,
        #         depths=depths,
        #         save_dir="./debug_vis"
        #     )
        # # ##############################################
        #
        # print("\n=== 关键点诊断 ===")
        # print("置信度分数范围:", scrs.min().item(), scrs.max().item())  # 正常应为0~1
        # print("置信度均值:", scrs.mean().item())  # 应大于0.1
        # print("有效点数(>0.1):", (scrs > 0.1).sum().item())  # 检查是否有响应
        return kpts, depths, scrs, dscs

    def train(self, mode: bool = True):
        self.dsc_head.train(mode)
        self.depth_head.train(mode)
        self.det_offset.train(mode)
        self.det_head.train(mode)

    def _visualize_detection_results(self, input_features, keypoints, scores, depths, save_dir):
        """
        可视化关键点检测全流程结果
        Args:
            input_features: 输入特征图 [B,C,H,W]
            keypoints: 关键点坐标 [B,2,H,W] (offset形式)
            scores: 关键点置信度 [B,1,H,W]
            depths: 深度估计 [B,1,H,W]
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(save_dir, exist_ok=True)
        B, _, H, W = input_features.shape

        # 转换为CPU numpy
        feat_np = input_features[0, 0].cpu().detach().numpy()  # 取第一个通道可视化
        kpt_np = keypoints[0].cpu().detach().numpy()  # [2,H,W]
        scr_np = scores[0, 0].cpu().detach().numpy()  # [H,W]
        dep_np = depths[0, 0].cpu().detach().numpy()  # [H,W]

        # 生成网格坐标 (用于可视化offset)
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        abs_kpts = np.stack([grid_x + kpt_np[0], grid_y + kpt_np[1]], axis=0)  # 绝对坐标

        # 创建可视化画布
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 输入特征图
        axes[0, 0].imshow(feat_np, cmap='viridis')
        axes[0, 0].set_title("Input Feature Map")

        # 2. 关键点置信度热力图
        axes[0, 1].imshow(scr_np, cmap='hot')
        axes[0, 1].set_title("Keypoint Scores Heatmap")

        # 3. 关键点offset向量场
        axes[1, 0].quiver(grid_x[::4, ::4], grid_y[::4, ::4],
                          kpt_np[0, ::4, ::4], kpt_np[1, ::4, ::4],
                          scale=50, color='red')
        axes[1, 0].set_title("Keypoint Offsets")

        # 4. 深度估计
        axes[1, 1].imshow(dep_np, cmap='plasma')
        axes[1, 1].set_title("Depth Estimation")

        # 保存结果
        plt.savefig(f"{save_dir}/extractor_debug_{np.random.randint(1000)}.png", dpi=120)
        plt.close()
        print(f"[Debug] Visualizations saved to {save_dir}")


class DeepResBlock_det(torch.nn.Module):
    def __init__(self, config, in_channels=None,padding_mode = 'zeros'):
        super().__init__()
        #in_channels = in_channels
        bn = config['KP_HEADS']['BN']
        in_channels = in_channels if in_channels is not None else config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['KP_HEADS']['POS_ENCODING']

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.score = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.use_softmax = config['KP_HEADS']['USE_SOFTMAX']
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        # Allow more exploration with reinforce algorithm
        self.tmp_softmax = 100

        self.eps = nn.Parameter(torch.tensor(1e-16), requires_grad=False)
        self.offset_par1 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.offset_par2 = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.ones_kernel = nn.Parameter(torch.ones((1, 1, 3, 3)), requires_grad=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def remove_borders(self, score_map: torch.Tensor, borders: int):
        '''
        It removes the borders of the image to avoid detections on the corners
        '''
        shape = score_map.shape
        mask = torch.ones_like(score_map)

        mask[:, :, 0:borders, :] = 0
        mask[:, :, :, 0:borders] = 0
        mask[:, :, shape[2] - borders:shape[2], :] = 0
        mask[:, :, :, shape[3] - borders:shape[3]] = 0

        return mask * score_map

    def remove_brd_and_softmax(self, scores, borders):

        B = scores.shape[0]

        scores = scores - (scores.view(B, -1).mean(-1).view(B, 1, 1, 1) + self.eps).detach()
        exp_scores = torch.exp(scores / self.tmp_softmax)

        # remove borders
        exp_scores = self.remove_borders(exp_scores, borders=borders)

        # apply softmax
        sum_scores = exp_scores.sum(-1).sum(-1).view(B, 1, 1, 1)
        return exp_scores / (sum_scores + self.eps)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy scores
        scores = self.score(x)

        if self.use_softmax:
            scores = self.remove_brd_and_softmax(scores, 3)
        else:
            scores = self.remove_borders(self.sigmoid(scores), borders=3)

        return scores


class DeepResBlock_offset(torch.nn.Module):
    def __init__(self, config, in_channels=None,padding_mode = 'zeros'):
        super().__init__()


        bn = config['KP_HEADS']['BN']
        #in_channels = config['DINOV2']['CHANNEL_DIM']
        in_channels = in_channels if in_channels is not None else config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['KP_HEADS']['POS_ENCODING']
        self.sigmoid = torch.nn.Sigmoid()

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.xy_offset = nn.Conv2d(block_dims[3], 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy offsets
        xy_offsets = self.xy_offset(x)

        # Offset goes from 0 to 1
        xy_offsets = self.sigmoid(xy_offsets)

        return xy_offsets


class DeepResBlock_depth(torch.nn.Module):
    def __init__(self, config, in_channels=None,padding_mode = 'zeros'):
        super().__init__()

        bn = config['KP_HEADS']['BN']
        #in_channels = config['DINOV2']['CHANNEL_DIM']
        in_channels = in_channels if in_channels is not None else config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['KP_HEADS']['POS_ENCODING']

        self.use_depth_sigmoid = config['KP_HEADS']['USE_DEPTHSIGMOID']
        self.max_depth = config['KP_HEADS']['MAX_DEPTH']
        self.sigmoid = torch.nn.Sigmoid()

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.depth = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy depths
        # depths = torch.clip(self.depth(x), min=1e-3, max=500)
        if self.use_depth_sigmoid:
            depths = self.max_depth * self.sigmoid(self.depth(x))
        else:
            depths = self.depth(x)

        return depths


class DeepResBlock_desc(torch.nn.Module):
    def __init__(self, config, in_channels = None,padding_mode = 'zeros'):
        super().__init__()

        bn = config['KP_HEADS']['BN']
        last_dim = config['DSC_HEAD']['LAST_DIM']
        #in_channels = config['DINOV2']['CHANNEL_DIM']
        in_channels = in_channels if in_channels is not None else config['DINOV2']['CHANNEL_DIM']
        block_dims = config['KP_HEADS']['BLOCKS_DIM']
        add_posEnc = config['DSC_HEAD']['POS_ENCODING']
        self.norm_desc = config['DSC_HEAD']['NORM_DSC']

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], last_dim, stride=1, bn=bn, padding_mode=padding_mode)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)


    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x, relu=False)

        if self.norm_desc:
            x = desc_l2norm(x)

        return x

