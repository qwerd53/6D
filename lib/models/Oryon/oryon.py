import torch
import numpy as np
from typing import OrderedDict, Tuple, List
from omegaconf import DictConfig
from torch import Tensor
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.transforms.functional import normalize
from torch.nn.functional import interpolate
from torchvision.models.feature_extraction import create_feature_extractor
from lib.models.Oryon.vlm import get_vlm
from lib.models.Oryon.fusion import get_fusion_module
from lib.models.Oryon.decoder import get_decoder
from torch import nn


def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Upsample) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,
                                                                                                                 nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Oryon(torch.nn.Module):
    def __init__(self, args: DictConfig, device: str):

        super().__init__()

        self.args = args.model
        self.device = device
        self.vlm = get_vlm(self.args, self.device)
        self.guidance_backbone = self.init_guidance_backbone(self.device)
        self.fusion = get_fusion_module(self.args, self.device)
        self.decoder = get_decoder(self.args, self.device)

        # 添加通道适配层
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(32, 512, kernel_size=1).to(device),
            # nn.AvgPool2d(kernel_size=14, stride=14)  # 空间下采样
        ).to(device)

        self.init_all()

        # #空间下采样和进行对齐
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(32, 1024, kernel_size=1),  # 通道对齐
        #     nn.AvgPool2d(kernel_size=14, stride=14)  # 空间下采样
        # )

    def get_trainable_parameters(self) -> list:

        param_list = []
        param_list.extend(self.fusion.parameters())
        param_list.extend(self.decoder.parameters())

        return param_list

    def init_guidance_backbone(self, device):
        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        for param in swin.parameters():
            param.requires_grad = False
        return_nodes = {
            'features.1.1.add_1': 'guidance3',  # [128,96,96]
            'features.2.reduction': 'guidance2',  # [256,48,48]
            'features.4.reduction': 'guidance1'  # [512,24,24]
        }

        backbone = create_feature_extractor(swin, return_nodes=return_nodes)
        backbone.eval()
        backbone = backbone.to(device)
        return backbone

    def get_guidance_embeds(self, img_: Tensor) -> List[Tensor]:
        '''
        Return guidance embeddings as in CATSeg, from Swin_b transformer
        normalization from https://pytorch.org/vision/main/models/generated/torchvision.models.swin_b.html
        '''
        img = img_.clone()

        img = interpolate(img, size=(384, 384), mode='bicubic', align_corners=True)
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        outs = self.guidance_backbone(img)

        guid1 = outs['guidance1'].transpose(2, 3).transpose(1, 2)
        guid2 = outs['guidance2'].transpose(2, 3).transpose(1, 2)
        guid3 = outs['guidance3'].transpose(2, 3).transpose(1, 2)

        return [guid1, guid2, guid3]

    def train(self, mode=True):

        self.training = mode
        self.vlm.train(mode)
        self.fusion.train(mode)
        self.decoder.train(mode)

        return self

    def eval(self):

        self.train(False)

    def get_image_input(self, xs: dict) -> Tuple[dict, dict]:

        # create input with RGB channels
        input_a = {'rgb': xs['anchor']['rgb'].to(self.device)}
        input_q = {'rgb': xs['query']['rgb'].to(self.device)}

        return (input_a, input_q)

    def init_all(self):
        self.fusion.clip_conv.apply(weights_init_kaiming)
        # print('self.args.use_catseg_ckpt:',self.args.use_catseg_ckpt)
        if self.args.use_catseg_ckpt:
            # print("Loading CATSeg checkpoint")
            ckpt = torch.load('pretrained_models/catseg.pth', map_location=self.device)
            # set checkpoint names
            new_state_dict = dict()
            # this is necessary because of the refactoring we carried out
            old_fusion_key = 'sem_seg_head.predictor.transformer'
            new_fusion_key = 'fusion'
            old_dec_key = 'fusion.decoder'
            new_dec_key = 'decoder.decoder'

            # changing prefix of fusion and decoder keys
            for k, v in ckpt['model'].items():
                if k.startswith(old_fusion_key):
                    new_k = k.replace(old_fusion_key, new_fusion_key)
                    if new_k.startswith(old_dec_key):
                        new_k = new_k.replace(old_dec_key, new_dec_key)
                    if new_k.startswith('fusion.head'):
                        new_k = new_k.replace('fusion.head', 'decoder.head')
                    new_state_dict[new_k] = v

            # if using CLIP, we are also loading CATSeg's finetuned CLIP
            if self.args.image_encoder.vlm == 'clip':
                old_clip_key = 'sem_seg_head.predictor.clip_model'
                new_clip_key = 'vlm.clip_model'

                for k, v in ckpt['model'].items():
                    if k.startswith(old_clip_key):
                        new_k = k.replace(old_clip_key, new_clip_key)
                        new_state_dict[new_k] = v

            inco_keys = self.load_state_dict(new_state_dict, strict=False)
            # print(inco_keys)

        else:
            print("Training from scratch")
            self.fusion.apply(weights_init_kaiming)
            self.decoder.apply(weights_init_kaiming)

    def forward(self, xs: dict):
        xs['image0'] = xs['image0'].float()
        xs['image1'] = xs['image1'].float()

        # extracting CLIP features
        # visual_a = self.vlm.encode_image(xs['anchor']['rgb'])
        # visual_q = self.vlm.encode_image(xs['query']['rgb'])
        # prompt_emb = self.vlm.encode_prompt(xs['prompt'])
        #
        # guid_a = self.get_guidance_embeds(xs['anchor']['rgb'])
        # guid_q = self.get_guidance_embeds(xs['query']['rgb'])

        # extracting CLIP features # collate_custom

        visual_a = self.vlm.encode_image(xs['image0'])  # was: xs['anchor']['rgb']
        visual_q = self.vlm.encode_image(xs['image1'])  # was: xs['query']['rgb']
        # visual_a = self.vlm.encode_image(xs['image0'].float())
        # visual_q = self.vlm.encode_image(xs['image1'].float())

        prompt_emb = self.vlm.encode_prompt(xs['prompt'])

        guid_a = self.get_guidance_embeds(xs['image0'])  # was: xs['anchor']['rgb']
        guid_q = self.get_guidance_embeds(xs['image1'])  # was: xs['query']['rgb']

        # get encoded feature maps [D,N,N]
        prompt_emb = prompt_emb.unsqueeze(1)
        feats_a = self.fusion.forward(visual_a, prompt_emb, guid_a)
        feats_q = self.fusion.forward(visual_q, prompt_emb, guid_q)
        # print("feats_a",feats_a.shape)

        # docoder
        mask_a, featmap_a = self.decoder.forward(feats_a, guid_a)
        mask_q, featmap_q = self.decoder.forward(feats_q, guid_q)
        # print("mask.shape:",mask_a.shape)
        # print(featmap_a.shape)
        # print("mask_q:",mask_a)
        # # 调整通道数和尺寸
        # featmap_a = self.channel_adapter(featmap_a)  # [B,512,H/14,W/14]
        # featmap_q = self.channel_adapter(featmap_q)  # [B,512,H/14,W/14]
        # assert featmap_a.shape[2:] == self.args.image_encoder.img_size
        # print("mask_a:",mask_a)
        # print("mask_q:", mask_q)
        return {
            # 'featmap_a' : featmap_a,
            # 'featmap_q' : featmap_q,
            'mask_a': mask_a,
            'mask_q': mask_q
        }

# talk2dino
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 需要引入 Talk2DINO 封装（请确认路径）
# from talk2dino_utils import Talk2DINO

# class Oryon(torch.nn.Module):
#     def __init__(self, args, device: str):
#         super().__init__()
#         self.args = args.model
#         self.device = device

#         # 原有模块
#         self.vlm = get_vlm(self.args, self.device)
#         self.guidance_backbone = self.init_guidance_backbone(self.device)
#         self.fusion = get_fusion_module(self.args, self.device)
#         self.decoder = get_decoder(self.args, self.device)

#         # Talk2DINO wrapper（包含 DINOv2 + CLIP projection）
#         self.t2d = Talk2DINO(proj_name='vitb_mlp_infonce', device=self.device)

#         # lazy adapters: create on first forward when dims are known
#         self._adapters_initialized = False
#         self.t2d_to_vlm = None        # Linear(C_dino -> D_vlm) to adapt dino features into vlm visual dim
#         self.promptproj_to_vlm = None # Linear(C_dino -> D_prompt) to adapt text_proj into prompt dim (if needed)
#         self.vlm_to_t2d = None        # optional: map vlm visual -> dino dim
#         # channel adapter if needed for spatial feature maps (kept from original)
#         self.channel_adapter = nn.Sequential(
#             nn.Conv2d(32, 512, kernel_size=1).to(device),
#         ).to(device)

#         self.init_all()

#     def _init_adapters_lazy(self, visual_a, prompt_emb_vlm, text_proj, t2d_visual_mean):
#         """
#         Create linear adapters based on the actual dims observed in a forward pass.
#         visual_a: [B, D_vlm]
#         prompt_emb_vlm: [B, 1, D_prompt]
#         text_proj: [B, C_dino]
#         t2d_visual_mean: [B, C_dino]
#         """
#         if self._adapters_initialized:
#             return

#         D_vlm = visual_a.shape[-1]
#         D_prompt = prompt_emb_vlm.shape[-1]
#         C_dino = text_proj.shape[-1]

#         # map DINO -> VLM visual dim for concatenation
#         self.t2d_to_vlm = nn.Linear(C_dino, D_vlm).to(self.device)

#         # map DINO text proj -> prompt dim (so we can concat to existing prompt embedding)
#         self.promptproj_to_vlm = nn.Linear(C_dino, D_prompt).to(self.device)

#         # optional: vlm->dino mapping if you want to compute alignment in dino space
#         self.vlm_to_t2d = nn.Linear(D_vlm, C_dino).to(self.device)

#         # initialize weights (kaiming or normal)
#         nn.init.normal_(self.t2d_to_vlm.weight, 0, 0.01)
#         nn.init.constant_(self.t2d_to_vlm.bias, 0)
#         nn.init.normal_(self.promptproj_to_vlm.weight, 0, 0.01)
#         nn.init.constant_(self.promptproj_to_vlm.bias, 0)
#         nn.init.normal_(self.vlm_to_t2d.weight, 0, 0.01)
#         nn.init.constant_(self.vlm_to_t2d.bias, 0)

#         self._adapters_initialized = True

#     def forward(self, xs: dict, return_alignment_for_loss: bool = False):
#         """
#         Full forward:
#          - xs['image0'], xs['image1'] : [B,3,H,W]
#          - xs['prompt'] : list[str] length B (compatible with self.vlm.encode_prompt and self.t2d.project_text)
#         """
#         device = self.device

#         # ensure float and device
#         img_a = xs['image0'].to(device).float()
#         img_q = xs['image1'].to(device).float()

#         # -----------------------
#         # 1. Original VLM features (Oryon)
#         # -----------------------
#         visual_a = self.vlm.encode_image(img_a)   # [B, D_vlm]
#         visual_q = self.vlm.encode_image(img_q)   # [B, D_vlm]
#         prompt_emb_vlm = self.vlm.encode_prompt(xs['prompt']).to(device)  # [B, D_prompt] or [B, D]
#         prompt_emb_vlm = prompt_emb_vlm.unsqueeze(1)  # [B,1,D_prompt] expected by fusion

#         # guidance features from Swin backbone
#         guid_a = self.get_guidance_embeds(img_a)
#         guid_q = self.get_guidance_embeds(img_q)

#         # -----------------------
#         # 2. Talk2DINO: project text into DINO space (ψ)
#         # -----------------------
#         # text_proj: [B, C_dino]
#         text_proj = self.t2d.project_text(xs['prompt'])  # already on correct device

#         # -----------------------
#         # 3. Talk2DINO: DINOv2 dense features + attention maps
#         #    returns feat_map [B, C_dino, Hf, Wf] and attn_maps list: [B, heads, Hf, Wf]
#         # -----------------------
#         feat_map_a, attn_maps_a = self.t2d.dino.forward_features_and_attn(img_a)
#         feat_map_q, attn_maps_q = self.t2d.dino.forward_features_and_attn(img_q)

#         # -----------------------
#         # 4. compute visual embeddings from feat_map + attn_maps
#         #    produce t2d_visual_a: [B, N_a, C_dino]
#         # -----------------------
#         def visual_embeds_from_feat(feat_map, attn_maps_list):
#             Bf, Cf, Hf, Wf = feat_map.shape
#             feat_flat = feat_map.view(Bf, Cf, -1).permute(0, 2, 1)  # [B, HW, C]
#             all_embs = []
#             for attn in attn_maps_list:
#                 # attn: [B, heads, Hf, Wf]
#                 B_attn, heads, Ha, Wa = attn.shape
#                 assert B_attn == Bf
#                 attn_flat = attn.view(Bf, heads, -1)  # [B, heads, HW]
#                 attn_norm = F.softmax(attn_flat, dim=-1)  # normalize over spatial dim
#                 emb = torch.bmm(attn_norm, feat_flat)  # [B, heads, C]
#                 all_embs.append(emb)
#             if len(all_embs) == 0:
#                 # fallback: global pooled feature
#                 return feat_map.view(Bf, Cf, -1).mean(-1).unsqueeze(1)  # [B,1,C]
#             return torch.cat(all_embs, dim=1)  # [B, total_heads, C]

#         t2d_visual_a = visual_embeds_from_feat(feat_map_a, attn_maps_a)  # [B, Na, C_dino]
#         t2d_visual_q = visual_embeds_from_feat(feat_map_q, attn_maps_q)  # [B, Nq, C_dino]

#         # -----------------------
#         # 5. compute alignment (cosine) between visual embeddings and projected text
#         # -----------------------
#         # normalize
#         t2d_visual_a_norm = F.normalize(t2d_visual_a, dim=-1)  # [B,Na,C]
#         t2d_visual_q_norm = F.normalize(t2d_visual_q, dim=-1)
#         text_proj_norm = F.normalize(text_proj, dim=-1).unsqueeze(1)  # [B,1,C]

#         # similarities: matmul over last dim -> [B, Na]
#         sim_a = torch.matmul(t2d_visual_a_norm, text_proj_norm.permute(0,2,1)).squeeze(-1)
#         sim_q = torch.matmul(t2d_visual_q_norm, text_proj_norm.permute(0,2,1)).squeeze(-1)

#         # global alignment scores (max over heads)
#         align_a_vals, _ = sim_a.max(dim=1)  # [B]
#         align_q_vals, _ = sim_q.max(dim=1)  # [B]

#         # gating scalar from alignment (optional scale)
#         alpha = getattr(self.args, 't2d_alpha', 10.0)
#         gate_a = torch.sigmoid(alpha * (align_a_vals.unsqueeze(1)))  # [B,1]
#         gate_q = torch.sigmoid(alpha * (align_q_vals.unsqueeze(1)))

#         # -----------------------
#         # 6. pooling & adapters (lazy init)
#         # -----------------------
#         t2d_visual_a_mean = t2d_visual_a.mean(dim=1)  # [B, C_dino]
#         t2d_visual_q_mean = t2d_visual_q.mean(dim=1)  # [B, C_dino]

#         # initialize adapters on first pass (based on actual dims)
#         self._init_adapters_lazy(visual_a, prompt_emb_vlm, text_proj, t2d_visual_a_mean)

#         # project DINO visual mean into VLM visual dim and weight by gate
#         t2d_to_vlm = self.t2d_to_vlm  # Linear(C_dino -> D_vlm)
#         t2d_visual_a_mapped = t2d_to_vlm(t2d_visual_a_mean) * gate_a  # [B, D_vlm]
#         t2d_visual_q_mapped = t2d_to_vlm(t2d_visual_q_mean) * gate_q

#         # fuse clipped visual features and dino features
#         fused_visual_a = torch.cat([visual_a, t2d_visual_a_mapped], dim=-1)  # [B, D_vlm + D_vlm]
#         fused_visual_q = torch.cat([visual_q, t2d_visual_q_mapped], dim=-1)

#         # also fuse text projection into prompt embedding (map text_proj -> D_prompt)
#         promptproj_map = self.promptproj_to_vlm(text_proj)  # [B, D_prompt]
#         promptproj_map = promptproj_map.unsqueeze(1)  # [B,1,D_prompt]
#         fused_prompt = torch.cat([prompt_emb_vlm, promptproj_map], dim=-1)  # [B,1,D_prompt + D_prompt]

#         # -----------------------
#         # 7. pass to fusion and decoder
#         # -----------------------
#         feats_a = self.fusion.forward(fused_visual_a, fused_prompt, guid_a)
#         feats_q = self.fusion.forward(fused_visual_q, fused_prompt, guid_q)

#         mask_a, featmap_a = self.decoder.forward(feats_a, guid_a)
#         mask_q, featmap_q = self.decoder.forward(feats_q, guid_q)

#         # -----------------------
#         # 8. prepare outputs (and optionally alignment for loss)
#         # -----------------------
#         out = {
#             'featmap_a': featmap_a,
#             'featmap_q': featmap_q,
#             'mask_a': mask_a,
#             'mask_q': mask_q,
#             't2d_visual_a': t2d_visual_a,
#             't2d_visual_q': t2d_visual_q,
#             'text_proj': text_proj,
#             'align_a_vals': align_a_vals,
#             'align_q_vals': align_q_vals,
#             'sim_a': sim_a,
#             'sim_q': sim_q
#         }

#         if return_alignment_for_loss:
#             # compute global image-text similarity matrix for InfoNCE
#             global_vis_a = F.normalize(t2d_visual_a.mean(dim=1), dim=-1)  # [B, C]
#             global_vis_q = F.normalize(t2d_visual_q.mean(dim=1), dim=-1)
#             all_texts = F.normalize(text_proj, dim=-1)  # [B, C]
#             sim_imgtxt_a = torch.matmul(global_vis_a, all_texts.t())  # [B,B]
#             sim_imgtxt_q = torch.matmul(global_vis_q, all_texts.t())  # [B,B]
#             out['sim_imgtxt_a'] = sim_imgtxt_a
#             out['sim_imgtxt_q'] = sim_imgtxt_q

#         return out
