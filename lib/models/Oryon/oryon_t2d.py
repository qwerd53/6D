# oryon.py (modified)
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.transforms.functional import normalize
from torch.nn.functional import interpolate
from torchvision.models.feature_extraction import create_feature_extractor

from lib.models.Oryon.vlm import get_vlm
from lib.models.Oryon.fusion import get_fusion_module
from lib.models.Oryon.decoder import get_decoder
from lib.models.Oryon.talk2dino_utils import Talk2DINO


def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.Upsample)):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Oryon(nn.Module):
    """
    Oryon module (fixed):
    - Delay-create adapter conv/linears until we know D_vlm / C_dino
    - Ensure device consistency when calling convs with DINO outputs
    - Do not cat raw prompt templates into spatial (avoids batch/template mismatch)
    """
    def __init__(self, args, device: str):
        super().__init__()
        self.args = args.model
        self.device = device

        # core modules
        self.vlm = get_vlm(self.args, self.device)
        self.guidance_backbone = self.init_guidance_backbone(self.device)
        self.fusion = get_fusion_module(self.args, self.device)
        self.decoder = get_decoder(self.args, self.device)

        # Talk2DINO wrapper（包含 DINOv2 + CLIP projection）
        self.t2d = Talk2DINO(proj_name='vitb_mlp_infonce', device=self.device)

        # lazy adapters placeholders (created in _init_adapters_lazy)
        self._adapters_initialized = False
        self.t2d_to_vlm = None           # Linear C_dino -> D_vlm
        self.promptproj_to_vlm = None    # Linear C_dino -> D_vlm (IMPORTANT)
        self.vlm_to_t2d = None
        self.t2d_conv_map = None         # Conv2d C_dino -> D_vlm

        # small convs we will create lazily to match D_vlm
        self.global2spatial = None

        self.init_all()

    def init_guidance_backbone(self, device):
        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        for param in swin.parameters():
            param.requires_grad = False
        return_nodes = {
            'features.1.1.add_1': 'guidance3',  # [128,96,96]
            'features.2.reduction': 'guidance2', # [256,48,48]
            'features.4.reduction': 'guidance1' #  [512,24,24]
        }
        backbone = create_feature_extractor(swin, return_nodes=return_nodes)
        backbone.eval().to(device)
        return backbone

    def get_guidance_embeds(self, img: torch.Tensor) -> List[torch.Tensor]:
        img = interpolate(img.clone(), size=(384,384), mode='bicubic', align_corners=True)
        img = normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        outs = self.guidance_backbone(img)
        guid1 = outs['guidance1'].transpose(2,3).transpose(1,2)
        guid2 = outs['guidance2'].transpose(2,3).transpose(1,2)
        guid3 = outs['guidance3'].transpose(2,3).transpose(1,2)
        return [guid1, guid2, guid3]

    def init_all(self):
        # initialize fusion/decoder weights
        self.fusion.clip_conv.apply(weights_init_kaiming)
        self.fusion.apply(weights_init_kaiming)
        self.decoder.apply(weights_init_kaiming)

    def _init_adapters_lazy(self, visual_a, prompt_emb_vlm, text_proj, t2d_feat_map):
        """
        Create adapters once we know D_vlm and C_dino.
        visual_a: [B, D_vlm, H, W] or [B, D_vlm, ...] -> use last dim
        prompt_emb_vlm: [B, D_prompt]
        text_proj: [B, C_dino]  (after template pooling)
        t2d_feat_map: any sample feat (to infer C_dino) -- not strictly required
        """
        if self._adapters_initialized:
            return

        # D_vlm: dimension used by VLM visual/global features
        # visual_a might be [B, C, H, W] or [B, C] if mean; handle robustly:
        if visual_a.ndim >= 2:
            D_vlm = visual_a.shape[-1] if visual_a.ndim == 2 else visual_a.shape[1]
        else:
            raise RuntimeError("visual_a has unexpected ndim")

        # C_dino: feature dim of text_proj (last dim)
        C_dino = text_proj.shape[-1]

        # Create adapters on self.device
        self.t2d_to_vlm = nn.Linear(C_dino, D_vlm).to(self.device)
        # IMPORTANT: make promptproj_to_vlm map into D_vlm so global2spatial will accept it
        self.promptproj_to_vlm = nn.Linear(C_dino, D_vlm).to(self.device)
        self.vlm_to_t2d = nn.Linear(D_vlm, C_dino).to(self.device)
        self.t2d_conv_map = nn.Conv2d(C_dino, D_vlm, kernel_size=1, bias=True).to(self.device)
        self.global2spatial = nn.Conv2d(D_vlm, D_vlm, kernel_size=1, bias=True).to(self.device)

        # init weights
        nn.init.normal_(self.t2d_to_vlm.weight, 0, 0.01)
        nn.init.constant_(self.t2d_to_vlm.bias, 0)
        nn.init.normal_(self.promptproj_to_vlm.weight, 0, 0.01)
        nn.init.constant_(self.promptproj_to_vlm.bias, 0)
        nn.init.normal_(self.vlm_to_t2d.weight, 0, 0.01)
        nn.init.constant_(self.vlm_to_t2d.bias, 0)
        nn.init.kaiming_normal_(self.t2d_conv_map.weight, a=0, mode='fan_out')
        if self.t2d_conv_map.bias is not None:
            nn.init.constant_(self.t2d_conv_map.bias, 0.0)
        nn.init.kaiming_normal_(self.global2spatial.weight, a=0, mode='fan_out')
        if self.global2spatial.bias is not None:
            nn.init.constant_(self.global2spatial.bias, 0.0)

        self._adapters_initialized = True

    def forward(self, xs: dict, return_alignment_for_loss: bool = False):
        device = self.device
        img_a, img_q = xs['image0'].to(device).float(), xs['image1'].to(device).float()

        def make_dino(img):
            return F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

        img_a_dino, img_q_dino = make_dino(img_a), make_dino(img_q)

        # VLM visual embeddings (on self.device)
        visual_a = self.vlm.encode_image(img_a)   # [B, D_vlm, Hv, Wv] or similar
        visual_q = self.vlm.encode_image(img_q)
        visual_a_global = visual_a.mean(dim=[2, 3])
        visual_q_global = visual_q.mean(dim=[2, 3])
        prompt_emb_vlm = self.vlm.encode_prompt(xs['prompt']).unsqueeze(1)  # [B,1,D_prompt]

        # guidance embeddings (Swin) on self.device
        guid_a, guid_q = self.get_guidance_embeds(img_a), self.get_guidance_embeds(img_q)

        # text projection from Talk2DINO (may be on different device) -> [B * num_prompts, C_dino]
        text_proj = self.t2d.project_text(xs['prompt'])

        # If multiple templates per image, pool them (mean)
        # compute num_prompts robustly
        B = img_a.shape[0]
        if text_proj.shape[0] % B == 0:
            num_prompts = text_proj.shape[0] // B
            if num_prompts > 1:
                text_proj = text_proj.view(B, num_prompts, -1).mean(dim=1)  # [B, C_dino]
            else:
                text_proj = text_proj.view(B, -1)
        else:
            # fallback: try to slice/reshape, but ensure length B
            text_proj = text_proj.to(self.device)[:B, :]

        # move pooled text_proj to self.device (will be used by adapter linears on self.device)
        text_proj = text_proj.to(self.device)

        # DINO backbone -> ensure input on dino device
        dino_device = next(self.t2d.dino.model.parameters()).device
        img_a_dino = img_a_dino.to(dino_device)
        img_q_dino = img_q_dino.to(dino_device)

        feat_a, attn_a = self.t2d.dino.forward_features_and_attn(img_a_dino)  # feat_a on dino_device
        feat_q, attn_q = self.t2d.dino.forward_features_and_attn(img_q_dino)

        # compute t2d globals (on dino_device)
        def t2d_global(feat_map, attn_list):
            Bf, Cf, Hf, Wf = feat_map.shape
            if attn_list:
                all_embs = []
                feat_flat = feat_map.view(Bf, Cf, -1).permute(0, 2, 1)
                for attn in attn_list:
                    attn_flat = F.softmax(attn.view(Bf, attn.shape[1], -1), dim=-1)
                    all_embs.append(torch.bmm(attn_flat, feat_flat))
                return torch.cat(all_embs, dim=1).mean(dim=1)
            else:
                return feat_map.mean(dim=[2, 3])

        t2d_a_global = t2d_global(feat_a, attn_a)  # on dino_device
        t2d_q_global = t2d_global(feat_q, attn_q)

        # Initialize adapters now that we have visual dims and text dims
        # pass visual_a_global (on self.device) and text_proj (on self.device)
        self._init_adapters_lazy(visual_a_global, prompt_emb_vlm, text_proj, feat_a)

        # Move t2d_global to self.device for mapping/gating operations
        t2d_a_global = t2d_a_global.to(self.device)
        t2d_q_global = t2d_q_global.to(self.device)

        # Move DINO dense feats to self.device BEFORE applying adapter conv (self.t2d_conv_map lives on self.device)
        feat_a = feat_a.to(self.device)
        feat_q = feat_q.to(self.device)

        # DINO -> VLM spatial (apply conv on self.device)
        t2d_a_spatial = F.interpolate(self.t2d_conv_map(feat_a),
                                      size=guid_a[0].shape[2:],
                                      mode='bilinear',
                                      align_corners=False)
        t2d_q_spatial = F.interpolate(self.t2d_conv_map(feat_q),
                                      size=guid_q[0].shape[2:],
                                      mode='bilinear',
                                      align_corners=False)

        # prompt as modulation: map pooled text_proj (C_dino) -> D_vlm then spatialize with global2spatial
        # text_proj is on self.device
        prompt_global = self.promptproj_to_vlm(text_proj)  # [B, D_vlm]
        # to spatial [B, D_vlm, 1, 1] then pass through conv to keep representation consistent
        prompt_global_spatial = self.global2spatial(prompt_global.unsqueeze(-1).unsqueeze(-1))  # [B, D_vlm, 1,1]
        # will broadcast when adding to [B, D_vlm, H, W]

        # guidance combine: do NOT concat template-expanded prompt into spatial (avoids batch mismatch)
        guid_a_mod = [torch.cat([guid_a[0], t2d_a_spatial], dim=1), guid_a[1], guid_a[2]]
        guid_q_mod = [torch.cat([guid_q[0], t2d_q_spatial], dim=1), guid_q[1], guid_q[2]]

        # gating: ensure both sides in same device (they are on self.device)
        alpha = getattr(self.args, 't2d_alpha', 10.0)
        gate_a = torch.sigmoid(alpha * F.cosine_similarity(
            F.normalize(t2d_a_global, dim=-1),
            F.normalize(text_proj, dim=-1), dim=-1)).unsqueeze(-1).unsqueeze(-1)
        gate_q = torch.sigmoid(alpha * F.cosine_similarity(
            F.normalize(t2d_q_global, dim=-1),
            F.normalize(text_proj, dim=-1), dim=-1)).unsqueeze(-1).unsqueeze(-1)

        # t2d_global_spatial: map t2d_global -> D_vlm -> spatial, multiplied by gate
        t2d_a_global_spatial = self.global2spatial(self.t2d_to_vlm(t2d_a_global).unsqueeze(-1).unsqueeze(-1)) * gate_a
        t2d_q_global_spatial = self.global2spatial(self.t2d_to_vlm(t2d_q_global).unsqueeze(-1).unsqueeze(-1)) * gate_q

        # visual upsampling to same spatial size (Hg,Wg)
        Hg, Wg = visual_a.shape[2], visual_a.shape[3]
        visual_a_up = F.interpolate(visual_a, size=(Hg, Wg), mode='bilinear', align_corners=False)
        visual_q_up = F.interpolate(visual_q, size=(Hg, Wg), mode='bilinear', align_corners=False)

        # fused: add prompt_global_spatial and t2d_global_spatial as modulation terms (they broadcast)
        fused_a = torch.cat([visual_a_up, t2d_a_spatial + t2d_a_global_spatial + prompt_global_spatial], dim=1)
        fused_q = torch.cat([visual_q_up, t2d_q_spatial + t2d_q_global_spatial + prompt_global_spatial], dim=1)

        feats_a = self.fusion(fused_a, prompt_emb_vlm, guid_a_mod)
        feats_q = self.fusion(fused_q, prompt_emb_vlm, guid_q_mod)
        mask_a, featmap_a_out = self.decoder(feats_a, guid_a_mod)
        mask_q, featmap_q_out = self.decoder(feats_q, guid_q_mod)

        out = {
            'featmap_a': featmap_a_out,
            'featmap_q': featmap_q_out,
            'mask_a': mask_a,
            'mask_q': mask_q,
            't2d_visual_a': t2d_a_global,
            't2d_visual_q': t2d_q_global,
            'text_proj': text_proj,
        }

        if return_alignment_for_loss:
            out['sim_imgtxt_a'] = torch.matmul(F.normalize(t2d_a_global, dim=-1),
                                               F.normalize(text_proj, dim=-1).t())
            out['sim_imgtxt_q'] = torch.matmul(F.normalize(t2d_q_global, dim=-1),
                                               F.normalize(text_proj, dim=-1).t())

        return out
