# talk2dino_utils.py
import torch
import clip
import os
from lib.models.Oryon.t2d.model import ProjectionLayer
from torch import nn
import torch.nn.functional as F
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 添加上一级目录到系统路径
sys.path.append('..')
# DINOv2 backbone
#from dinov2.models import dinov2_vitb14
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',pretrained=True)
from lib.models.Oryon.tokenizer import SimpleTokenizer  # 修改为实际路径
class DINOV2BackboneWithAttn(nn.Module):
    """
    封装 DINOv2 backbone，自动提取特征和 attention maps
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = dinov2_vitb14.to(self.device)
        self.model.eval()

        # 保存注意力 maps
        self.attn_maps = []

        # 注册 hook 到每个 Attention 层
        for blk in self.model.blocks:
            blk.attn.register_forward_hook(self._make_attn_hook())

    def _make_attn_hook(self):
        def hook(module, input, output):
            """
            output: Attention 层输出
            新版 DINOv2 输出通常是 attn 矩阵或 tuple (x_out, attn)
            """
            # 如果是 tuple，则取第二个元素作为 attn
            attn = output[1] if isinstance(output, tuple) else output
            self.attn_maps.append(attn.detach())
        return hook

   #@torch.no_grad()
    @torch.no_grad()
    @torch.no_grad()
    def forward_features_and_attn(self, img: torch.Tensor):
        """
        完全防止 hub DINOv2 自动 resize 的版本
        img: [B,3,H,W]，float32，range [0,1] 或 [-1,1]
        returns:
            feat_map: [B, C, Hf, Wf]
            attn_maps_proc: list of [B, num_heads, Hf, Wf]
        """
        self.attn_maps = []  # 清空之前的 attn

        B_img, C_img, H_img, W_img = img.shape

        # ------------------------
        # 1. patch embedding
        # ------------------------
        feat_out = self.model.forward_features(img)  # 不会自动 resize

        # ------------------------
        # 2. feat_map 处理
        # ------------------------
        if isinstance(feat_out, dict):
            if "x_norm_patchtokens" in feat_out:
                feat_map = feat_out["x_norm_patchtokens"]  # [B, N, C]
            elif "last_hidden_state" in feat_out:
                feat_map = feat_out["last_hidden_state"]
            else:
                raise ValueError(f"Unexpected keys in model output: {feat_out.keys()}")
        else:
            feat_map = feat_out

        # ------------------------
        # 3. reshape feat_map
        # ------------------------
        if feat_map.ndim == 4:  # 已经 [B,C,Hf,Wf]
            Bf, Cf, Hf, Wf = feat_map.shape
        elif feat_map.ndim == 3:  # [B, N, C] token
            B, N, C = feat_map.shape

            # 自动判断 CLS token
            patch_size = getattr(getattr(self.model, 'patch_embed', None), 'patch_size', 14)
            patch_size = patch_size if isinstance(patch_size, int) else int(patch_size[0])

            # Hf/Wf 通过输入图像尺寸 + patch_size 计算
            Hf = H_img // patch_size
            Wf = W_img // patch_size
            expected = Hf * Wf

            if N == expected + 1:  # CLS token
                feat_map = feat_map[:, 1:, :]
            elif N != expected:
                # 如果 token 数不匹配，直接按 sqrt(N) 强制 reshape
                Hf = int(N ** 0.5)
                Wf = N // Hf
                if Hf * Wf != N:
                    raise ValueError(f"[DINO] Cannot infer Hf/Wf from N={N}, input H,W=({H_img},{W_img})")
                feat_map = feat_map

            # reshape
            feat_map = feat_map.permute(0, 2, 1).reshape(B, C, Hf, Wf)
            Bf, Cf, Hf, Wf = feat_map.shape
        else:
            raise ValueError(f"[DINO] Unexpected feat_map shape: {feat_map.shape}")

        # ------------------------
        # 4. attention maps处理
        # ------------------------
        attn_maps_proc = []
        for attn in self.attn_maps:
            if attn.ndim == 3:  # [B,N,N] -> 1 head
                attn = attn.unsqueeze(1)

            B_attn, H_heads, N1, N2 = attn.shape
            assert B_attn == Bf, f"attn batch {B_attn} != feat batch {Bf}"

            # 去掉 cls token 列
            if N2 == Hf * Wf + 1:
                patch_attn = attn[:, :, 0, 1:]
            elif N2 == Hf * Wf:
                patch_attn = attn[:, :, 0, :]
            elif N2 > Hf * Wf:
                patch_attn = attn[:, :, 0, :Hf * Wf]
            else:
                raise ValueError(f"[DINO] attn tokens {N2} != expected {Hf * Wf} (Hf={Hf},Wf={Wf})")

            patch_attn = patch_attn.view(Bf, H_heads, Hf, Wf)
            attn_maps_proc.append(patch_attn)

        return feat_map, attn_maps_proc


class Talk2DINO:
    """
    Talk2DINO pipeline:
    1. CLIP文本 -> DINOv2空间投影
    2. DINOv2特征 + attention map 加权 -> 视觉嵌入
    3. 返回视觉嵌入和投影后的文本特征
    """
    def __init__(self, proj_name='vitb_mlp_infonce', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Text projection
        self.config_path = os.path.join("config", f"{proj_name}.yaml")
        self.weights_path = os.path.join("weights", f"{proj_name}.pth")
        self.proj = ProjectionLayer.from_config(self.config_path)
        self.proj.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.proj.to(self.device)
        self.proj.eval()

        # CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        self.tokenizer = SimpleTokenizer()  # 替代 clip.tokenize

        # DINOv2 backbone + attention
        self.dino = DINOV2BackboneWithAttn(self.device)

    @torch.no_grad()
    def project_text(self, texts):
        """
        texts: list of str 或嵌套列表
        returns: projected text features [B, C_dino]
        """
        # 如果是嵌套列表，先 flatten
        flattened_texts = []
        for t in texts:
            if isinstance(t, list):
                flattened_texts.extend([str(x) for x in t])
            else:
                flattened_texts.append(str(t))

        # 使用自定义 tokenizer
        tokens = self.tokenizer(flattened_texts)  # [B, context_length]

        # 确保 tokens 在与 clip_model 相同的设备上
        device = next(self.clip_model.parameters()).device
        tokens = tokens.to(device)

        # 编码文本
        clip_feat = self.clip_model.encode_text(tokens)

        # 投影
        proj_feat = self.proj.project_clip_txt(clip_feat)
        return proj_feat

    @torch.no_grad()
    def extract_visual_embeddings(self, img: torch.Tensor, attn_maps: list):
        """
        img: [B,3,H,W] normalized in [0,1]
        attn_maps: list of attention maps from DINOv2 [B, num_heads, Hf, Wf]
        returns: visual embeddings [B, total_N, C_dino]
        """
        # 1. DINOv2 dense feature map
        feat_map, _ = self.dino.forward_features_and_attn(img)  # [B, C, Hf, Wf]
        B, C, Hf, Wf = feat_map.shape

        embeddings = []
        for attn in attn_maps:
            # attn: [B, heads, Hf, Wf]
            B_attn, H, H_attn, W_attn = attn.shape
            assert B_attn == B
            attn_flat = attn.view(B, H, -1)  # [B, heads, Hf*Wf]
            feat_flat = feat_map.view(B, C, -1).permute(0, 2, 1)  # [B,Hf*Wf,C]
            emb = torch.bmm(attn_flat, feat_flat)  # [B, heads, C]
            emb = emb / (attn_flat.sum(-1, keepdim=True) + 1e-6)  # 加权平均
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=1)  # [B, total_heads, C]
        return embeddings
