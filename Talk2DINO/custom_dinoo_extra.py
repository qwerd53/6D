import argparse
import clip
import json
import math
import os
import requests
import webdataset as wds
import tarfile
import timm
import torch
import torchvision.transforms as T
import sys
import yaml
from pathlib import Path

from io import BytesIO
from src.hooks import get_self_attention, process_self_attention, get_second_last_out, get_vit_out, get_dinov1_patches, \
    feats
from src.webdatasets_util import cc2coco_format, create_webdataset_tar, read_coco_format_wds
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken

# 导入自定义数据模块
sys.path.append('/data/WDY/mickey-main/lib/datasets')
sys.path.append('/data/WDY/mickey-main')
from datamodules import DataModule, custom_collate

# ============ 新增从训练脚本移植的部分 ============
from omegaconf import DictConfig


def get_dataset_args_dict(dataset_name: str, root_path: str, seed: int = 42):
    assert dataset_name in ['Shapenet6D', 'NOCS', 'TOYL'], f"Unsupported dataset: {dataset_name}"
    if dataset_name == 'Shapenet6D':
        obj_id, name = 'all', 'ShapeNet6D'
    elif dataset_name == 'NOCS':
        obj_id, name = 'all', 'NOCS'
    elif dataset_name == 'TOYL':
        obj_id, name = 'all', 'TOYL'

    args_dict = {
        'dataset': {
            'root': root_path,
            'img_size': [480, 640],
            'max_corrs': 4,
            'train': {'name': 'Shapenet6D', 'split': 'train', 'obj': "all"},
            'train_shapenet': {'name': 'Shapenet6D', 'split': 'train', 'obj': 'all'},
            'train_nocs': {'name': 'NOCS', 'split': 'cross_scene_test', 'obj': 'all'},
            'train_toyl': {'name': 'TOYL', 'split': 'cross_scene_test', 'obj': 'all'},
            'test': {'name': 'NOCS', 'split': 'val', 'obj': 'all'}
        },
        'TRAINING': {
            'BATCH_SIZE': 64,
            'NUM_WORKERS': 4,
            'SAMPLER': 'scene_balance',
            'N_SAMPLES_SCENE': 4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter': True, 'bright': True, 'hflip': True, 'vflip': True},
            'text': {'synset': True}
        },
        'test': {'mask': 'oracle', 'add_description': 'yes'},
        'use_seed': True,
        'seed': seed,
        'debug_valid': 'anchor'
    }
    return args_dict, dataset_name


# ==================================================

def generate_caption(model, processor, images, prompt="a photography of"):
    image_token = AddedToken("<image>", normalized=False, special=True)
    processor.tokenizer.add_tokens([image_token], special_tokens=True)

    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
    model.config.image_token_index = len(processor.tokenizer) - 1
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(
        next(model.parameters()).device)
    inputs['pixel_values'] = inputs['pixel_values'].float()

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [x.strip() for x in generated_text]


def run_custom_dinov2_extraction(model_name, config_path, batch_size, resize_dim=518, crop_dim=518, out_path=None,
                                 write_as_wds=False, num_shards=25, n_in_splits=4, in_batch_offset=0, out_offset=0,
                                 extract_cls=False, extract_avg_self_attn=False, extract_second_last_out=False,
                                 extract_patch_tokens=False, extract_self_attn_maps=False,
                                 extract_disentangled_self_attn=False,
                                 blip_model_name=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取YAML配置（这里主要为了得到root路径）
    # with open(config_path, 'r') as f:
    #     cfg_data = yaml.safe_load(f)
    root_path ="/data/WDY/mickey-main/filesOfOryon/data" # 若配置中无dataset_root则默认'./data'

    # ============================================================
    # 🔹 按照训练脚本的方式创建 DataModule 并获取 train_dataloader()
    # ============================================================
    args_dict, dataset_name = get_dataset_args_dict('Shapenet6D', root_path)
    args_cfg = DictConfig(args_dict)
    data_module = DataModule(args_cfg)
    data_module.setup('fit')

    #train_dataloader = data_module.train_dataloader()
    # # ============================================================
    #
    # print(f"Loaded training dataset with {len(train_dataloader.dataset)} samples")
    # print(f"Number of batches: {len(train_dataloader)}")

    use_val=False
    if use_val:
        dataloader = data_module.val_dataloader()  # ✅ 如果选择验证集
    else:
        dataloader = data_module.train_dataloader()  # 默认训练集

    print(f"Loaded dataset with {len(dataloader.dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")

    # ======= 以下部分保持完全不变 =======
    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
        embed_dim = 1024
    elif 'vitb' in model_name or '_base' in model_name or 'ViT-B' in model_name:
        embed_dim = 768
    elif 'vits' in model_name or 'vit_small' in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")

    scale = 0.125

    if 'dinov2' in model_name:
        model_family = 'facebookresearch/dinov2'
        model = torch.hub.load(model_family, model_name)
        image_transforms = T.Compose([
            T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_dim),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        num_attn_heads = model.num_heads
    elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
        model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=crop_dim)
        data_config = timm.data.resolve_model_data_config(model)
        image_transforms = timm.data.create_transform(**data_config, is_training=False)
        if 'mae' in model_name or 'dino' in model_name or 'beit' in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = 1 + num_patch_tokens
        elif 'sam' in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = num_patch_tokens
            num_global_tokens = 0
            model.blocks[-1].register_forward_hook(get_vit_out)
        elif 'clip' in model_name:
            crop_dim = resize_dim = 224
            num_patch_tokens = crop_dim // 16 * crop_dim // 16 if 'vit_base' in model_name else crop_dim // 14 * crop_dim // 14
            num_tokens = 1 + num_patch_tokens
        num_attn_heads = model.blocks[-1].attn.num_heads
    elif 'ViT' in model_name:
        model, image_transforms = clip.load(model_name, device)
        num_attn_heads = model.num_heads
    else:
        raise Exception("Unknown ViT model")

    model.eval()
    model.to(device)

    if blip_model_name is not None:
        blip_processor = Blip2Processor.from_pretrained(blip_model_name)
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip_model_name, torch_dtype=torch.float16,
        ).to(device)
        blip_processor.num_query_tokens = blip_model.config.num_query_tokens

    if extract_second_last_out:
        model.blocks[-2].register_forward_hook(get_second_last_out)
    if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        if 'beit' in model_name:
            model.blocks[-1].attn.qkv_bias_separate = True

    print("Starting the features extraction...")
    output_data = {'images': [], 'annotations': []}
    n_errors = 0
    sample_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        batch_size_ = batch['image0'].shape[0]

        # 处理image0和image1，将它们作为独立的样本
        for i in range(batch_size_):
            # 处理image0
            try:
                # 从tensor转换为PIL图像
                img0_tensor = batch['image0'][i]  # [3, H, W]
                img0_pil = T.ToPILImage()(img0_tensor)

                # 获取对应的prompt
                prompt = batch['prompt'][i][0]
                #print(prompt)

                # 预处理图像
                img0_processed = image_transforms(img0_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

                # 提取特征
                with torch.no_grad():
                    if 'dinov2' in model_name:
                        outs = model(img0_processed, is_training=True)
                    elif 'mae' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
                        output = model.forward_features(img0_processed)
                        outs = {
                            'x_norm_clstoken': output[:, 0, :],
                            'x_norm_patchtokens': output[:, 1:, :],
                        }
                    elif 'sam' in model_name:
                        sam_output = model.forward_features(img0_processed)
                        if extract_cls:
                            cls = model.forward_head(sam_output, pre_logits=True)
                        else:
                            cls = None
                        outs = {
                            'x_norm_clstoken': cls,
                            'x_norm_patchtokens': feats['vit_out'].reshape(1, num_patch_tokens, embed_dim)
                        }
                    elif 'ViT' in model_name:
                        outs = {
                            'x_norm_clstoken': model.encode_image(img0_processed)
                        }

                cls_token = outs['x_norm_clstoken']

                # 处理自注意力特征
                if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
                    self_attn, self_attn_maps = process_self_attention(feats['self_attn'], 1, num_tokens,
                                                                       num_attn_heads, embed_dim, scale,
                                                                       num_global_tokens, ret_self_attn_maps=True)

                # 创建样本数据
                sample_data = {
                    'file_name': f'sample_{sample_count}_image0.jpg',
                    'image_id': sample_count,
                    'width': img0_pil.width,
                    'height': img0_pil.height,
                    'coco_url': '',
                    'flickr_url': ''
                }

                # 添加提取的特征
                if extract_cls or (not extract_avg_self_attn and not extract_second_last_out):
                    sample_data['dino_features'] = cls_token[0].to('cpu')
                if extract_avg_self_attn:
                    avg_self_attn_token = (self_attn.unsqueeze(-1) * outs['x_norm_patchtokens']).mean(dim=1)
                    sample_data['avg_self_attn_out'] = avg_self_attn_token[0].to('cpu')
                if extract_second_last_out:
                    second_last_cls = feats['second_last_out'][:, 0, :]
                    sample_data['second_last_out'] = second_last_cls[0].to('cpu')
                if extract_patch_tokens:
                    sample_data['patch_tokens'] = outs['x_norm_patchtokens'][0].to('cpu')
                if extract_self_attn_maps:
                    sample_data['self_attn_maps'] = self_attn_maps[0].to('cpu')
                if extract_disentangled_self_attn:
                    self_attn_maps = self_attn_maps.softmax(dim=-1)
                    disentangled_self_attn = (
                                outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
                    sample_data['disentangled_self_attn'] = disentangled_self_attn[0].to('cpu')

                # 添加标注信息
                annotation_data = {
                    'id': sample_count,
                    'image_id': sample_count,
                    'caption': prompt,
                    'annotation_id': sample_count
                }

                output_data['images'].append(sample_data)
                output_data['annotations'].append(annotation_data)
                sample_count += 1

            except Exception as e:
                print(f"Error processing image0 in batch {batch_idx}, sample {i}: {e}")
                n_errors += 1
                continue

            # 处理image1
            try:
                # 从tensor转换为PIL图像
                img1_tensor = batch['image1'][i]  # [3, H, W]
                img1_pil = T.ToPILImage()(img1_tensor)

                # 预处理图像
                img1_processed = image_transforms(img1_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

                # 提取特征
                with torch.no_grad():
                    if 'dinov2' in model_name:
                        outs = model(img1_processed, is_training=True)
                    elif 'mae' in model_name or 'clip' in model_name or 'dino' in model_name or 'beit' in model_name:
                        output = model.forward_features(img1_processed)
                        outs = {
                            'x_norm_clstoken': output[:, 0, :],
                            'x_norm_patchtokens': output[:, 1:, :],
                        }
                    elif 'sam' in model_name:
                        sam_output = model.forward_features(img1_processed)
                        if extract_cls:
                            cls = model.forward_head(sam_output, pre_logits=True)
                        else:
                            cls = None
                        outs = {
                            'x_norm_clstoken': cls,
                            'x_norm_patchtokens': feats['vit_out'].reshape(1, num_patch_tokens, embed_dim)
                        }
                    elif 'ViT' in model_name:
                        outs = {
                            'x_norm_clstoken': model.encode_image(img1_processed)
                        }

                cls_token = outs['x_norm_clstoken']

                # 处理自注意力特征
                if extract_avg_self_attn or extract_self_attn_maps or extract_disentangled_self_attn:
                    self_attn, self_attn_maps = process_self_attention(feats['self_attn'], 1, num_tokens,
                                                                       num_attn_heads, embed_dim, scale,
                                                                       num_global_tokens, ret_self_attn_maps=True)

                # 创建样本数据
                sample_data = {
                    'file_name': f'sample_{sample_count}_image1.jpg',
                    'image_id': sample_count,
                    'width': img1_pil.width,
                    'height': img1_pil.height,
                    'coco_url': '',
                    'flickr_url': ''
                }

                # 添加提取的特征
                if extract_cls or (not extract_avg_self_attn and not extract_second_last_out):
                    sample_data['dino_features'] = cls_token[0].to('cpu')
                if extract_avg_self_attn:
                    avg_self_attn_token = (self_attn.unsqueeze(-1) * outs['x_norm_patchtokens']).mean(dim=1)
                    sample_data['avg_self_attn_out'] = avg_self_attn_token[0].to('cpu')
                if extract_second_last_out:
                    second_last_cls = feats['second_last_out'][:, 0, :]
                    sample_data['second_last_out'] = second_last_cls[0].to('cpu')
                if extract_patch_tokens:
                    sample_data['patch_tokens'] = outs['x_norm_patchtokens'][0].to('cpu')
                if extract_self_attn_maps:
                    sample_data['self_attn_maps'] = self_attn_maps[0].to('cpu')
                if extract_disentangled_self_attn:
                    self_attn_maps = self_attn_maps.softmax(dim=-1)
                    disentangled_self_attn = (
                                outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
                    sample_data['disentangled_self_attn'] = disentangled_self_attn[0].to('cpu')

                # 添加标注信息
                annotation_data = {
                    'id': sample_count,
                    'image_id': sample_count,
                    'caption': prompt,  # 使用相同的prompt
                    'annotation_id': sample_count
                }

                output_data['images'].append(sample_data)
                output_data['annotations'].append(annotation_data)
                sample_count += 1

            except Exception as e:
                print(f"Error processing image1 in batch {batch_idx}, sample {i}: {e}")
                n_errors += 1
                continue


            # 🔹 每2000个样本写盘并清空缓存，防止内存爆掉
            if sample_count > 0 and sample_count % 2000 == 0:
                partial_path = f"{out_path or 'features'}_part_{sample_count // 2000}.pth"
                torch.save(output_data, partial_path)
                print(f"[Partial save] {partial_path} ({sample_count} samples processed)")
                output_data = {'images': [], 'annotations': []}  # 清空缓存
                torch.cuda.empty_cache()

    # ===== 保存结果部分保持不变 =====
    print("Feature extraction done!")
    print(f"Successfully processed {sample_count} samples")
    print(f"Failed to extract {n_errors} samples")

    # if write_as_wds:
    #     os.makedirs(out_path, exist_ok=True)
    #     create_webdataset_tar(output_data, out_path, num_shards, out_offset)
    # else:
    #     if out_path is None:
    #         out_path = f"custom_extracted_features.pth"
    #     torch.save(output_data, out_path)
    #     print(f"Features saved at {out_path}")
    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(output_data, out_path, num_shards, out_offset)
    else:
        # 🔹 保存当前批次的剩余样本
        if out_path is None:
            out_path = "custom_extracted_features"
        partial_path = f"{out_path}_final.pth"
        torch.save(output_data, partial_path)
        print(f"[Final save] {partial_path}")

        # 🔹 自动合并所有 part 文件
        import glob
        part_files = sorted(glob.glob(f"{out_path}_part_*.pth"))
        if os.path.exists(partial_path):
            part_files.append(partial_path)

        if len(part_files) > 1:
            print(f"Merging {len(part_files)} feature parts...")
            merged = {'images': [], 'annotations': []}
            for f in part_files:
                data = torch.load(f)
                merged['images'].extend(data['images'])
                merged['annotations'].extend(data['annotations'])
            merged_path = f"{out_path}_merged.pth"
            torch.save(merged, merged_path)
            print(f"✅ All parts merged and saved to: {merged_path}")
        else:
            print("No multiple parts detected, skipping merge.")


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--blip_model', type=str, default=None)
    parser.add_argument('--model', type=str, default="dinov2_vitl14_reg")
    parser.add_argument('--resize_dim', type=int, default=518)
    parser.add_argument('--crop_dim', type=int, default=518)
    parser.add_argument('--extract_cls', default=False, action="store_true")
    parser.add_argument('--extract_avg_self_attn', default=False, action="store_true")
    parser.add_argument('--extract_second_last_out', default=False, action="store_true")
    parser.add_argument('--extract_patch_tokens', default=False, action="store_true")
    parser.add_argument('--extract_self_attn_maps', default=False, action="store_true")
    parser.add_argument('--extract_disentangled_self_attn', default=False, action="store_true")
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--write_as_wds', action="store_true", default=False)
    parser.add_argument('--n_shards', type=int, default=25)
    parser.add_argument('--n_in_split', type=int, default=1)
    parser.add_argument('--in_batch_offset', type=int, default=0)
    parser.add_argument('--out_offset', type=int, default=0)
    args = parser.parse_args()

    run_custom_dinov2_extraction(args.model, None,args.batch_size, args.resize_dim, args.crop_dim,
                                 args.out_path,
                                 args.write_as_wds, args.n_shards, args.n_in_split, args.in_batch_offset,
                                 args.out_offset,
                                 args.extract_cls, args.extract_avg_self_attn, args.extract_second_last_out,
                                 args.extract_patch_tokens,
                                 args.extract_self_attn_maps, args.extract_disentangled_self_attn, args.blip_model)


if __name__ == '__main__':
    main()
