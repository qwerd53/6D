
import torch.utils as utils
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from filesOfOryon.datasets import Shapenet6DDataset, NOCSDataset, TOYLDataset
from lib.datasets.sampler import RandomConcatSampler

from torch.utils.data import ConcatDataset
import torch
from torch.utils.data import Subset

import torch
from torch.nn.functional import interpolate
import numpy as np
def custom_collate(batch, img_size=(256, 256)):
    """
    将 dataset 返回的 tuple 转为 dict，包括：
    - 'image0', 'image1': RGB 图像 [B, 3, H, W]
    - 'mask0', 'mask1': 掩码图 [B, H, W]
    - 'K_color0', 'K_color1': 相机内参 [B, 3, 3]
    - 'pose': 相对位姿 [B, 4, 4]
    - 'prompt': list[str] or list[list[str]]
    - 'obj_id': list[int]
    - 'instance_id': list[str]
    """

    def to_tensor(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype)
        else:
            raise TypeError(f"Unsupported type {type(x)} for to_tensor")
    image0_list = []
    image1_list = []
    mask0_list = []
    mask1_list = []
    K0_list = []
    K1_list = []
    poses = []
    prompts = []
    obj_ids = []
    instance_ids = []
    instance_id_a_list = []
    instance_id_q_list = []
    item_a_pose_list = []
    item_q_pose_list = []

    #metadata
    metadata0_list=[]
    metadata1_list=[]
    #depth
    depth0_list=[]
    depth1_list=[]
    orig_depth0_list = []
    orig_depth1_list = []


    for sample in batch:
        item_a, item_q, prompt, pose, obj_id, instance_id = sample  # 解包

        # 图像
        image0_list.append(item_a['rgb'])
        image1_list.append(item_q['rgb'])

        # 相机内参
        # K0_list.append(torch.tensor(item_a['camera'], dtype=torch.float32))
        # K1_list.append(torch.tensor(item_q['camera'], dtype=torch.float32))
        K0_list.append(item_a['camera'].clone().detach().float())
        K1_list.append(item_q['camera'].clone().detach().float())

        # mask
        mask_a = item_a.get('mask')
        mask_q = item_q.get('mask')
        #print(mask_a.shape)
        mask0_list.append(mask_a.squeeze(0).float())
        mask1_list.append(mask_q.squeeze(0).float())
        # if mask_a is not None:
        #     if mask_a.dim() == 2:
        #         mask_a = mask_a.unsqueeze(0)  # [1, H, W]
        #         print(imag)
        #     mask_a = interpolate(mask_a.float().unsqueeze(0), size=img_size, mode='nearest').squeeze(0)  # [1, H, W] → [H, W]
        #     mask0_list.append(mask_a.squeeze(0))
        # else:
        #     mask0_list.append(torch.zeros(img_size, dtype=torch.float32))  # fallback
        #
        # if mask_q is not None:
        #     if mask_q.dim() == 2:
        #         mask_q = mask_q.unsqueeze(0)
        #     mask_q = interpolate(mask_q.float().unsqueeze(0), size=img_size, mode='nearest').squeeze(0)
        #     mask1_list.append(mask_q.squeeze(0))
        # else:
        #     mask1_list.append(torch.zeros(img_size, dtype=torch.float32))  # fallback

        # pose
        #poses.append(pose)
        poses.append(to_tensor(pose))

        # prompt
        prompts.append(prompt)

        # ids
        obj_ids.append(obj_id)
        instance_ids.append(instance_id)
        instance_id_a_list.append(item_a['instance_id'])
        instance_id_q_list.append(item_q['instance_id'])

        # GT pose
        #item_a_pose = torch.tensor(item_a['metadata']['poses'][0], dtype=torch.float32)
        #item_q_pose = torch.tensor(item_q['metadata']['poses'][0], dtype=torch.float32)
        item_a_pose = item_a['metadata']['poses'][0].clone().detach().float()
        item_q_pose = item_q['metadata']['poses'][0].clone().detach().float()

        item_a_pose_list.append(item_a_pose)
        item_q_pose_list.append(item_q_pose)

        # GT pose
        # item_a_pose_list.append(to_tensor(item_a['metadata']['poses'][0]))
        # item_q_pose_list.append(to_tensor(item_q['metadata']['poses'][0]))

        metadata0_list.append(item_a['metadata'])
        metadata1_list.append(item_q['metadata'])

        #depth
        depth0_list.append(item_a['depth'])
        depth1_list.append(item_q['depth'])
        #print(item_a['depth'])

        orig_depth0_list.append(item_a['orig_depth'].squeeze())
        orig_depth1_list.append(item_q['orig_depth'].squeeze())


    batch_dict = {
        'image0': torch.stack(image0_list),           # [B, 3, H, W]
        'image1': torch.stack(image1_list),
        #'mask0': torch.stack(mask0_list),             # [B, H, W]
        #'mask1': torch.stack(mask1_list),
        'mask0_gt': torch.stack(mask0_list),  # 添加 _gt 后缀
        'mask1_gt': torch.stack(mask1_list),
        'K_color0': torch.stack(K0_list),             # [B, 3, 3]
        'K_color1': torch.stack(K1_list),
        'pose': torch.stack(poses),                   # [B, 4, 4]
        'prompt': prompts,
        'obj_id': obj_ids,
        'instance_id': instance_ids,
        'instance_id_a': instance_id_a_list,
        'item_a_pose': torch.stack(item_a_pose_list), # [B, 4, 4]
        'instance_id_q': instance_id_q_list,
        'item_q_pose': torch.stack(item_q_pose_list), # [B, 4, 4]

        'metadata0':metadata0_list, #B,'cls_ids','mask_ids','cls_names','cls_descs', 'poses','boxes'
        'metadata1':metadata1_list,

        #depth
        'depth0': torch.stack(depth0_list),
        'depth1': torch.stack(depth1_list),
        'orig_depth0':torch.stack(orig_depth0_list),
        'orig_depth1':torch.stack(orig_depth1_list)
    }

    return batch_dict



class DataModule(pl.LightningDataModule):
    def __init__(self, args, train_dataset_name='Shapenet6D', val_dataset_name='NOCS', drop_last_val=True):
        super().__init__()
        self.args = args
        #self.dataset_name = dataset_name
        self.drop_last_val = drop_last_val

        self.datasets_map = {
            'Shapenet6D': Shapenet6DDataset,
            'NOCS': NOCSDataset,
            'TOYL': TOYLDataset
        }

        # 指定训练集和验证集类
        assert train_dataset_name in self.datasets_map, f'Unknown train dataset: {train_dataset_name}'
        assert val_dataset_name in self.datasets_map, f'Unknown val dataset: {val_dataset_name}'
        self.train_dataset_class = self.datasets_map[train_dataset_name]
        self.val_dataset_class = self.datasets_map[val_dataset_name]
        # assert dataset_name in self.datasets_map, f'Unknown dataset: {dataset_name}'
        # self.dataset_class = self.datasets_map[dataset_name]

    def get_sampler(self, dataset, reset_epoch=False):
        if self.args['TRAINING']['SAMPLER'] == 'scene_balance':
            return RandomConcatSampler(
                dataset,
                self.args['TRAINING']['N_SAMPLES_SCENE'],
                self.args['TRAINING']['SAMPLE_WITH_REPLACEMENT'],
                shuffle=True,
                reset_on_iter=reset_epoch
            )
        return None



    def train_dataloader(self):
        #dataset = self.dataset_class(self.args, eval=False)
        dataset = self.train_dataset_class(self.args, eval=False)
        print(f"Loaded training dataset with {len(dataset)} samples")

    #    使用完整数据
        #sampler = self.get_sampler(dataset)


        # 包装成 ConcatDataset 以兼容 sampler
        dataset = ConcatDataset([dataset])

        sampler = self.get_sampler(dataset)
        collate_fn = dataset.datasets[0].collate
        return DataLoader(
            dataset,
            batch_size=self.args['TRAINING']['BATCH_SIZE'],
            num_workers=self.args['TRAINING']['NUM_WORKERS'],
            #sampler=sampler,
            # collate_fn=collate_fn,  # 直接从数据集获取collate函数
            collate_fn=custom_collate, #dict
            #drop_last=True,
            shuffle=True
        )

    # ------------------------------------------------------
    #

    # def train_dataloader(self):
    #     # 保存原始 train 配置
    #     orig_train = self.args['dataset']['train']
    #
    #     # -------- Shapenet6D --------
    #     self.args['dataset']['train'] = self.args['dataset']['train_shapenet']
    #     shapenet_dataset = Shapenet6DDataset(self.args, eval=False)
    #
    #     # -------- NOCS --------
    #    # self.args['dataset']['train'] = self.args['dataset']['train_nocs']
    #    # nocs_dataset = NOCSDataset(self.args, eval=False)
    #
    #     # -------- TOYL --------
    #     self.args['dataset']['train'] = self.args['dataset']['train_toyl']
    #     toyl_dataset = TOYLDataset(self.args, eval=False)
    #
    #     # 恢复原始配置，防止副作用
    #     self.args['dataset']['train'] = orig_train
    #
    #     print(f"Loaded Shapenet6D training dataset with {len(shapenet_dataset)} samples")
    #    # print(f"Loaded NOCS dataset with {len(nocs_dataset)} samples")
    #     print(f"Loaded TOYL dataset with {len(toyl_dataset)} samples")
    #
    #    # nocs_subset_size = max(1, len(nocs_dataset) // 10)
    #     toyl_subset_size = max(1, len(toyl_dataset) // 100)
    #    # nocs_indices = torch.randperm(len(nocs_dataset))[:nocs_subset_size]
    #     toyl_indices = torch.randperm(len(toyl_dataset))[:toyl_subset_size]
    #
    #    # nocs_subset = Subset(nocs_dataset, nocs_indices)
    #     toyl_subset = Subset(toyl_dataset, toyl_indices)
    #
    #     #merged_dataset = ConcatDataset([shapenet_dataset, nocs_subset, toyl_subset])
    #     merged_dataset = ConcatDataset([shapenet_dataset, toyl_subset])
    #
    #     return DataLoader(
    #         merged_dataset,
    #         batch_size=self.args['TRAINING']['BATCH_SIZE'],
    #         num_workers=self.args['TRAINING']['NUM_WORKERS'],
    #         collate_fn=custom_collate,
    #         drop_last=True,
    #         shuffle=True
    #     )

    def val_dataloader(self):
        dataset = self.val_dataset_class(self.args, eval=True)
        print(f"Loaded valting dataset with {len(dataset)} samples")
        return DataLoader(
            dataset,
            batch_size=self.args['TRAINING']['BATCH_SIZE'],
            num_workers=self.args['TRAINING']['NUM_WORKERS'],
            # collate_fn=dataset.collate,
            collate_fn=custom_collate,  # dict
            #drop_last=self.drop_last_val
        )

    def test_dataloader(self):
        dataset = self.dataset_class(self.args, eval=True)
        print(f"Loaded testing dataset with {len(dataset)} samples")
        return DataLoader(
            dataset,
            batch_size=self.args['TRAINING']['BATCH_SIZE'],
            num_workers=self.args['TRAINING']['NUM_WORKERS'],
            shuffle=False,
            # collate_fn=dataset.collate,
            collate_fn=custom_collate,  # dict
            #drop_last=self.drop_last_val
        )
