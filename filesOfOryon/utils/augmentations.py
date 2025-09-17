import torch
import random
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter
from filesOfOryon.utils import coordinates
from typing import Tuple
from torch import Tensor

class identity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample

class random_brightness(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, sample):
        item1, item2, corrs = sample
        if random.random() < self.prob:
            item1['rgb'] = ColorJitter(brightness=.25, contrast=.0, saturation=.0, hue=.00)(item1['rgb'])
        if random.random() < self.prob:
            item2['rgb'] = ColorJitter(brightness=.25, contrast=.0, saturation=.0, hue=.00)(item2['rgb'])

        return (
            item1,
            item2,
            corrs
        )

class random_jitter(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, sample):
        item1, item2, corrs = sample
        if random.random() < self.prob:
            item1['rgb'] = ColorJitter(brightness=.125, contrast=.5, saturation=.5, hue=.05)(item1['rgb'])
        if random.random() < self.prob:
            item2['rgb'] = ColorJitter(brightness=.125, contrast=.5, saturation=.5, hue=.05)(item2['rgb'])

        return (
            item1,
            item2,
            corrs
        )       

# class horizontal_flip(object):
#     def __init__(self, prob=.5):
#         self.prob = prob
#
#     def flip_item(self, item, coords):
#         H,W = item['hw_size']
#
#         # flip images
#         _coords = coords.clone()
#         item['rgb'] = F.hflip(item['rgb'])
#         item['depth'] = F.hflip(item['depth'].unsqueeze(0)).squeeze(0)
#         item['mask'] = F.hflip(item['mask'].unsqueeze(0)).squeeze(0)
#         # flip coordinates and boxes
#         y,x,h,w = item['metadata']['boxes'][0]
#         item['metadata']['boxes'][0] = torch.tensor([y,W-w-x,h,w])
#         _coords[:,1] = W - _coords[:,1] - 1
#
#         return item, _coords
#
#     def __call__(self, sample):
#
#         item1, item2, corrs = sample
#
#         _coords1, _coords2 = corrs[:,:2], corrs[:,2:]
#
#         if random.random() < self.prob:
#             item1, _coords1 = self.flip_item(item1, _coords1)
#
#         if random.random() < self.prob:
#             item2, _coords2 = self.flip_item(item2, _coords2)
#
#         corrs = torch.cat([_coords1,_coords2],dim=1)
#
#         return (
#             item1,
#             item2,
#             corrs
#         )
#
# class vertical_flip(object):
#     def __init__(self, prob=.5):
#         self.prob = prob
#
#     def flip_item(self, item, coords):
#         H,W = item['hw_size']
#         _coords = coords.clone()
#         # flip images
#         item['rgb'] = F.vflip(item['rgb'])
#         item['depth'] = F.vflip(item['depth'].unsqueeze(0)).squeeze(0)
#         item['mask'] = F.vflip(item['mask'].unsqueeze(0)).squeeze(0)
#         # flip coordinates: boxes are in x,y,w,h
#         #y,x,h,w = item['metadata']['boxes']
#         y, x, h, w = item['metadata']['boxes'][0]
#         item['metadata']['boxes'][0] = torch.tensor([H-y-h,x,h,w])
#         # correspondences are in y1,x1,y2,x2
#         _coords[:,0] = H - _coords[:,0] - 1
#
#         return item, _coords
#
#     def __call__(self, sample):
#
#         item1, item2, corrs = sample
#
#         _coords1, _coords2 = corrs[:,:2], corrs[:,2:]
#
#         if random.random() < self.prob:
#             item1, _coords1 = self.flip_item(item1, _coords1)
#
#         if random.random() < self.prob:
#             item2, _coords2 = self.flip_item(item2, _coords2)
#
#         corrs = torch.cat([_coords1,_coords2],dim=1)
#
#         return (
#             item1,
#             item2,
#             corrs
#         )
#
import torch
import random
from torchvision.transforms import functional as F

import torch
import random
from torchvision.transforms import functional as F
import numpy as np

class horizontal_flip(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def flip_item(self, item, coords):
        H, W = item['hw_size']
        _coords = coords.clone()

        # flip images
        item['rgb'] = F.hflip(item['rgb'])
        item['depth'] = F.hflip(item['depth'].unsqueeze(0)).squeeze(0)
        item['mask'] = F.hflip(item['mask'].unsqueeze(0)).squeeze(0)

        # flip coordinates and boxes
        box = item['metadata'].get('boxes', None)
        if box is not None and len(box) > 0:
            b = box[0]
            if isinstance(b, torch.Tensor):
                b = b.tolist() if b.ndim > 0 else [b.item()]
            elif isinstance(b, np.ndarray):
                b = b.tolist()
            # 仅在长度为4时解包
            if isinstance(b, (list, tuple)) and len(b) == 4:
                y, x, h, w = b
                item['metadata']['boxes'][0] = torch.tensor([y, W - w - x, h, w])
                # flip keypoint coords
                _coords[:, 1] = W - _coords[:, 1] - 1

        return item, _coords

    def __call__(self, sample):
        item1, item2, corrs = sample
        _coords1, _coords2 = corrs[:, :2], corrs[:, 2:]

        if random.random() < self.prob:
            item1, _coords1 = self.flip_item(item1, _coords1)
        if random.random() < self.prob:
            item2, _coords2 = self.flip_item(item2, _coords2)

        corrs = torch.cat([_coords1, _coords2], dim=1)
        return item1, item2, corrs


class vertical_flip(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def flip_item(self, item, coords):
        H, W = item['hw_size']
        _coords = coords.clone()

        # flip images
        item['rgb'] = F.vflip(item['rgb'])
        item['depth'] = F.vflip(item['depth'].unsqueeze(0)).squeeze(0)
        item['mask'] = F.vflip(item['mask'].unsqueeze(0)).squeeze(0)

        # flip coordinates and boxes
        box = item['metadata'].get('boxes', None)
        if box is not None and len(box) > 0:
            b = box[0]
            if isinstance(b, torch.Tensor):
                b = b.tolist() if b.ndim > 0 else [b.item()]
            elif isinstance(b, np.ndarray):
                b = b.tolist()
            # 仅在长度为4时解包
            if isinstance(b, (list, tuple)) and len(b) == 4:
                y, x, h, w = b
                item['metadata']['boxes'][0] = torch.tensor([H - y - h, x, h, w])
                # flip keypoint coords
                _coords[:, 0] = H - _coords[:, 0] - 1

        return item, _coords

    def __call__(self, sample):
        item1, item2, corrs = sample
        _coords1, _coords2 = corrs[:, :2], corrs[:, 2:]

        if random.random() < self.prob:
            item1, _coords1 = self.flip_item(item1, _coords1)
        if random.random() < self.prob:
            item2, _coords2 = self.flip_item(item2, _coords2)

        corrs = torch.cat([_coords1, _coords2], dim=1)
        return item1, item2, corrs

class resize(object):
    def __init__(self, size):
        self.size = size

    # def resize_item(self, item : dict, coords: Tensor) -> Tuple[dict, Tensor]:
    #
    #     H,W = item['mask'].shape
    #     # original rgb and original depth are untouched
    #     item['rgb'] = F.resize(item['rgb'], size=list(self.size), interpolation=F.InterpolationMode.BILINEAR)
    #     item['mask'] = F.resize(item['mask'].unsqueeze(0), size=list(self.size), interpolation=F.InterpolationMode.NEAREST).squeeze(0)
    #     item['depth'] = F.resize(item['depth'].unsqueeze(0), size=list(self.size), interpolation=F.InterpolationMode.BILINEAR).squeeze(0)
    #     #item['hw_size'] = torch.tensor(self.crop_size)
    #
    #     # rescale box (even though is not used in this setting)
    #     #y1,x1,h,w = item['metadata']['boxes'][0]
    #     box = item['metadata']['boxes']
    #     #assert box.shape[-1] == 4, f"Expected box of shape [4], got {box.shape}"
    #     import numpy as np
    #     if isinstance(box, list):
    #         box = np.array(box, dtype=np.float32)
    #
    #     assert box.shape[-1] == 4, f"Expected box of shape [4], got {box.shape}"
    #     y1, x1, h, w = box.tolist()  # or just: y1, x1, h, w = box
    #
    #     h_ratio, w_ratio = self.size[0] / float(H), self.size[1] / float(W)
    #     #item['metadata']['boxes'][0] = torch.tensor([y1*h_ratio,x1*w_ratio,h*h_ratio,w*w_ratio])
    #
    #     item['metadata']['boxes'] = torch.tensor([y1 * h_ratio, x1 * w_ratio, h * h_ratio, w * w_ratio])
    #     # rescale coordinates
    #     coords = coordinates.scale_coords(coords, (H,W), self.size)
    #
    #     return item, coords
    def resize_item(self, item: dict, coords: Tensor) -> Tuple[dict, Tensor]:
        """
        Resize rgb, mask, depth, boxes, and scale keypoint coordinates.

        Args:
            item: dict containing 'rgb', 'mask', 'depth', 'metadata' with 'boxes'
            coords: Tensor of shape [N, 2] keypoint coordinates (x, y)

        Returns:
            item: updated dict with resized tensors
            coords: scaled coordinates
        """
        H, W = item['mask'].shape

        # Resize rgb, mask, depth
        item['rgb'] = F.resize(item['rgb'], size=list(self.size), interpolation=F.InterpolationMode.BILINEAR,antialias=True)
        item['mask'] = F.resize(item['mask'].unsqueeze(0), size=list(self.size),
                                 interpolation=F.InterpolationMode.NEAREST,antialias=False).squeeze(0)
        item['depth'] = F.resize(item['depth'].unsqueeze(0), size=list(self.size),
                                  interpolation=F.InterpolationMode.BILINEAR,antialias=True).squeeze(0)

        # ----- Process boxes -----
        box = item['metadata'].get('boxes', None)
        if box is not None:
            # Convert list -> np.array
            if isinstance(box, list):
                box = np.array(box, dtype=np.float32)
            # Convert tensor -> numpy
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            # Flatten if shape is (1,4)
            if box.ndim == 2 and box.shape[0] == 1:
                box = box[0]
            # Scale box if 4 elements
            if box.shape[-1] == 4:
                y1, x1, h, w = box.tolist()
                h_ratio, w_ratio = self.size[0] / float(H), self.size[1] / float(W)
                item['metadata']['boxes'] = torch.tensor([y1 * h_ratio, x1 * w_ratio, h * h_ratio, w * w_ratio])
            else:
                # 不符合 4 维，直接保留原样
                item['metadata']['boxes'] = torch.tensor(box)

        # ----- Scale keypoint coordinates -----
        if coords is not None and len(coords) > 0:
            coords = coordinates.scale_coords(coords, (H, W), self.size)

        return item, coords

    def __call__(self, sample):

        item1, item2, corrs = sample

        item1, corrs_a = self.resize_item(item1, corrs[:,:2])
        item2, corrs_q = self.resize_item(item2, corrs[:,2:])

        corrs = torch.cat([corrs_a,corrs_q],dim=1)

        return (
            item1,
            item2,
            corrs
        )

