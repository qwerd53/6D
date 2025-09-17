import os
import pickle
import numpy as np

# 配置路径
ANNOTATION_PATH = 'filesOfOryon/data/NOCS/fixed_split/val/annots.pkl'
OUTPUT_TXT_PATH = 'filesOfOryon/data/NOCS/fixed_split/val/annots.txt'

# 加载GT数据
with open(ANNOTATION_PATH, 'rb') as f:
    annots_all = pickle.load(f)

print(f'Loaded {len(annots_all)} ground truth poses.')

# 写入txt文件
with open(OUTPUT_TXT_PATH, 'w') as f:
    for instance_id, gt_data in annots_all.items():
        gt_pose = gt_data['gt']
        # 提取旋转矩阵和平移向量
        R = gt_pose[:3, :3]
        t = gt_pose[:3, 3]
        # 将矩阵展平为一行
        R_flat = ' '.join(map(str, R.flatten()))
        t_flat = ' '.join(map(str, t.flatten()))
        # 写入文件：实例ID + 旋转矩阵(9个值) + 平移向量(3个值)
        f.write(f"{instance_id} {R_flat} {t_flat}\n")

print(f'Successfully exported GT poses to {OUTPUT_TXT_PATH}')