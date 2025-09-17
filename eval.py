import os
import json
import zipfile
import pickle
import numpy as np
from tqdm import tqdm
from filesOfOryon.bop_toolkit_lib import pose_error
from transforms3d.quaternions import quat2mat

# ---------------------- Config ----------------------
SUBMISSION_PATH = 'results/submission.zip'
ANNOTATION_PATH = 'filesOfOryon/data/NOCS/fixed_split/val/annots.pkl'
MODELS_INFO_PATH = 'filesOfOryon/data/NOCS/obj_models/real_test/models_info.json'
MODELS_ROOT = 'filesOfOryon/data/NOCS/obj_models/real_test/'
CATEGORIES_PATH = 'filesOfOryon/data/NOCS/categories.json'
ADD_THRESHOLD = 0.005

# ---------------------- Load models info ----------------------
with open(MODELS_INFO_PATH, 'r') as f:
    models_info = json.load(f)

model_diameters = {k: v['diameter'] for k, v in models_info.items()}
model_symmetry = {k: ('symmetries_continuous' in v and len(v['symmetries_continuous']) > 0) for k, v in models_info.items()}

# ---------------------- Load categories ----------------------
with open(CATEGORIES_PATH, 'r') as f:
    category_id_to_name = json.load(f)
name_to_category_id = {v: k for k, v in category_id_to_name.items()}  # name -> id

def fix_instance_id(instance_id):
    parts = instance_id.split('_')
    if len(parts) < 6:
        return instance_id  # 不合法返回

    obj_name = parts[4]
    category_id = name_to_category_id.get(obj_name, None)
    if category_id is None:
        return instance_id

    if parts[4] != category_id:  # 避免重复添加
        parts.insert(4, category_id)
    return '_'.join(parts)

# ---------------------- Load ground truth ----------------------
with open(ANNOTATION_PATH, 'rb') as f:
    annots_all = pickle.load(f)
    #print(annots_all)

print(f'Loaded {len(annots_all)} ground truth poses.')

# ---------------------- Load submission ----------------------
pred_poses = {}
with zipfile.ZipFile(SUBMISSION_PATH, 'r') as zipf:
    for name in zipf.namelist():
        with zipf.open(name) as f:
            lines = f.read().decode('utf-8').splitlines()
            for line in lines:
                items = line.strip().split()
                instance_id = items[0]
                q = np.array([float(v) for v in items[1:5]], dtype=np.float32)
                #t = np.array([float(v) for v in items[5:8]], dtype=np.float32).reshape(3, 1)/1000
                t = np.array([float(v) for v in items[5:8]], dtype=np.float32).reshape(3, 1)
                #print(t)
                pred_poses[instance_id] = {'q': q, 't': t}

print(f'Loaded {len(pred_poses)} predicted relative poses.')

# ---------------------- Helper functions ----------------------
def pose_to_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def load_vertices_txt(models_root, filename):
    path = os.path.join(models_root, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model vertices file not found: {path}")
    pts = np.loadtxt(path, dtype=np.float32)  # 读成 [N,3]
    return pts

# ---------------------- Evaluate ----------------------
adds_accurate = []

for instance_id, pred in tqdm(pred_poses.items(), desc="Evaluating"):
    fixed_instance_id = fix_instance_id(instance_id)
    print(fixed_instance_id)

    if fixed_instance_id not in annots_all:
        print(f"[Warning] Missing GT for instance_id: {fixed_instance_id}")
        continue

    gt_data = annots_all[fixed_instance_id]
    gt_pose = gt_data['gt']
    # print(gt_pose)
    # print( '\n' )
    obj_name = '_'.join(fixed_instance_id.split('_')[5:])  # 物体名

    Trel_gt = gt_pose
    # gt_pose: 4x4
    gt_pose[:3, 3] = gt_pose[:3, 3] / 1000.0

    #print(gt_pose)


    R_pred = quat2mat(pred['q'])
    t_pred = pred['t']
    Trel_pred = pose_to_matrix(R_pred, t_pred)

    try:
        model_pts = load_vertices_txt(MODELS_ROOT, obj_name + '_vertices.txt')
    except FileNotFoundError:
        print(f'[Warning] Missing model vertices for: {obj_name}')
        continue

    diameter = model_diameters.get(obj_name, None)
    if diameter is None:
        print(f'[Warning] Missing diameter for {obj_name}')
        continue

    is_symmetric = model_symmetry.get(obj_name, False)

    if is_symmetric:
        error = pose_error.adi(
            Trel_pred[:3, :3], Trel_pred[:3, 3],
            Trel_gt[:3, :3], Trel_gt[:3, 3],
            model_pts
        )
    else:
        error = pose_error.add(
            Trel_pred[:3, :3], Trel_pred[:3, 3],
            Trel_gt[:3, :3], Trel_gt[:3, 3],
            model_pts
        )

    adds_accurate.append(error < ADD_THRESHOLD * diameter)

accuracy = float('nan') if len(adds_accurate) == 0 else 100.0 * np.mean(adds_accurate)
print(f'ADD(S)-{ADD_THRESHOLD}d Accuracy: {accuracy:.2f}%')
