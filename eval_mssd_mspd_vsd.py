import os
import json
import zipfile
import pickle
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
from transforms3d.quaternions import quat2mat
from transforms3d.axangles import axangle2mat

from filesOfOryon.bop_toolkit_lib import pose_error
from filesOfOryon.bop_toolkit_lib.pose_error import my_mssd, my_mspd, vsd
from filesOfOryon.bop_toolkit_lib.renderer_vispy import RendererVispy
from filesOfOryon.utils.data.nocs import get_obj_rendering  # ✅ 加载模型的正确方法

# ---------------------- Config ----------------------
SUBMISSION_PATH = 'results/submission.zip'
ANNOTATION_PATH = 'filesOfOryon/data/NOCS/fixed_split/val/annots.pkl'
MODELS_INFO_PATH = 'filesOfOryon/data/NOCS/obj_models/real_test/models_info.json'
MODELS_ROOT = 'filesOfOryon/data/NOCS'
CATEGORIES_PATH = 'filesOfOryon/data/NOCS/categories.json'
DEPTH_ROOT = 'filesOfOryon/data/NOCS/split/real_test'

ADD_THRESHOLD = 0.005
MSSD_REC = np.arange(0.05, 0.51, 0.05)
MSPD_REC = np.arange(5, 51, 5)
VSD_REC = np.arange(0.05, 0.51, 0.05)
VSD_DELTA = 15.0

K_dict = {"fx": 591.0125, "fy": 590.16775, "cx": 322.525, "cy": 244.11084}
K = np.array([
    [K_dict["fx"], 0, K_dict["cx"]],
    [0, K_dict["fy"], K_dict["cy"]],
    [0, 0, 1]
], dtype=np.float32)

# ---------------------- Load models info ----------------------
with open(MODELS_INFO_PATH, 'r') as f:
    models_info = json.load(f)
model_diameters = {k: v['diameter'] for k, v in models_info.items()}

def get_symmetries(model_name):
    info = models_info.get(model_name, {})
    syms = info.get('symmetries_continuous', [])
    sym_matrices = []
    for sym in syms:
        axis = np.array(sym['axis'], dtype=np.float32)
        axis_norm = axis / np.linalg.norm(axis)
        R_sym = axangle2mat(axis_norm, np.pi * 2)
        T_sym = np.eye(4, dtype=np.float32)
        T_sym[:3, :3] = R_sym
        sym_matrices.append(T_sym)
    if len(sym_matrices) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    else:
        return np.stack(sym_matrices, axis=0).astype(np.float32)

with open(CATEGORIES_PATH, 'r') as f:
    category_id_to_name = json.load(f)
name_to_category_id = {v: k for k, v in category_id_to_name.items()}

def fix_instance_id(instance_id):
    parts = instance_id.split('_')
    if len(parts) < 6:
        return instance_id
    obj_name = parts[4]
    category_id = name_to_category_id.get(obj_name, None)
    if category_id is None:
        return instance_id
    if parts[4] != category_id:
        parts.insert(4, category_id)
    return '_'.join(parts)

def pose_to_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def load_depth_image(scene_q, img_q):
    depth_path = os.path.join(DEPTH_ROOT, f'scene_{scene_q}', f'{img_q.zfill(4)}_depth.png')
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"[Missing] depth: {depth_path}")
    depth = imageio.imread(depth_path).astype(np.float32) / 1000.0
    return depth

# ---------------------- Load GT ----------------------
with open(ANNOTATION_PATH, 'rb') as f:
    annots_all = pickle.load(f)
print(f'Loaded {len(annots_all)} ground truth poses.')

# ---------------------- Load predictions ----------------------
pred_poses = {}
with zipfile.ZipFile(SUBMISSION_PATH, 'r') as zipf:
    for name in zipf.namelist():
        with zipf.open(name) as f:
            lines = f.read().decode('utf-8').splitlines()
            for line in lines:
                items = line.strip().split()
                instance_id = items[0]
                q = np.array([float(v) for v in items[1:5]], dtype=np.float32)
                t = np.array([float(v) for v in items[5:8]], dtype=np.float32).reshape(3, 1)
                pred_poses[instance_id] = {'q': q, 't': t}
print(f'Loaded {len(pred_poses)} predicted poses.')

# ---------------------- Init Renderer for VSD ----------------------
renderer = RendererVispy(640, 480, mode='depth')
models_cache = {}
for obj_name in model_diameters.keys():
    try:
        model = get_obj_rendering(MODELS_ROOT, obj_name)
        renderer.my_add_object(model, obj_name)
        models_cache[obj_name] = model
    except FileNotFoundError:
        print(f'[Warning] Missing model: {obj_name}')
        continue

# ---------------------- Evaluation ----------------------
adds_accurate, mssd_all, mspd_all, vsd_all = [], [], [], []

for instance_id, pred in tqdm(pred_poses.items(), desc="Evaluating"):
    fixed_id = fix_instance_id(instance_id)
    if fixed_id not in annots_all:
        print(f"[Warning] Missing GT for: {fixed_id}")
        continue
    gt_data = annots_all[fixed_id]
    gt_pose = gt_data['gt']
    gt_pose[:3, 3] = gt_pose[:3, 3] / 1000.0

    R_pred = quat2mat(pred['q'])
    t_pred = pred['t']
    T_pred = pose_to_matrix(R_pred, t_pred)

    obj_name = '_'.join(fixed_id.split('_')[5:])
    scene_q, img_q = fixed_id.split('_')[2], fixed_id.split('_')[3]
    try:
        model = models_cache[obj_name]
        model_pts = model['pts'] / 1000.0  # Convert mm to meters
        depth = load_depth_image(scene_q, img_q)
    except KeyError:
        print(f"[Warning] Missing cached model for {obj_name}")
        continue
    except FileNotFoundError as e:
        print(e)
        continue

    diameter = model_diameters.get(obj_name, None)
    if diameter is None:
        print(f'[Warning] Missing diameter for {obj_name}')
        continue

    symmetries = get_symmetries(obj_name)
    if symmetries.shape[0] == 0:
        error = pose_error.add(T_pred[:3, :3], T_pred[:3, 3], gt_pose[:3, :3], gt_pose[:3, 3], model_pts)
    else:
        error = pose_error.adi(T_pred[:3, :3], T_pred[:3, 3], gt_pose[:3, :3], gt_pose[:3, 3], model_pts)
    adds_accurate.append(error < ADD_THRESHOLD * diameter)

    pred_R, pred_t = T_pred[:3, :3], T_pred[:3, 3:4] * 1000
    gt_R, gt_t = gt_pose[:3, :3], gt_pose[:3, 3:4] * 1000

    if symmetries.shape[0] == 0:
        symmetries = np.eye(4, dtype=np.float32)[None, ...]

    mssd_err = my_mssd(pred_R, pred_t, gt_R, gt_t, model_pts, symmetries)
    mssd_rec = MSSD_REC * diameter
    mssd_all.append(np.mean(mssd_err < mssd_rec))

    mspd_err = my_mspd(pred_R, pred_t, gt_R, gt_t, K, model_pts, symmetries)
    mspd_all.append(np.mean(mspd_err < MSPD_REC))

    vsd_errs = vsd(pred_R, pred_t, gt_R, gt_t, depth, K, VSD_DELTA, VSD_REC, True, diameter, renderer, obj_name)
    vsd_score = np.mean(np.asarray(vsd_errs) < VSD_REC[:, None])
    vsd_all.append(vsd_score)

# ---------------------- Results ----------------------
def report(name, values):
    if values:
        print(f'{name} Accuracy: {100. * np.mean(values):.2f}%')
    else:
        print(f'{name} Accuracy: NaN')

report('ADD(S)-0.1d', adds_accurate)
report('MSSD', mssd_all)
report('MSPD', mspd_all)
report('VSD', vsd_all)
