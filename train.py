import argparse
import os
# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from config.default import cfg
#from lib.datasets.datamodules import DataModuleTraining, CustomDataset
from lib.models.MicKey.model import MicKeyTrainingModel
from lib.models.MicKey.modules.utils.training_utils import create_exp_name, create_result_dir
import random
import shutil
from lib.models.Oryon.oryon import Oryon
from torch.utils.data import DataLoader

import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import pytorch_lightning as pl
import torch
import random
import shutil
from pytorch_lightning.loggers import TensorBoardLogger
from lib.models.MicKey.model import MicKeyTrainingModel
from lib.models.MicKey.modules.utils.training_utils import create_exp_name, create_result_dir
from lib.datasets.datamodules import DataModule
from omegaconf import DictConfig

def get_dataset_args_dict(dataset_name: str, root_path: str, seed: int = 42):
    assert dataset_name in ['Shapenet6D', 'NOCS', 'TOYL'], f"Unsupported dataset: {dataset_name}"
    if dataset_name == 'Shapenet6D':
        obj_id, name = 'all', 'ShapeNet6D'
    elif dataset_name == 'NOCS':
        obj_id, name = 'all', 'NOCS'
        #obj_id, name = '1', 'NOCS'
    elif dataset_name == 'TOYL':
        obj_id, name = '1', 'TOYL'

    args_dict = {
        'dataset': {
            'root': root_path,
            # 'img_size': [192, 192],#[256, 256],[480, 640]
            'img_size': [384,384],  # [
            'max_corrs': 4,
            'train': {'name': name, 'split': 'train', 'obj': obj_id},
            'test': {'name': name, 'split': 'val', 'obj': obj_id}
        },
        'TRAINING': {
            'BATCH_SIZE': 2 , #8,
            'NUM_WORKERS':2,  #16,
            'SAMPLER': 'scene_balance',
            # 'N_SAMPLES_SCENE': 100,
            'N_SAMPLES_SCENE':4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter':False, #True,
                    'bright':False, #True,
                    'hflip':False, #True,
                    'vflip':False}, #True},
            'text':
                {'synset': False}
        },
        'test': {'mask': 'oracle', 'add_description': 'yes'},
        'use_seed': True,
        'seed': seed,
        'debug_valid': 'anchor'
    }
    return args_dict, dataset_name

def train_model(args, cfg):
    cfg.DATASET.SEED = random.randint(0, 1000000)
    model = MicKeyTrainingModel(cfg)

    logger = TensorBoardLogger(save_dir=args.path_weights, name=args.experiment)
    exp_name = create_exp_name(args.experiment, cfg)
    print('Start training of', exp_name)

    trainer = pl.Trainer(
        #OOM
        # precision="16-mixed",  # 启用混合精度
        # accumulate_grad_batches=4,
        precision="32",
        #gradient_clip_val=1.0,  # 防止梯度爆炸
        # limit_train_batches=1, # train用10%数据测试内存

        devices=cfg.TRAINING.NUM_GPUS,
        #devices=2,
        log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
        val_check_interval=cfg.TRAINING.VAL_INTERVAL,
        limit_val_batches=cfg.TRAINING.VAL_BATCHES,
        max_epochs=cfg.TRAINING.EPOCHS,
        logger=logger,
        # callbacks=[
        #     pl.callbacks.ModelCheckpoint(filename='{epoch}-best_vcre', monitor='val_vcre/auc_vcre', mode='max'),
        #     pl.callbacks.ModelCheckpoint(filename='{epoch}-best_pose', monitor='val_AUC_pose/auc_pose', mode='max'),
        #     pl.callbacks.ModelCheckpoint(filename='e{epoch}-last', every_n_epochs=1),
        #     pl.callbacks.LearningRateMonitor(logging_interval='step')
        # ],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                filename='{epoch}-best_vcre{val_vcre/auc_vcre:.3f}',
                monitor='val_vcre/auc_vcre',
                mode='max',
                save_top_k=1,
                verbose=True  # ← 输出日志
            ),
            pl.callbacks.ModelCheckpoint(
                filename='{epoch}-best_pose{val_AUC_pose/auc_pose:.3f}',
                monitor='val_AUC_pose/auc_pose',
                mode='max',
                save_top_k=1,
                verbose=True
            ),
            pl.callbacks.ModelCheckpoint(
                filename='e{epoch}-last',
                every_n_epochs=1,
                save_top_k=1,  # 保存zuihou epoch
                save_on_train_epoch_end=True
            ),


            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ],
        num_sanity_val_steps=0,
        # gradient_clip_val=0.5
    )

    create_result_dir(logger.log_dir + '/config.yaml')
    shutil.copyfile(args.config, logger.log_dir + '/config.yaml')

    args_dict, dataset_name = get_dataset_args_dict(args.dataset, args.dataset_root, seed=cfg.DATASET.SEED)
    args_dict = DictConfig(args_dict)
    datamodule = DataModule(args_dict, dataset_name)

    ckpt_path = args.resume if args.resume else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    import argparse
    from config.default import cfg as default_cfg
    from yacs.config import CfgNode as CN

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/MicKey/curriculum_learning_warm_up.yaml')
    parser.add_argument('--dataset', choices=['Shapenet6D', 'NOCS', 'TOYL'], default='NOCS')
    parser.add_argument('--dataset_root', default='filesOfOryon/data')
    parser.add_argument('--experiment', default='MicKey_default')
    parser.add_argument('--path_weights', default='weights')
    parser.add_argument('--resume', default='')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_cfg = CN.load_cfg(f)
    cfg = default_cfg.clone()
    cfg.merge_from_other_cfg(yaml_cfg)
    args.model = cfg.model

    train_model(args, cfg)
