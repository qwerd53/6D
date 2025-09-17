import argparse
import os
# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402
# import pytorch_lightning as pl
# import torch
# from pytorch_lightning.loggers import TensorBoardLogger
#
# #from config.default import cfg
# #from lib.datasets.datamodules import DataModuleTraining, CustomDataset
# from lib.models.MicKey.model import MicKeyTrainingModel
# from lib.models.MicKey.modules.utils.training_utils import create_exp_name, create_result_dir
# import random
# import shutil
# from lib.models.Oryon.oryon import Oryon
# from torch.utils.data import DataLoader

import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import os, sys
sys.path.append(os.getcwd())
import pytorch_lightning as pl
import torch
import random
import shutil
from pytorch_lightning.loggers import TensorBoardLogger
from lib.models.MicKey.model_train_lofter import MicKeyTrainingModel
from lib.models.MicKey.modules.utils.training_utils import create_exp_name, create_result_dir
from lib.datasets.datamodules import DataModule
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
def get_dataset_args_dict(dataset_name: str, root_path: str, seed: int = 42):
    assert dataset_name in ['Shapenet6D', 'NOCS', 'TOYL'], f"Unsupported dataset: {dataset_name}"
    if dataset_name == 'Shapenet6D':
        obj_id, name = 'all', 'ShapeNet6D'
    elif dataset_name == 'NOCS':
        obj_id, name = 'all', 'NOCS'
        #obj_id, name = '1', 'NOCS'
    elif dataset_name == 'TOYL':
        obj_id, name = 'all', 'TOYL'

    args_dict = {
        'dataset': {
            'root': root_path,
            'img_size': [480, 640],#[256, 256],[480, 640]
            #'img_size': [480,640],  # [
            'max_corrs': 4,
            'train': {'name': 'Shapenet6D', 'split': 'train', 'obj': "all"},

            'train_shapenet': {'name': 'Shapenet6D', 'split': 'train', 'obj': 'all'},
            'train_nocs': {'name': 'NOCS', 'split': 'cross_scene_test', 'obj': 'all'},
            'train_toyl': {'name': 'TOYL', 'split': 'cross_scene_test', 'obj': 'all'},
            #'test': {'name': name, 'split': 'val', 'obj': obj_id}
            'test': {'name': 'NOCS', 'split': 'val', 'obj': 'all'} #nocs
            #'test': {'name': 'TOYL', 'split': 'cross_scene_test', 'obj': 'all'}
        },
        'TRAINING': {
            'BATCH_SIZE':4 , #8,
            'NUM_WORKERS':1,  #16,
            'SAMPLER': 'scene_balance',
            # 'N_SAMPLES_SCENE': 100,
            'N_SAMPLES_SCENE':4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter': True,
                    'bright': True,
                    'hflip': True,
                    'vflip': True},
            'text': {'synset': True}
        },
        # 'train_augs': {
        #     'rgb': {'jitter': True,
        #             'bright': True,
        #             'hflip': True,
        #             'vflip': True},
        #     'text': {'synset': True}
        # },
        # #
        # 'test_augs': {
        #     'rgb': {'jitter': False,
        #             'bright': False,
        #             'hflip': False,
        #             'vflip': False},
        #     'text': {'synset': False}
        # },
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
    #exp_name = create_exp_name(args.experiment, cfg)
    #print('Start training of', exp_name)

    trainer = pl.Trainer(
        #strategy=DDPStrategy(find_unused_parameters=False),
        precision="32",
        devices=cfg.TRAINING.NUM_GPUS,
        log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
        val_check_interval=1.0,
        limit_val_batches=cfg.TRAINING.VAL_BATCHES,
        max_epochs=cfg.TRAINING.EPOCHS,
        logger=logger,
        callbacks=[
            # pl.callbacks.ModelCheckpoint(
            #     filename='{epoch}-best_iou{val/mask_iou:.3f}',
            #     monitor='val/mask_iou',
            #     mode='max',
            #     save_top_k=1,
            #     verbose=True
            # ),

            pl.callbacks.ModelCheckpoint(
                filename='{epoch}-best_add{val/add01d_acc:.3f}',
                monitor='val/add01d_acc',
                mode='max',
                save_top_k=1,
                verbose=True
            ),
            # pl.callbacks.ModelCheckpoint(
            #     filename='e{epoch}-last',
            #     every_n_epochs=1,
            #     save_top_k=1,
            #     save_on_train_epoch_end=True
            # ),
            pl.callbacks.ModelCheckpoint(
                filename='e{epoch}-every1',
                every_n_epochs=1,
                save_top_k=-1,  # -1 不覆盖
                save_on_train_epoch_end=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ],
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=True)
    )

    create_result_dir(logger.log_dir + '/config.yaml')
    shutil.copyfile(args.config, logger.log_dir + '/config.yaml')

    args_dict, dataset_name = get_dataset_args_dict(args.dataset, args.dataset_root, seed=cfg.DATASET.SEED)
    args_dict = DictConfig(args_dict)
    #datamodule = DataModule(args_dict, dataset_name)
    #shapenet6D，NOCS
    datamodule = DataModule(
        args_dict,
        train_dataset_name='Shapenet6D',
        val_dataset_name='NOCS'
        #val_dataset_name='TOYL'
    )

    ckpt_path = args.resume if args.resume else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    import argparse
    from config.default_lofter import cfg as default_cfg
    from yacs.config import CfgNode as CN

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--dataset', choices=['Shapenet6D', 'NOCS', 'TOYL'], default='Shapenet6D')
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
