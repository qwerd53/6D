import os
import torch
import sys
#sys.path.append("/data/WDY/Talk2DINO")
sys.path.append(os.getcwd())
import sys
#sys.path.append("/data/WDY/Talk2DINO")  # 添加 Talk2DINO 根目录
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pytorch_lightning as pl
import random
import argparse
from omegaconf import DictConfig
from yacs.config import CfgNode as CN

#from lib.models.MicKey.model_test import MicKeyTrainingModel
#from lib.models.MicKey.model_test_sift import MicKeyTrainingModel
from lib.models.MicKey.model_test_objmatching import MicKeyTrainingModel
#from lib.models.MicKey.model_test_talk2dino import MicKeyTrainingModel
from lib.datasets.datamodules import DataModule


def setup_environment():
    """Set environment variables for optimal performance"""
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def get_dataset_args_dict(dataset_name='NOCS', root_path='filesOfOryon/data', seed=42):
    """Create dataset configuration matching training setup"""
    args_dict = {
        'dataset': {
            'root': root_path,
            #'img_size': [224, 224],
            #'img_size': [480, 640],
            'img_size': [480, 640],
            'max_corrs': 4,
            'test': {'name': 'NOCS', 'split': 'val', 'obj': 'all'}
            #'test': {'name': 'TOYL', 'split': 'cross_scene_test', 'obj': 'all'}
            #'test': {'name': 'YCBV', 'split': 'test', 'obj': 'all'}
            #'test': {'name': 'LM', 'split': 'test', 'obj': 'all'}
        },
        'TRAINING': {
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'SAMPLER': 'scene_balance',
            'N_SAMPLES_SCENE': 4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter': False, 'bright': False, 'hflip': False, 'vflip': False},
            'text': {'synset': False}
            # 'rgb': {'jitter': False, 'bright': False, 'hflip': False, 'vflip': False},
            # 'text': {'synset': False}
        },
        'test': {'mask': 'oracle', 'add_description': 'yes'},
        'use_seed': True,
        'seed': seed,
        'debug_valid': 'anchor'
    }
    return DictConfig(args_dict)


def load_config(config_path, default_cfg):
    """Load and merge config files following training procedure"""
    with open(config_path, "r") as f:
        yaml_cfg = CN.load_cfg(f)
    cfg = default_cfg.clone()
    cfg.merge_from_other_cfg(yaml_cfg)
    return cfg


def test_model(args, cfg):
    """Main testing function using PyTorch Lightning Trainer"""
    # Set random seed
    if cfg.DATASET.get('SEED') is None:
        cfg.DATASET.SEED = random.randint(0, 1000000)

    # Load model checkpoint
    model = MicKeyTrainingModel.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        #map_location='cuda' if torch.cuda.is_available() else 'cpu'
        map_location='cuda:1'
    )
    # model.eval()
    #model = MicKeyTrainingModel(cfg=cfg)
    model.eval()

    # Setup datamodule
    args_dict = get_dataset_args_dict(dataset_name='NOCS', root_path=args.dataset_root, seed=cfg.DATASET.SEED)
    datamodule = DataModule(
        args_dict,
        train_dataset_name='NOCS',  # 占位，不用训练
        #val_dataset_name='TOYL'
        val_dataset_name='NOCS'
       #val_dataset_name='YCBV'
        #val_dataset_name='LM'
    )
    datamodule.setup('test')

    # Setup Trainer
    trainer = pl.Trainer(
        devices=cfg.TRAINING.NUM_GPUS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
        enable_checkpointing=False
    )

    # Run validation to compute ADD(S)-0.1d
    result = trainer.validate(model, datamodule=datamodule, verbose=True)

    # # 输出最终 ADD(S)-0.1d
    # if result and isinstance(result, list) and len(result) > 0:
    #     add_acc = result[0].get('add01d_acc', None)  # 假设 validation_step 返回的 dict 中 key 为 'add01d_acc'
    #     if add_acc is not None:
    #         print(f"\nFinal ADD(S)-0.1d Accuracy: {add_acc:.2f}%")
    #     else:
    #         print("Warning: 'add01d_acc' not found in validation results.")
    # else:
    #     print("Warning: Validation returned empty results.")


if __name__ == '__main__':
    # Setup environment
    setup_environment()
    ##
    #nocs:weights/lofter_Dicemask_shapenet_flip_lr/version_18/checkpoints/epoch=15-best_addval/add01d_acc=40.827.ckpt
    #toyl:weights/lofter_Dicemask_shapenet_flip_lr1e-3_valTOYL/version_2/checkpoints/epoch=5-best_addval/add01d_acc=53.000.ckpt
    #ycbv:weights/ycbv/version_0/checkpoints/epoch=3-best_addval/add01d_acc=0.200.ckpt
    #LM:weights/LM/version_2/checkpoints/epoch=15-best_addval/add01d_acc=9.775.ckpt
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='weights/lofter_Dicemask_shapenet_flip_lr/version_18/checkpoints/epoch=15-best_addval/add01d_acc=40.827.ckpt')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--dataset_root', default='filesOfOryon/data', help='Path to dataset root')
    args = parser.parse_args()

    # Load default config structure
    from config.default import cfg as default_cfg

    # Load and merge configs
    cfg = load_config(args.config, default_cfg)

    # Run testing
    test_model(args, cfg)
