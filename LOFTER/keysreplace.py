import os
import torch
import sys

sys.path.append(os.getcwd())
import pytorch_lightning as pl
import random
import argparse
from omegaconf import DictConfig
from yacs.config import CfgNode as CN
from lib.models.MicKey.model_test import MicKeyTrainingModel
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
            'img_size': [480, 640],
            #'img_size': [224,224],
            'max_corrs': 4,
            #'test': {'name': 'NOCS', 'split': 'val', 'obj': 'all'}
            'test': {'name': 'TOYL', 'split': 'cross_scene_test', 'obj': 'all'}
        },
        'TRAINING': {
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 8,
            'SAMPLER': 'scene_balance',
            'N_SAMPLES_SCENE': 4,
            'SAMPLE_WITH_REPLACEMENT': True
        },
        'augs': {
            'rgb': {'jitter': False, 'bright': False, 'hflip': False, 'vflip': False},
            'text': {'synset': False}
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

    # 1初始化（
    model = MicKeyTrainingModel(cfg=cfg)

    # 加载 ckpt
    ckpt = torch.load(args.checkpoint, map_location="cuda:0")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("model.", "oryon_model."): v for k, v in state_dict.items()}
    #print(list(ckpt["state_dict"].keys())[:50])
    #print(list(model.state_dict().keys())[:50])
    # 取出参数字典
    model_dict = model.state_dict()
    # 加载相同的部分
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    #model.load_state_dict(model_dict,strict=False)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} layers from checkpoint")
    print("Loaded keys:",pretrained_dict)
    model.eval()

    # Setup datamodule
    args_dict = get_dataset_args_dict(dataset_name='NOCS', root_path=args.dataset_root, seed=cfg.DATASET.SEED)
    datamodule = DataModule(
        args_dict,
        train_dataset_name='NOCS',  # 占位，不用训练
        #val_dataset_name='NOCS'
        val_dataset_name='TOYL'
    )
    datamodule.setup('test')

    # Setup Trainer
    trainer = pl.Trainer(
        devices=cfg.TRAINING.NUM_GPUS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
        enable_checkpointing=False
    )

    # Run validation
    result = trainer.validate(model, datamodule=datamodule, verbose=True)


if __name__ == '__main__':
    # Setup environment
    setup_environment()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='weights/lofter_Dicemask_shapenet_flip_lr/version_9/checkpoints/epoch=9-best_addval/add01d_acc=39.667.ckpt')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--dataset_root', default='filesOfOryon/data', help='Path to dataset root')
    args = parser.parse_args()

    # Load default config structure
    from config.default import cfg as default_cfg

    # Load and merge configs
    cfg = load_config(args.config, default_cfg)

    # Run testing
    test_model(args, cfg)
