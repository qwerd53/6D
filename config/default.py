
from yacs.config import CfgNode as CN
_CN = CN()

##############  Model    ##############
_CN.MODEL =None
#test

_CN.DEBUG = False

_CN.model = CN()
_CN.model.use_catseg_ckpt = True

_CN.model.image_encoder = CN()
_CN.model.image_encoder.img_size = [192, 192]
_CN.model.image_encoder.out_channels =1024
_CN.model.image_encoder.extra_upsampling =True
_CN.model.image_encoder.vlm = "clip"
_CN.model.image_encoder.use_decoder_guidance = True
_CN.model.image_encoder.use_cost_guidance = True
_CN.model.image_encoder.decoder_type = "standard"

############# TRAINING #############
_CN.TRAINING = CN()
# Data Loader settings
_CN.TRAINING.BATCH_SIZE = None
_CN.TRAINING.NUM_WORKERS = None
_CN.TRAINING.NUM_GPUS = None
_CN.TRAINING.SAMPLER = None  # options: ['random', 'scene_balance']
_CN.TRAINING.N_SAMPLES_SCENE = None  # if 'scene_balance' sampler, the number of samples to get per scene
_CN.TRAINING.SAMPLE_WITH_REPLACEMENT = None  # if 'scene_balance' sampler, whether to sample with replacement

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.DATA_SOURCE = None # options: ['ScanNet', '7Scenes', 'MapFree']
_CN.DATASET.SCENES = None    # scenes to use (for 7Scenes/MapFree); should be a list []; If none, use all scenes.
_CN.DATASET.DATA_ROOT = None   # path to dataset folder
_CN.DATASET.SEED = None   # SEED for dataset generation
_CN.DATASET.NPZ_ROOT = None    # path to npz files containing pairs of frame indices per sample
_CN.DATASET.MIN_OVERLAP_SCORE = None  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MAX_OVERLAP_SCORE = None  # discard data with overlap_score > max_overlap_score
_CN.DATASET.CONSECUTIVE_PAIRS = None  # options: [None, 'colorjitter']
_CN.DATASET.FRAME_RATE = None  # options: [None, 'colorjitter']
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'colorjitter']
_CN.DATASET.BLACK_WHITE = False       # if true, transform images to black & white
_CN.DATASET.PAIRS_TXT = CN()          # Path to text file defining the train/val/test pairs (7Scenes)
_CN.DATASET.PAIRS_TXT.TRAIN = None
_CN.DATASET.PAIRS_TXT.VAL = None
_CN.DATASET.PAIRS_TXT.TEST = None
_CN.DATASET.PAIRS_TXT.ONE_NN = False  # If true, keeps only reference image w/ highest similarity to each query
_CN.DATASET.HEIGHT = None
_CN.DATASET.WIDTH = None
# Training settings
_CN.TRAINING.LR = None
_CN.TRAINING.WEIGHT_DECAY =None
_CN.TRAINING.LR_GAMMA=None  # 学习率缩放因子
_CN.TRAINING.LR_PATIENCE= None
_CN.TRAINING.LR_STEP_INTERVAL = None
_CN.TRAINING.LR_STEP_GAMMA = None      # multiplicative factor of LR every LR_STEP_ITERATIONS
_CN.TRAINING.VAL_INTERVAL = None
_CN.TRAINING.VAL_BATCHES = None
_CN.TRAINING.LOG_INTERVAL = None
_CN.TRAINING.EPOCHS = None
_CN.TRAINING.GRAD_CLIP = 0.   #  Indicates the L2 norm at which to clip the gradient. Disabled if 0

cfg = _CN