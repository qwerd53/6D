import torch.cuda
#
from lib.models.MicKey.compute_pose import MickeyRelativePose

def build_model(cfg, checkpoint=''):

    if cfg.MODEL == 'MicKey':

        model = MickeyRelativePose(cfg)

        checkpoint_loaded = torch.load(checkpoint)
        model.on_load_checkpoint(checkpoint_loaded)
        model.load_state_dict(checkpoint_loaded['state_dict'])

        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    else:
        raise NotImplementedError()
# from lib.models.MicKey.compute_pose import MickeyRelativePose
#
# def build_model(cfg, checkpoint=''):
#     if cfg.MODEL == 'MicKey':
#         model = MickeyRelativePose.load_from_checkpoint(checkpoint_path=checkpoint, cfg=cfg, strict=False)
#
#         if torch.cuda.is_available():
#             model = model.cuda()
#         model.eval()
#         return model
#     else:
#         raise NotImplementedError()
