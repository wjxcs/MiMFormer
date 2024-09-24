'''
Function:
    build normalization
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from .layernorm import LayerNorm
from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from .syncbatchnorm import TORCHSyncBatchNorm


'''build normalization'''
def BuildNormalization(norm_type='batchnorm2d', instanced_params=(0, {}), only_get_all_supported=False, **kwargs):
    supported_dict = {
        'layernorm': LayerNorm,
        'identity': nn.Identity,
        'batchnorm1d': BatchNorm1d,
        'batchnorm2d': BatchNorm2d,
        'batchnorm3d': BatchNorm3d,
        'syncbatchnorm': TORCHSyncBatchNorm
    }
    if only_get_all_supported: return list(supported_dict.values())
    assert norm_type in supported_dict, 'unsupport norm_type %s...' % norm_type
    norm_layer = supported_dict[norm_type](instanced_params[0], **instanced_params[1])
    if norm_type in ['syncbatchnorm_mmcv']: norm_layer._specify_ddp_gpu_num(1)
    return norm_layer