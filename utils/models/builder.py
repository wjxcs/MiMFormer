'''
Function:
    builder for different models
Author:
    Zhenchao Jin
'''
from .ce2p import CE2P
from .icnet import ICNet
from .isnet import ISNet
from .danet import DANet
from .gcnet import GCNet
from .dmnet import DMNet
from .isanet import ISANet
from .encnet import ENCNet
from .apcnet import APCNet
from .emanet import EMANet
from .pspnet import PSPNet
from .ocrnet import OCRNet
from .dnlnet import DNLNet
from .annnet import ANNNet
from .fastfcn import FastFCN
from .upernet import UPerNet
from .deeplabv3 import Deeplabv3
from .lrasppnet import LRASPPNet
from .segformer import Segformer
from .memorynet import MemoryNet
from .setr import SETRUP, SETRMLA
from .semanticfpn import SemanticFPN
from .nonlocalnet import NonLocalNet
from .deeplabv3plus import Deeplabv3Plus
from .fcn import FCN, DepthwiseSeparableFCN


'''build model'''
def BuildModel(cfg, mode, **kwargs):
    supported_models = {
        'fcn': FCN,
        'ce2p': CE2P,
        'icnet': ICNet,
        'isnet': ISNet,
        'danet': DANet,
        'gcnet': GCNet,
        'dmnet': DMNet,
        'isanet': ISANet,
        'encnet': ENCNet,
        'apcnet': APCNet,
        'emanet': EMANet,
        'pspnet': PSPNet,
        'ocrnet': OCRNet,
        'dnlnet': DNLNet,
        'annnet': ANNNet,
        'setrup': SETRUP,
        'setrmla': SETRMLA,
        'fastfcn': FastFCN,
        'upernet': UPerNet,
        'segformer': Segformer,
        'memorynet': MemoryNet,
        'deeplabv3': Deeplabv3,
        'lrasppnet': LRASPPNet,
        'semanticfpn': SemanticFPN,
        'nonlocalnet': NonLocalNet,
        'deeplabv3plus': Deeplabv3Plus,
        'depthwiseseparablefcn': DepthwiseSeparableFCN,
    }
    model_type = cfg['type']
    assert model_type in supported_models, 'unsupport model_type %s...' % model_type
    return supported_models[model_type](cfg, mode=mode)