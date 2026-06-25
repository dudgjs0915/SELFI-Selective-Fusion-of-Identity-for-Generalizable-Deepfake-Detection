'''
# author: Younghun Kim
# email: younghun1664@kaist.ac.kr
# date: 2025-0320

The code is for IResNet backbone.
'''
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.registry import BACKBONE
from timm.models.registry import register_model
from .iresnet_network.iresnet import (
    iresnet100
)

logger = logging.getLogger(__name__)

@BACKBONE.register_module(module_name="selfi_iresnet100")
class IResNet(nn.Module):
    def __init__(self, pretrained_path):
        super(IResNet, self).__init__()
        """ Constructor
        Args:
            iresnet_config: configuration file with the dict format
        """                  
        if pretrained_path:
            self.iresnet = iresnet100()
            self.iresnet.load_state_dict(torch.load(pretrained_path), strict=True)
        else:
            self.iresnet = iresnet100()
            logger.warning("Pretrained weights for IResNet not found. Using random initialization.")

    def features(self, inp):
        x = self.iresnet(inp)
        return x
        