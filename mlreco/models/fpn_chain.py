import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.backbone.fpn import FPN
from collections import defaultdict

class FPN_Chain(nn.Module):

    def __init__(self, cfg, name='uresnet_chain'):
        super(FPN_Chain, self).__init__()
        self.model_cfg = cfg
        self.net = FPN(cfg, name='fpn')
        self.F = self.model_cfg.get('num_filters', 16)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.segmentation = ME.MinkowskiLinear(self.F, self.num_classes)

    def forward(self, input):
        out = defaultdict(list)
        num_gpus = len(input)
        for igpu, x in enumerate(input):
            res = self.net(x)
            seg = res['decoderTensors'][-1]
            seg = self.segmentation(seg)
            out['segmentation'].append(seg.F)
        return out