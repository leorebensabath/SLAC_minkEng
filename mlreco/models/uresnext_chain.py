import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.backbone.uresnext import UResNeXt
from collections import defaultdict


class UResNeXt_Chain(nn.Module):

    def __init__(self, cfg, name='uresnet_chain'):
        super(UResNeXt_Chain, self).__init__()
        self.model_cfg = cfg
        self.net = UResNeXt(cfg, name='uresnext')
        self.F = self.net.num_filters
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.segmentation = ME.MinkowskiLinear(self.F, self.num_classes)

        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        out = defaultdict(list)
        num_gpus = len(input)
        for igpu, x in enumerate(input):
            res = self.net(x)
            seg = res['decoderTensors'][-1]
            seg = self.segmentation(seg)
            out['segmentation'].append(seg.F)
        return out
