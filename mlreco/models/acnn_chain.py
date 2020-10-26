import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.backbone.acnn import ACNN
from collections import defaultdict

class ACNN_Chain(nn.Module):

    def __init__(self, cfg, name='acnn'):
        super(ACNN_Chain, self).__init__()
        self.model_cfg = cfg['modules'][name]
        self.net = ACNN(cfg, name='acnn')
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.segmentation = ME.MinkowskiLinear(self.net.outputFeatures,
                                               self.num_classes)
        print('Total Number of Trainable Parameters = {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        out = defaultdict(list)
        num_gpus = len(input)
        for igpu, x in enumerate(input):
            res = self.net(x)
            res = self.segmentation(res)
            out['segmentation'].append(res.F)
        return out