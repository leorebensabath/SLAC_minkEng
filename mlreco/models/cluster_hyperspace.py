import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import MinkowskiEngine as ME

from mlreco.nn.layers.network_base import NetworkBase
# from mlreco.nn.cluster.hyperspace_embeddings import HyperspaceEmbeddings
from mlreco.nn.loss.hyperspace import HypersphereMultiLoss2


class ClusterHyperspace(NetworkBase):
    '''
    Wrapper module for hyperspace clustering.
    '''
    def __init__(self, cfg, name='hyperspace_embeddings'):
        super(ClusterHyperspace, self).__init__(cfg)
        self.net = HyperspaceEmbeddings(cfg)
        print('Total Number of Trainable Parameters = {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            res = self.net(x)
            out['clusterTensors'].append(
                [t.F for t in res['clusterTensors']])
            out['final_embedding'].append(res['final_embedding'].F)
        return out


class ClusteringLoss(HypersphereMultiLoss2):
    '''
    Wrapper module for hyperspace clustering loss
    '''
    def __init__(self, cfg, name='hyperspace_loss'):
        super(ClusteringLoss, self).__init__(cfg)
