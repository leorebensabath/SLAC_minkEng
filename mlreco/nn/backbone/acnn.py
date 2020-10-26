import torch
import torch.nn as nn

# For MinkowskiEngine
import MinkowskiEngine as ME

from mlreco.nn.layers.misc import AtrousIIBlock
from mlreco.nn.layers.network_base import NetworkBase

class ACNN(NetworkBase):
    '''
    <ACNN: a Full Resolution DCNN for Medical Image Segmentation>
    Original Paper: https://arxiv.org/pdf/1901.09203.pdf

    The current implementation is a slight variation from the original paper. 
    '''
    def __init__(self, cfg, name='acnn'):
        super(ACNN, self).__init__(cfg)
        model_cfg = cfg['modules'][name]
        # Depth of (RF - 1) / 8 is required for receptive field of "RF".
        self.depth = model_cfg.get('depth', 32)
        self.num_features = model_cfg.get('num_filters', 4)
        self.input_layer = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_features,
            kernel_size=3, stride=1, dimension=self.D)
        acnn = []
        nIn = self.num_features
        for i in range(self.depth):
            nOut = self.num_features * (i // 2 + 1)
            acnn.append(AtrousIIBlock(nIn, nOut, self.D, self.leakiness))
            nIn = nOut
        self.acnn = nn.Sequential(*acnn)
        self.outputFeatures = nOut

    def forward(self, input):
        coords = input[:, 0:self.D+1].cpu().int()
        features = input[:, self.D+1:].float()
        x = ME.SparseTensor(features, coords=coords)
        x = self.input_layer(x)
        out = self.acnn(x)
        return out
