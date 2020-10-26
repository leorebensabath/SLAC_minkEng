import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.layers.misc import *
from mlreco.nn.layers.nonlinearities import MinkowskiLeakyReLU
from mlreco.nn.layers.network_base import NetworkBase
from .uresnext import UResNeXt


class DeepLabUNet(UResNeXt):
    '''
    DeepLabUNet, an extension of UResNeXt with an SPP module
    in the last layer.
    '''

    def __init__(self, cfg, name='deeplab_unet'):
        super(DeepLabUNet, self).__init__(cfg, name=name)
        self.model_config = cfg['modules'][name]

        # Configurations
        self.reps = self.model_cfg.get('reps', 2)
        self.depth = self.model_cfg.get('depth', 5)
        self.num_filters = self.model_cfg.get('num_filters', 16)
        self.input_kernel = self.model_cfg.get('input_kernel', 3)

        self.spp = SPP(16, 16, kernel_sizes=[32, 64, 128, 256], dilations=1)

    def forward(self, input):
        coords = input[:, 0:self.D + 1].cpu().int()
        features = input[:, self.D + 1:].float()

        x = ME.SparseTensor(features, coords=coords)
        encoderOutput = self.encoder(x)
        decoderTensors = self.decoder(
            encoderOutput['finalTensor'], encoderOutput['encoderTensors'])
        sppFeatures = self.spp(decoderTensors[-1])
        finalFeatures = ME.cat((decoderTensors[-1], sppFeatures))

        res = {
            'encoderTensors': encoderOutput['encoderTensors'],
            'decoderTensors': decoderTensors,
            'finalFeatures': finalFeatures
        }
        return res
