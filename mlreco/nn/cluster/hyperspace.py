import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from mlreco.nn.backbone.factories import *
from mlreco.nn.layers.factories import *
from mlreco.nn.layers.network_base import NetworkBase
from mlreco.nn.layers.misc import normalize_coords

class HyperspaceEmbeddings(NetworkBase):
    '''
    Eats backbone and attaches separate clustering branch
    '''
    def __init__(self, cfg, name='hyperspace_embeddings'):
        super(HyperspaceEmbeddings, self).__init__(cfg)
        self.model_config = cfg['modules'][name]

        # Construct Backbone
        self.net = backbone_construct()
        self.num_filters = self.net.num_filters

        # Clustering specific additions
        self.hyperspace_dim = self.model_config.get('hyperspace_dim', 8)
        self.coord_conv = self.model_config.get('coord_conv', True)
        self.block_mode = self.model_config.get('block_mode', 'resnet')
        self.cluster_convs = []
        self.cluster_blocks = []
        self.nPlanes = self.net.nPlanes[::-1]
        self.N = self.model_config.get('N', 3)

        for i, F in enumerate(self.nPlanes):
            if i < len(self.nPlanes)-1
                m = nn.Sequential(
                    normalizations_construct(self.norm, F, **self.norm_args),
                    activations_construct(self.act, **self.act_args),
                    ME.MinkowskiConvolutionTranspose(
                        in_channels=self.nPlanes[i+1],
                        out_channels=self.nPlanes[i],
                        kernel_size=2,
                        stride=2,
                        dimension=self.D))
                self.cluster_convs.append(m)
            m = []
            for j in range(self.N):
                num_input = F
                if i > 0 and j == 0:
                    num_input *= 2
                if self.coord_conv and j == 0:
                    num_inpnut += self.D
                m.append(ResNetBlock(
                    num_input, F,
                    dimension=self.D,
                    activation=self.act,
                    activation_args=self.act_args))
            m = nn.Sequential(*m)
            self.cluster_blocks.append(m)
        self.cluster_convs = nn.Sequential(self.cluster_convs)
        self.cluster_blocks = nn.Sequential(self.cluster_blocks)

        self.final_embedding = ResNetBlock(
            self.num_filters, self.hyperspace_dim,
            dimension=self.D,
            activation=self.act,
            activation_args=self.act_args)

        print(self)


    def cluster_decoder(self, final, decoderTensors):
        '''
        Feature Pyramid Decoder for clustering coordinates
        '''
        clusterTensors = []
        x = final
        for i, layer in enumerate(self.cluster_convs):
            if self.coord_conv:
                coords = normalize_coords(
                    x.C, spatial_size=self.spatial_size)
                x.F = torch.cat((x.F, coords), dim=1)
            x = self.cluster_blocks[i](x)
            clusterTensors.append(x)
            x = layer(x)
            x = ME.cat((x, decoderTensors[-1-i]))
        x = self.cluster_blocks[-1](x)
        clusterTensors.append(x)
        return clusterTensors


    def forward(self, input):
        res_backbone = self.net(input)
        clusterTensors = self.cluster_decoder(
            res_backbone['finalTensor'], res_segment['encoderTensors'])
        final_embedding = self.final_embedding(clusterTensors[-1])
        
        res = {
            'clusterTensors': clusterTensors,
            'final_embedding': final_embedding
        }

        return res
