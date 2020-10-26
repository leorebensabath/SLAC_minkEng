import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.backbone.factories import backbone_dict, backbone_construct
from mlreco.nn.layers.factories import activations_dict, activations_construct
from mlreco.nn.layers.misc import *
from mlreco.nn.layers.network_base import NetworkBase

class GaussianMixture(NetworkBase):
    '''
    Sparse version of the SpatialEmbeddings clustering network.
    '''
    def __init__(self, cfg, name='gaussian_mixtures'):
        super(GaussianMixture, self).__init__(cfg)
        self.model_config = cfg['modules'][name]

        # Define Backbone Network
        self.backbone_name = self.model_config.get('backbone_name', 'uresnet')
        net_constructor = backbone_construct(self.backbone_name)
        self.net = net_constructor(cfg)
        self.nPlanes = self.net.nPlanes
        self.num_filters = self.net.num_filters
        self.depth = self.net.depth
        self.reps = self.net.reps
        self.num_classes = self.model_config.get('num_classes', 5)

        # Define Embeddings
        self.sigma_dim = self.model_config.get('sigma_dim', 1)
        self.embedding_dim = self.D
        # Spherical, Axis-aligned Ellipsoidal, Fully Ellipsoidal
        assert self.sigma_dim in set([1, self.embedding_dim, self.embedding_dim**2])

        # Clustering Decoder
        self.clusterDecoderBlocks = []
        self.clusterDecoderConvs = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            m = nn.Sequential(*m)
            self.clusterDecoderConvs.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                    self.nPlanes[i],
                                     # dilations=[1, 1, 1, 1, 2, 2, 4, 4],
                                     # cardinality=8,
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.clusterDecoderBlocks.append(m)
        self.clusterDecoderBlocks = nn.Sequential(*self.clusterDecoderBlocks)
        self.clusterDecoderConvs = nn.Sequential(*self.clusterDecoderConvs)


        # Lateral Connections from encoder features to cluster decoding path.
        self.lateral_type = self.model_config.get('lateral_type', 'resnet')
        self.lateral = []
        for i, F in enumerate(self.nPlanes[-2::-1]):
            m = []
            if self.lateral_type == 'resnext':
                self.cluster_reps = self.model_config.get(
                    'lateral_block_repetitions', 2)
                for j in range(self.cluster_reps):
                    m.append(ResNeXtBlock(F, F,
                        dimension=self.D,
                        cardinality=4,
                        dilations=[1, 1, 3, 9],
                        activation=self.activation_name,
                        activation_args=self.activation_args))
            elif self.lateral_type == 'resnet':
                self.cluster_reps = self.model_config.get(
                    'lateral_block_repetitions', 2)
                for j in range(self.cluster_reps):
                    m.append(ResNetBlock(F, F,
                        dimension=self.D,
                        activation=self.activation_name,
                        activation_args=self.activation_args))
            elif self.lateral_type == 'nin':
                m.append(ME.MinkowskiLinear(F, F))
            elif self.lateral_type == 'identity':
                m.append(Identity())
            else:
                raise ValueError
            m = nn.Sequential(*m)
            self.lateral.append(m)
        self.lateral = nn.Sequential(*self.lateral)

        # SPP
        # self.spp = SPP(self.num_filters, self.num_filters,
        #     kernel_sizes=[32, 64, 128, 256], dilations=1)

        # The last layer in the clustering decoder has total of 32 features,
        # 16 (segmentation features) + 16 (cluster decoder) + spp (32).

        # Seediness
        self.seediness = ME.MinkowskiLinear(self.num_filters, 1)

        # Embeddings
        self.embeddings = ME.MinkowskiLinear(
            self.num_filters, self.D + self.sigma_dim)

        # print(self)


    def cluster_decoder(self, final, encoderTensors):
        clusterTensors = []
        x = final
        for i, layer in enumerate(self.clusterDecoderConvs):
            lateral = self.lateral[i](encoderTensors[-i-2])
            x = layer(x)
            x = ME.cat((lateral, x))
            x = self.clusterDecoderBlocks[i](x)
            clusterTensors.append(x)
        return clusterTensors


    def forward(self, input):
        res_segment = self.net(input)
        clusterTensors = self.cluster_decoder(
            res_segment['finalTensor'], res_segment['encoderTensors'])
        # F = self.spp(clusterTensors[-1])
        F = clusterTensors[-1]

        seediness_features = self.seediness(res_segment['decoderTensors'][-1])
        embedding_features = self.embeddings(F)

        # NOTE: We still have to apply additional nonlinearities and
        # coordinate concatenation schemes to the features.

        res = {
            'seediness': seediness_features,
            'embeddings': embedding_features,
        }

        return res
