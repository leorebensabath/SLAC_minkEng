import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.nn.layers.network_base import NetworkBase
from mlreco.nn.layers.normalizations import MinkowskiAdaIN
from mlreco.nn.layers.factories import *
from mlreco.nn.backbone.uresnet import UResNet

from collections import defaultdict, namedtuple
from scipy.spatial import cKDTree

Logits = namedtuple('Logits', ['batch_id', 'class_id', 'group_id', 'scores'])

class ControllerNet(nn.Module):
    '''
    MLP Network to transform PPN point to AdaIN parameter vector
    '''
    def __init__(self, in_channels, out_channels, depth=3, hidden_dims=None):
        super(ControllerNet, self).__init__()
        modules = []
        if hidden_dims is not None:
            assert (len(hidden_dims) == depth-1)
            dims = [num_input] + hidden_dims + [num_output]
        else:
            dims = [num_input] + [num_output] * depth
        for i in range(depth):
            modules.append(nn.PReLU())
            modules.append(nn.Linear(dims[i], dims[i+1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class RelativeCoordConv(nn.Module):
    '''
    Relative Coordinate Convolution Blocks introduced in AdaptIS paper.
    '''
    def __init__(self, in_channels, out_channels, 
                 dimension=3, 
                 spatial_size=512, 
                 kernel_size=3, 
                 allow_bias=True):
        super(RelativeCoordConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.spatial_size = spatial_size

        self.conv = ME.MinkowskiConvolution(
            in_channels + dimension, 
            out_channels, 
            kernel_size=3, 
            dimension=dimension)

    def forward(self, x, point):
        '''
        INPUTS:
            - x (N x num_input)
            - coords (N x data_dim)
            - point (1 x data_dim)
        '''
        coords = x.C[:, :3].float().cuda()
        point = point.float().cuda()
        normalized_coords = (coords - point) / float(self._spatial_size / 2)
        x.F = torch.cat([x.F, normalized_coords], dim=1)
        x = self.conv(x)
        return x
        
