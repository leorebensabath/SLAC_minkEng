import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from torch.optim import SGD
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiModuleBase
import torch.nn.functional as F

import time 

class MinkowskiLeakyReLU(MinkowskiModuleBase):
    MODULE = nn.LeakyReLU

class Block(ME.MinkowskiNetwork):

    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 dimension=3,
                 normalization='batch_norm',
                 normalization_args={}):
        super(Block, self).__init__(dimension)
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm_fn1 = ME.MinkowskiBatchNorm(num_features = out_features)
        self.act_fn1 = MinkowskiLeakyReLU()
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm_fn2 = ME.MinkowskiBatchNorm(num_features = out_features)
        self.act_fn2 =MinkowskiLeakyReLU()
        
    def forward(self, x):
        out = self.norm_fn1(x)
        out = self.act_fn1(out)
        out = self.conv1(out)
        out = self.norm_fn2(out)
        out = self.act_fn2(out)
        out = self.conv2(out)
        return out

class SingleParticleNetwork(ME.MinkowskiNetwork) :

    def __init__(self, cfg, name = 'single_particule_classifier_bis') :
        D = 3
        super(SingleParticleNetwork, self).__init__(D)
        in_feat = 1
        
        self.num_filters = 16
        self.depth = 7
        self.nPlanes = [self.num_filters*i for i in range(1, self.depth+1)]
        self.activation_name = 'lrelu'
        self.activation_args = {}
        self.reps = 2
        self.num_features = 5
        self.coordConv = False
        self.spatial_size = 768
        
        if self.coordConv : 
            self.input_layer = ME.MinkowskiConvolution(
                in_channels=in_feat + D,
                out_channels=self.num_filters,
                kernel_size=7, stride=1, dimension=D)
        else : 
            self.input_layer = ME.MinkowskiConvolution(
                in_channels = in_feat,
                out_channels = self.num_filters,
                kernel_size = 3, stride=1, dimension = D)

        #self.output = scn.SparseToDense(self.dimension, self.nPlanes[-1])
        
         # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []

        for i in range(self.depth):
            m = []
            for _ in range(self.reps):
                m.append(Block(self.nPlanes[i], self.nPlanes[i],
                    dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_block.append(m.cuda())

            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i]))
                m.append(MinkowskiLeakyReLU())
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m.cuda())
            
        self.pool = ME.MinkowskiGlobalMaxPooling()                
        self.norm_fcn3 = ME.MinkowskiBatchNorm(num_features = self.nPlanes[-1])
        self.norm_fcn4 = ME.MinkowskiBatchNorm(num_features = 10)
        self.linear = ME.MinkowskiLinear(10, self.num_features)
        self.maxPool = ME.MinkowskiMaxPooling(kernel_size=2, stride = 2, dimension=3)
        self.act_fn = MinkowskiLeakyReLU()
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=self.nPlanes[-1], out_channels=self.nPlanes[-1],
            kernel_size=10, stride=10, dimension=self.D)
        
        self.conv4 = ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1], out_channels = 10,
                kernel_size=1, stride=1, dimension=self.D)
        self.max_pool = ME.MinkowskiMaxPooling(kernel_size=4, stride = 4, dimension=3)
        self.global_pooling = ME.MinkowskiGlobalPooling(average=True)
        
    def forward(self, x) :

        coords = x[0][:, 0:4].float()
        feats = x[0][:, 4].float().reshape([-1, 1])

        if self.coordConv:
            normalized_coords = (coords[:, 1:4] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
            coords = torch.reshape(coords[:, 0], (-1, 1))
            
            feats = torch.cat([normalized_coords, feats], dim=1)
            normalized_coords = torch.cat([coords, normalized_coords], dim=1)           
            x = ME.SparseTensor(feats = feats, coords=normalized_coords)
            
        else : 
            x = ME.SparseTensor(feats = feats, coords=coords)

        x = self.input_layer(x)
        
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            x = self.encoding_conv[i](x)
        
        #out = self.pool(x)
        out = self.norm_fcn3(x)
        out = self.act_fn(out)
        out = self.conv3(out)
        out = self.norm_fcn3(out)
        out = self.act_fn(out)
        out = self.conv4(out)
        out = self.global_pooling(out)
        out = self.norm_fcn4(out)
        out = self.act_fn(out)
        out = self.linear(out)
        out = {'logits' : out.F}
#        out = self.maxPool(x)
#        out = MF.relu(out)
#        out = self.norm_fcn3(out)
#        out = self.linear(out)
        return(out)

class SingleParticleLoss(nn.Module) : 
    
    def __init__(self, cfg, name='particle_type_loss_bis'):
        super(SingleParticleLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, out, type_labels):

        logits = out['logits']

        labels = torch.tensor(type_labels[0]).cuda().to(dtype=torch.long)

        loss = self.xentropy(logits, labels)
        pred = torch.argmax(logits, dim=1)

        accuracy = float(torch.sum(pred == labels)) / float(labels.shape[0])

        res = {
            'loss': loss,
            'accuracy': accuracy
        }
        acc_types = {}
        
        for c in labels.unique():
            mask = labels == c
            acc_types['accuracy_{}'.format(int(c))] = \
                float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask))
        return res