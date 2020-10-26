import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from torch.optim import SGD
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiModuleBase
import torch.nn.functional as F

import time 

class ConcatTable(nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)

    def forward(self, input):
        return [module(input) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self
    
class AddTable(nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)

    def forward(self, input):
        coordinates = input[0].coords
        features = sum([i.feats for i in input])
        r = input[0]+input[1]
        #return(ME.SparseTensor(coords = coordinates, feats = features, tensor_stride = input[0].tensor_stride))
        return r
    
    def input_spatial_size(self, out_size):
        return out_size
    
    
class MinkowskiLeakyReLU(MinkowskiModuleBase):
    MODULE = nn.LeakyReLU
    
class Identity(ME.MinkowskiNetwork):
    def forward(self, input):
        return input
    
    
class ParticleImageClassifier(ME.MinkowskiNetwork) :
    
    def __init__(self, cfg, name = "particle_image_classifier_same2") :
        self._model_config = cfg[name]
        self._dimension = self._model_config.get('data_dim', 3)
        super(ParticleImageClassifier, self).__init__(self._dimension)
        
        reps = self._model_config.get('reps', 3)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 32)  # Unet number of features
        nInputFeatures = self._model_config.get('in_features', 4)
        self.spatial_size = self._model_config.get('spatial_size', 1024)
        num_classes = self._model_config.get('num_classes', 5)
        leakiness = self._model_config.get('leakiness', 0.33)
        self.coordConv = self._model_config.get('coordConv', True)
        allow_bias =  self._model_config.get('allow_bias', True)
        
        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        #downsample = [kernel_size, 2]
        downsample = [2, 2]# [filter size, filter stride]
        leakiness = 0
        
        self.input_layer = ME.MinkowskiConvolution(nInputFeatures, m, kernel_size = 3, stride = 1, dimension = self._dimension, has_bias = False)
        
        def block(m, a, b, num):  # ResNet style blocks
            
            module = nn.Sequential(ConcatTable(Identity(self._dimension) if a == b else ME.MinkowskiLinear(a, b), \
nn.Sequential( \
ME.MinkowskiBatchNorm(num_features = a), MinkowskiLeakyReLU(leakiness), \
ME.MinkowskiConvolution(a, b, kernel_size=3, stride=1, dimension=self._dimension, has_bias = allow_bias), \
ME.MinkowskiBatchNorm(num_features = b), MinkowskiLeakyReLU(leakiness), \
ME.MinkowskiConvolution(b, b, kernel_size=3, stride=1, dimension=self._dimension, has_bias = allow_bias)) \
), AddTable())
            m.add_module(f'block_{num}', module)
        
        self.encoding_block = nn.Sequential()
        self.encoding_conv = nn.Sequential()
        
        for i in range(num_strides):
            module = nn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i], _)
            self.encoding_block.add_module(f'encod_block_{i}', module)
            module2 = nn.Sequential()
            if i < num_strides-1:
                module2 = nn.Sequential(ME.MinkowskiBatchNorm(num_features = nPlanes[i]), \
                            MinkowskiLeakyReLU(leakiness), \
                            ME.MinkowskiConvolution(nPlanes[i], nPlanes[i+1], \
                        kernel_size = downsample[0], stride = downsample[1], dimension = self._dimension, has_bias = allow_bias))
            self.encoding_conv.add_module(f'encod_block_conv_{i}', module2)
            
            #self.global_pooling = ME.MinkowskiMaxPooling(kernel_size = 4, dimension = 3, stride = 2)
            self.global_pooling = ME.MinkowskiGlobalPooling(average=True)
            self.bnr = nn.Sequential(ME.MinkowskiBatchNorm(num_features = nPlanes[-1]), ME.MinkowskiReLU(leakiness))
            self.linear = ME.MinkowskiLinear(nPlanes[-1], 5)
            
            
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
        
        out = self.global_pooling(x)

        out = self.linear(out)
        out = {'logits' : out.F}
        return out
        
class ParticleTypeLoss(nn.Module):
    
    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleTypeLoss, self).__init__()
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