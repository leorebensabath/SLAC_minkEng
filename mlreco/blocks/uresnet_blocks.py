import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from torch.optim import SGD
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiModuleBase
import torch.nn.functional as F

class MinkowskiLeakyReLU(MinkowskiModuleBase):
    MODULE = nn.LeakyReLU

class ResNetBlock(ME.MinkowskiNetwork) : 
    
    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 D=3,
                 kernel_size=2) : 
        
        super(ResNetBlock, self).__init__(D)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features, kernel_size=kernel_size,
            stride=stride, dimension=D)
        self.norm_fn1 = ME.MinkowskiBatchNorm(num_features = in_features)
        self.act_fn1 = MinkowskiLeakyReLU()
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features, kernel_size=kernel_size,
            stride=stride, dimension=D)
        self.norm_fn2 = ME.MinkowskiBatchNorm(num_features = out_features)
        self.act_fn2 = MinkowskiLeakyReLU()
        
        self.res = ME.MinkowskiLinear(in_features, out_features)
            
    
    def forward(self, x) :
        
        if self.in_features != self.out_features : 
            res = self.res(x)
        else : 
            res = x 
        out = self.norm_fn1(x)
        out = self.conv1(out)
        out = self.act_fn1(out)
        
        out = self.norm_fn2(out)
        out = self.conv2(out)
        out = self.act_fn2(out + res)
        
        return(out)
    
class Encoder(ME.MinkowskiNetwork) : 
    
    def __init__(self, cfg, name = 'uresnet_encoder') : 
        
        self.model_config = cfg[name]
        self.D = self.model_config.get('D', 3)
        
        super(Encoder, self).__init__(self.D)
        
        self.reps = self.model_config.get('reps', 2)
        self.encoder_depth = self.model_config.get('depth', 7)
        self.encoder_num_filters = self.model_config.get('encoder_num_filters', 16)
        self.kernel_size_res = self.model_config.get('kernel_size_res', 2)
        self.kernel_size_conv = self.model_config.get('kernel_size_conv', 2)
        self.nPlanes = [self.encoder_num_filters*i for i in range(1, self.encoder_depth+1)]
        
        self.encoding_conv = []
        self.encoding_block = []

        for i in range(self.encoder_depth):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i], self.nPlanes[i], 
                                     D=self.D, kernel_size = self.kernel_size_res))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            
            m = []
            if i < self.encoder_depth-1:
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i]))
                
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=self.kernel_size_conv, stride=self.kernel_size_conv, dimension=self.D))
                
                m.append(MinkowskiLeakyReLU())
                
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        
        self.encoding_block = nn.Sequential(*self.encoding_block)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
    
    def forward(self, x) :
        encoding_features = []
        
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoding_features.append(x)
            x = self.encoding_conv[i](x)
        return(x, encoding_features)
    
    
class Decoder(ME.MinkowskiNetwork) : 
    
    def __init__(self, cfg, name = 'uresnet_decoder') :

        self.model_config = cfg[name]
        self.D = self.model_config.get('D', 3)
        super(Decoder, self).__init__(self.D)

        self.reps = self.model_config.get('reps', 2)
        self.decoder_depth = self.model_config.get('depth', 7)
        self.decoder_num_filters = self.model_config.get('num_filters', 16)
        self.kernel_size_res = self.model_config.get('kernel_size_res', 3)
        self.kernel_size_deconv = self.model_config.get('kernel_size_deconv', 2)
        self.nPlanes = [self.decoder_num_filters*i for i in range(1, self.decoder_depth+1)]

        self.decoding_conv = []
        self.decoding_block = []

        
        for i in range(self.decoder_depth) :  
            m = []
            if i > 0 :
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[self.decoder_depth - i]))
                
                m.append(ME.MinkowskiConvolutionTranspose(
                    in_channels=self.nPlanes[self.decoder_depth - i],
                    out_channels=self.nPlanes[self.decoder_depth - 1 - i],
                    kernel_size=self.kernel_size_res, stride=self.kernel_size_deconv, 
                    generate_new_coords = False,
                    dimension=self.D))
                
                m.append(MinkowskiLeakyReLU())
                
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
        
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock((2 if (_==0)&(i>0) else 1)*self.nPlanes[self.decoder_depth - 1 - i], self.nPlanes[self.decoder_depth - 1 - i], D=self.D, kernel_size=self.kernel_size_res))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)
    
    def forward(self, x, encoding_features) :
        for i, layer in enumerate(self.decoding_block):
            x = self.decoding_conv[i](x)
            
            if i > 0 :  
                x = ME.cat(encoding_features[-i-1], x)
            x = self.decoding_block[i](x)
        return(x)