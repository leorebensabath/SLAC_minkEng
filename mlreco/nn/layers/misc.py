import torch
import torch.nn as nn

# For MinkowskiEngine
import MinkowskiEngine as ME
from SparseTensor import SparseTensor
from .factories import *


# Custom Network Units/Blocks
class Identity(nn.Module):
    def forward(self, input):
        return input


def normalize_coords(coords, spatial_size=512):
    '''
    Utility Method for attaching normalized coordinates to
    sparse tensor features.

    INPUTS:
        - input (scn.SparseConvNetTensor): sparse tensor to
        attach normalized coordinates with range (-1, 1)

    RETURNS:
        - output (scn.SparseConvNetTensor): sparse tensor with 
        normalized coordinate concatenated to first three dimensions.
    '''
    with torch.no_grad():
        coords = coords.float()
        normalized_coords = (coords[:, :3] - spatial_size / 2) \
            / (spatial_size / 2)
        if torch.cuda.is_available():
            normalized_coords = normalized_coords.cuda()
    return normalized_coords


class ConvolutionBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 dimension=3,
                 activation='relu',
                 activation_args={},
                 normalization='batch_norm',
                 normalization_args={}):
        super(ConvolutionBlock, self).__init__()
        assert dimension > 0
        self.act_fn1 = activations_construct(activation, **activation_args)
        self.act_fn2 = activations_construct(activation, **activation_args)

        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm1 = normalizations_construct(
            normalization, out_features, **normalization_args)
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = normalizations_construct(
            normalization, out_features, **normalization_args)

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fn1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act_fn2(out)
        return out


class ResNetBlock(nn.Module):
    '''
    ResNet Block with Leaky ReLU nonlinearities.
    '''
    expansion = 1

    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 dimension=3,
                 activation='relu',
                 activation_args={},
                 normalization='batch_norm',
                 normalization_args={}):
        super(ResNetBlock, self).__init__()
        assert dimension > 0
        self.act_fn1 = activations_construct(activation, **activation_args)
        self.act_fn2 = activations_construct(activation, **activation_args)

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm1 = normalizations_construct(
            normalization, out_features, **normalization_args)
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = normalizations_construct(
            normalization, out_features, **normalization_args)

    def forward(self, x):

        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fn1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.act_fn2(out)
        return out


class AtrousIIBlock(nn.Module):
    '''
    ResNet-type block with Atrous Convolutions, as developed in ACNN paper:
    <ACNN: a Full Resolution DCNN for Medical Image Segmentation>
    Original Paper: https://arxiv.org/pdf/1901.09203.pdf
    '''

    def __init__(self, in_features, out_features, 
                 dimension=3, 
                 activation='relu',
                 activation_args={},
                 normalization='batch_norm',
                 normalization_args={}):
        super(AtrousIIBlock, self).__init__()
        assert dimension > 0
        self.D = dimension
        self.act_fn1 = activations_construct(activation, **activation_args)
        self.act_fn2 = activations_construct(activation, **activation_args)

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features,
            kernel_size=3, stride=1, dilation=1, dimension=self.D)
        self.norm1 = normalizations_construct(
            normalization, out_features, **normalization_args)
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features,
            kernel_size=3, stride=1, dilation=3, dimension=self.D)
        self.norm2 = normalizations_construct(
            normalization, out_features, **normalization_args)

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fn1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.act_fn2(out)
        return out


class ResNeXtBlock(nn.Module):
    '''
    ResNeXt block with leaky relu nonlinearities and atrous convs.

    CONFIGURATIONS:
    -------------------------------------------------------
        - in_features (int): total number of input features

        - out_features (int): total number of output features
          NOTE: if in_features != out_features, then the identity skip
          connection is replaced with a 1x1 conv layer.

        - dimension (int): dimension of dataset.

        - leakiness (float): leakiness for LeakyReLUs.

        - cardinality (int): number of different paths, see ResNeXt paper.

        - depth (int): number of convolutions + BN + LeakyReLU layers inside
        each cardinal path.

        - dilations (int or list of ints): dilation rates for atrous
        convolutions.

        - kernel_sizes (int or list of ints): kernel sizes for each conv layers
        inside cardinal paths.

        - strides (int or list of ints): strides for each conv layers inside
        cardinal paths.
    -------------------------------------------------------
    NOTE: For vanilla resnext blocks, set dilation=1 and others to default.
    '''

    def __init__(self, in_features, out_features, 
                 dimension=3,
                 cardinality=4, 
                 depth=1,
                 dilations=None, 
                 kernel_sizes=3, 
                 strides=1, 
                 activation='relu',
                 activation_args={},
                 normalization='batch_norm',
                 normalization_args={}):
        super(ResNeXtBlock, self).__init__()
        assert dimension > 0
        assert cardinality > 0
        assert (in_features % cardinality == 0 and
                out_features % cardinality == 0)
        self.D = dimension
        nIn = in_features // cardinality
        nOut = out_features // cardinality

        self.dilations = []
        if dilations is None:
            # Default
            self.dilations = [3**i for i in range(cardinality)]
        elif isinstance(dilations, int):
            self.dilations = [dilations for _ in range(cardinality)]
        elif isinstance(dilations, list):
            assert len(dilations) == cardinality
            self.dilations = dilations
        else:
            raise ValueError(
                'Invalid type for input strides, must be int or list!')

        self.kernels = []
        if isinstance(kernel_sizes, int):
            self.kernels = [kernel_sizes for _ in range(cardinality)]
        elif isinstance(kernel_sizes, list):
            assert len(kernel_sizes) == cardinality
            self.kernels = kernel_sizes
        else:
            raise ValueError(
                'Invalid type for input strides, must be int or list!')

        self.strides = []
        if isinstance(strides, int):
            self.strides = [strides for _ in range(cardinality)]
        elif isinstance(strides, list):
            assert len(strides) == cardinality
            self.strides = strides
        else:
            raise ValueError(
                'Invalid type for input strides, must be int or list!')

        # For each path, generate sequentials
        self.paths = []
        for i in range(cardinality):
            m = []
            m.append(ME.MinkowskiLinear(in_features, nIn))
            for j in range(depth):
                in_C = (nIn if j == 0 else nOut)
                m.append(ME.MinkowskiConvolution(
                    in_channels=in_C, out_channels=nOut,
                    kernel_size=self.kernels[i], stride=self.strides[i],
                    dilation=self.dilations[i], dimension=self.D))
                m.append(normalizations_construct(
                    normalization, nOut, **normalization_args))
                m.append(activations_construct(activation, **activation_args))
            m = nn.Sequential(*m)
            self.paths.append(m)
        self.paths = nn.Sequential(*self.paths)
        self.linear = ME.MinkowskiLinear(out_features, out_features)

        # Skip Connection
        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()

    def forward(self, x):
        residual = self.residual(x)
        cat = tuple([layer(x) for layer in self.paths])
        out = ME.cat(cat)
        out = self.linear(out)
        out += residual
        return out


class SPP(nn.Module):
    '''
    Spatial Pyramid Pooling Module.
    PSPNet (Pyramid Scene Parsing Network) uses vanilla SPPs, while
    DeeplabV3 and DeeplabV3+ uses ASPP (atrous versions).

    Default parameters will construct a global average pooling + unpooling
    layer which is done in ParseNet.

    CONFIGURATIONS:
    -------------------------------------------------------
        - in_features (int): number of input features

        - out_features (int): number of output features

        - D (int): dimension of dataset.

        - mode (str): pooling mode. In MinkowskiEngine, currently
        'avg', 'max', and 'sum' are supported.

        - dilations (int or list of ints): dilation rates for atrous
        convolutions.

        - kernel_sizes (int or list of ints): kernel sizes for each
        pooling operation. Note that kernel_size == stride for the SPP layer.
    -------------------------------------------------------
    '''

    def __init__(self, in_features, out_features,
                 kernel_sizes=None, dilations=None, mode='avg', D=3):
        super(SPP, self).__init__()
        if mode == 'avg':
            self.pool_fn = ME.MinkowskiAvgPooling
        elif mode == 'max':
            self.pool_fn = ME.MinkowskiMaxPooling
        elif mode == 'sum':
            self.pool_fn = ME.MinkowskiSumPooling
        else:
            raise ValueError("Invalid pooling mode, must be one of \
                'sum', 'max' or 'average'")
        self.unpool_fn = ME.MinkowskiPoolingTranspose

        # Include global pooling as first modules.
        self.pool = [ME.MinkowskiGlobalPooling(dimension=D)]
        self.unpool = [ME.MinkowskiBroadcast(dimension=D)]
        multiplier = 1
        # Define subregion poolings
        self.spp = []
        if kernel_sizes is not None:
            if isinstance(dilations, int):
                dilations = [dilations for _ in range(len(kernel_sizes))]
            elif isinstance(dilations, list):
                assert len(kernel_sizes) == len(dilations)
            else:
                raise ValueError("Invalid input to dilations, must be either \
                    int or list of ints")
            multiplier = len(kernel_sizes) + 1  # Additional 1 for globalPool
            for k, d in zip(kernel_sizes, dilations):
                pooling_layer = self.pool_fn(
                    kernel_size=k, dilation=d, stride=k, dimension=D)
                unpooling_layer = self.unpool_fn(
                    kernel_size=k, dilation=d, stride=k, dimension=D)
                self.pool.append(pooling_layer)
                self.unpool.append(unpooling_layer)
        self.pool = nn.Sequential(*self.pool)
        self.unpool = nn.Sequential(*self.unpool)
        self.linear = ME.MinkowskiLinear(in_features * multiplier, out_features)

    def forward(self, input):

        cat = []
        for i, pool in enumerate(self.pool):
            x = pool(input)
            # First item is Global Pooling
            if i == 0:
                x = self.unpool[i](input, x)
            else:
                x = self.unpool[i](x)
            cat.append(x)
        out = ME.cat(cat)
        out = self.linear(out)

        return out

class CascadeDilationBlock(nn.Module):
    '''
    Cascaded Atrous Convolution Block
    '''
    def __init__(self, in_features, out_features, 
                 dimension=3,
                 depth=6,
                 dilations=[1, 2, 4, 8, 16, 32],
                 activation='relu',
                 activation_args={}):
        super(CascadeDilationBlock, self).__init__()
        self.D = dimension
        F = out_features
        m = []
        self.input_layer = ME.MinkowskiLinear(in_features, F)
        for i in range(depth):
            m.append(ResNetBlock(F, F, dilation=dilations[i],
            activation=activation, activation_args=activation_args))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        
        x = self.input_layer(x)
        sumTensor = x
        for i, layer in enumerate(self.net):
            x = layer(x)
            sumTensor += x
        return sumTensor

class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module
    '''
    def __init__(self, in_features, out_features,
                 dimension=3,
                 width=5,
                 dilations=[2, 4, 6, 8, 12]):
        assert len(dilations) == width
        m = []
        m.append(ME.MinkowskiLinear(in_features, out_features))
        for d in dilations:
            m.append(ME.MinkowskiConvolution(in_features, out_features, 
                kernel_size=3, dilation=d, dimension=self.D))
        self.net = nn.Sequential(*m)
        self.pool = ME.MinkowskiGlobalPooling(dimension=self.D)
        self.unpool = ME.MinkowskiBroadcast(dimension=self.D)
        self.out = nn.Sequential(
            ME.MinkowskiConvolution(in_features * (2 + width), out_features,
            kernel_size=3, dilation=1, dimension=self.D),
            ME.MinkowskiBatchNorm(out_features),
            ME.MinkowskiReLU()
        )

    def forward(self, x):
        cat = []
        for i, layer in enumerate(self.net):
            x_i = layer(x)
            cat.append(x_i)
        x_global = self.pool(x)
        x_global = self.unpool(x, x_global)
        cat.append(x_global)
        out = ME.cat(cat)
        return self.out(out)


class FeaturePyramidPooling(nn.Module):
    '''
    Feature Pyramid Pooling (FPP) Module (formerly Stacked UNet)

    FPP eats list of sparse tensors at each spatial resolution and unpools
    each sparse tensor to the highest spatial resolution. It then concatenates
    the unpooled features vertically to create a stack of features upsampled
    from every feature tensor in the decoding path. To obtain the final 
    feature tensor, we reduce number of features in the stacked sparse tensor
    through a series of feature reducing network blocks. 

    INPUTS:
        - in_channels: number of total stacked input channels
        - out_channels: desired number of output channels
        - block_mode: Choice of feature reduction blocks
        - depth: depth of feature reduction path
        - 
    '''
    def __init__(self, in_channels, out_channels, 
                 block_mode='resnext', depth=3):
        super(FeaturePyramidPooling, self).__init__()

    def forward(self, featureTensors):
        '''
        INPUTS:
            - featureTensors (list of SparseTensor):
            list of sparse tensors to be stacked and unpooled for FPP. 
        '''
        out = None
        return out


# TODO
class SeparableConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        pass


class DepthwiseConv(nn.Module):

    def __init__(self, in_features, out_features):
        pass


class DenseASPP(nn.Module):

    def __init__(self):
        pass
