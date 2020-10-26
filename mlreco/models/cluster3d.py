import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.nn.layers.factories import *
from mlreco.nn.layers.network_base import NetworkBase
from mlreco.nn.layers.misc import ResNetBlock, ConvolutionBlock

from mlreco.nn.loss.lovasz import *


def get_target(out, target_key, kernel_size=1):
    with torch.no_grad():
        target = torch.zeros(len(out), dtype=torch.bool)
        cm = out.coords_man
        strided_target_key = cm.stride(
            target_key, out.tensor_stride[0], force_creation=True)
        ins, outs = cm.get_kernel_map(
            out.coords_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1)
        for curr_in in ins:
            target[curr_in] = 1
    return target


class Cluster3d(NetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction. 
    '''
    def __init__(self, cfg, name='cluster3d'):
        super(Cluster3d, self).__init__(cfg)

        # ----------------Configurations----------------------

        self.model_config = cfg['modules'][name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        print(self.nPlanes)
        self.input_kernel = self.model_config.get('input_kernel', 3)

        # --------------- LAYER DEFINITIONS -------------------

        # self.norm_layer = normalizations_construct(self.norm, **self.norm_args)

        # Initialize Input Layer
        self.input_layer1 = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=2)

        self.input_layer2 = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=2)

        # Initialize Encoder XY (2D)
        self.encoding_conv1 = []
        self.encoding_block1 = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension=2,
                    activation=self.activation_name,
                    activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.encoding_block1.append(m)
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=2))
            m = nn.Sequential(*m)
            self.encoding_conv1.append(m)
        self.encoding_conv1 = nn.Sequential(*self.encoding_conv1)
        self.encoding_block1 = nn.Sequential(*self.encoding_block1)

        # Initialize Encoder XZ (2D)
        self.encoding_conv2 = []
        self.encoding_block2 = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ConvolutionBlock(F, F,
                    dimension=2,
                    activation=self.activation_name,
                    activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.encoding_block2.append(m)
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=2))
            m = nn.Sequential(*m)
            self.encoding_conv2.append(m)
        self.encoding_conv2 = nn.Sequential(*self.encoding_conv2)
        self.encoding_block2 = nn.Sequential(*self.encoding_block2)

        # Active site predictions
        self.pred_active = []

        # Initialize 3D Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                generate_new_coords=True,
                stride=2,
                dimension=3
                ))
            self.pred_active.append(ME.MinkowskiConvolution(
                self.nPlanes[i], 1, kernel_size=1, has_bias=True, dimension=3))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ConvolutionBlock(self.nPlanes[i],
                                     self.nPlanes[i],
                                     dimension=3,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)
        self.pred_active = nn.Sequential(*self.pred_active)

        # Output Layer for energy regression
        self.output_layer = ConvolutionBlock(
            self.nPlanes[0], 1, dimension=3, 
            activation=self.activation_name, activation_args=self.activation_args)

        # Pruning (Sparsification Layers)
        self.pruning = ME.MinkowskiPruning()
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))

        # Pooling mode (learnable vs. Global Average Pooling

        self.pool_mode = self.model_config.get('pool_mode', 'cavg')

        if self.pool_mode == 'conv':

            self.global_pool1 = ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1], out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape, stride=final_tensor_shape, dimension=2)

            self.global_pool2 = ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1], out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape, stride=final_tensor_shape, dimension=2)

            self.global_unpool = ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[-1],
                out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape,
                generate_new_coords=True,
                stride=final_tensor_shape,
                dimension=3)

        else:

            self.global_pool1 = ME.MinkowskiGlobalPooling()
            self.global_pool2 = ME.MinkowskiGlobalPooling()

            self.global_unpool = ME.MinkowskiPoolingTranspose(
                kernel_size=final_tensor_shape,
                stride=final_tensor_shape,
                dimension=3)

        self.latent_size = self.model_config.get('latent_size', 256)

        self.linear_xy_enc = nn.Sequential(
            ME.MinkowskiBatchNorm(self.nPlanes[-1]),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1], out_channels = self.latent_size,
                kernel_size=1, stride=1, dimension=2
            )
        )

        self.linear_enc = nn.Sequential(
            ME.MinkowskiBatchNorm(self.nPlanes[-1] * 2),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1] * 2, out_channels = self.latent_size,
                kernel_size=1, stride=1, dimension=2
            )
        )

        self.linear_dec = nn.Sequential(
            ME.MinkowskiBatchNorm(self.latent_size),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolution(
                in_channels=self.latent_size, out_channels = self.nPlanes[-1],
                kernel_size=1, stride=1, dimension=3
            )
        )

        self.thresholding_score = self.model_config.get('thresholding_score', 1)

        self.union = ME.MinkowskiUnion()

        print(self)


    def encoder1(self, x):
        '''
        Vanilla UResNet Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        '''
        x = self.input_layer1(x)
        for i, layer in enumerate(self.encoding_block1):
            x = self.encoding_block1[i](x)
            x = self.encoding_conv1[i](x)
        return x


    def encoder2(self, x):
        '''
        Vanilla UResNet Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        '''
        x = self.input_layer2(x)
        for i, layer in enumerate(self.encoding_block2):
            x = self.encoding_block2[i](x)
            x = self.encoding_conv2[i](x)
        return x


    def encoder(self, input_xy, input_xz):

        # Encoder
        encode_xy = self.encoder1(input_xy)
        encode_xz = self.encoder2(input_xz)

        # FC Layers
        z_xy = self.global_pool1(encode_xy)
        z_xz = self.global_pool2(encode_xz)
        z = ME.cat(z_xy, z_xz)
        latent = self.linear_enc(z)
        return latent


    def decoder(self, latent, target_key=None):
        '''
        Vanilla UResNet Decoder
        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''
        z = self.linear_dec(latent)
        x = self.global_unpool(z)
        # print("Global Unpool = ", x)
        decoderTensors = []
        voxel_predictions = []
        targets = []
        for i, layer in enumerate(self.decoding_conv):
            x = layer(x)
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
            pred = self.pred_active[i](x)
            # print(pred.C)
            mask = (pred.F > self.thresholding_score).cpu().squeeze()
            if self.training:
                target = get_target(x, target_key)
                targets.append(target)
                mask += target
                # print("Depth = {}".format(i), x)
            x = self.pruning(x, mask)
            # print("Depth = {}".format(i), x)
            voxel_predictions.append(pred)
        return decoderTensors, voxel_predictions, targets


    def forward(self, input):

        coords_xy = input[0][:, [0, 1, 2]].cpu().int()
        coords_xz = input[0][:, [0, 1, 3]].cpu().int()

        coords = input[0][:, 0:self.D+1].cpu().int()
        features = input[0][:, -1].view(-1, 1).float()

        input_xy = ME.SparseTensor(features, coords=coords_xy)
        input_xz = ME.SparseTensor(features, coords=coords_xz, 
            coords_manager=input_xy.coords_man, force_creation=True)

        latent = self.encoder(input_xy, input_xz)
        latent_coords, latent_feats = latent.C, latent.F
        latent_coords = torch.cat(
            [latent_coords, torch.zeros(latent_coords.shape[0], 1).int()], dim=1)

        if self.training:
            truth_3d = ME.SparseTensor(features, coords=coords)
            cm = truth_3d.coords_man
            input_generator = ME.SparseTensor(
                latent_feats, coords=latent_coords, 
                tensor_stride=self.spatial_size, coords_manager=cm)
            target_key = cm.create_coords_key(
                coords,
                force_creation=True,
                allow_duplicate_coords=True)
            decoderTensors, voxel_predictions, targets = \
                self.decoder(input_generator, target_key)
        else:
            input_generator = ME.SparseTensor(
                latent_feats, coords=latent_coords, 
                tensor_stride=self.spatial_size)
            decoderTensors, voxel_predictions, _ = self.decoder(input_generator)
        out = self.output_layer(decoderTensors[-1])

        res = {
            'voxel_predictions': [voxel_predictions],
            'latent': [latent],
            'out': [out],
        }

        # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/output3d', out.F.detach().cpu().numpy())
        # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/outputCoords', out.C.detach().cpu().numpy())
        # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/input', input[0].detach().cpu().numpy())

        # for i, t in enumerate(voxel_predictions):
        #     np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/intermedFeats_{}.npy'.format(i), t.F.detach().cpu().numpy())
        #     np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/intermedCoords_{}.npy'.format(i), t.C.detach().cpu().numpy())

        if self.training:
            out_key = ME.CoordsKey(cm.D)
            ins, outs = cm.get_union_map(
                (out.coords_key, truth_3d.coords_key), out_key)
            N = cm.get_coords_size_by_coords_key(out_key)
            diff = torch.zeros((N, out.F.size(1)), dtype=out.dtype).cuda()
            diff[outs[0]] = out.F[ins[0]]
            diff[outs[1]] -= truth_3d.F[ins[1]]
            res['targets'] = [targets]
            res['diff'] = [diff]

        return res


class Cluster3dResidual(NetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction. 
    '''
    def __init__(self, cfg, name='cluster3d'):
        super(Cluster3dResidual, self).__init__(cfg)

        self.model_config = cfg['modules'][name]
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        print(self.nPlanes)
        self.input_kernel = self.model_config.get('input_kernel', 3)

        # --------------- LAYER DEFINITIONS -------------------

        # Initialize Input Layer
        self.input_layer1 = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=2)

        self.input_layer2 = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=2)


        # Initialize Encoder XY (2D)
        self.encoding_conv1 = []
        self.encoding_block1 = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension=2,
                    activation=self.activation_name,
                    activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.encoding_block1.append(m)
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=2))
            m = nn.Sequential(*m)
            self.encoding_conv1.append(m)
        self.encoding_conv1 = nn.Sequential(*self.encoding_conv1)
        self.encoding_block1 = nn.Sequential(*self.encoding_block1)

        # Initialize Encoder XZ (2D)
        self.encoding_conv2 = []
        self.encoding_block2 = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension=2,
                    activation=self.activation_name,
                    activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.encoding_block2.append(m)
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=2))
            m = nn.Sequential(*m)
            self.encoding_conv2.append(m)
        self.encoding_conv2 = nn.Sequential(*self.encoding_conv2)
        self.encoding_block2 = nn.Sequential(*self.encoding_block2)

        # Active site predictions
        self.pred_active = []

        # Initialize 3D Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[-1],
                out_channels=self.nPlanes[-1],
                kernel_size=2,
                generate_new_coords=True,
                stride=2,
                dimension=3
                ))
            self.pred_active.append(ME.MinkowskiConvolution(
                self.nPlanes[-1], 1, kernel_size=1, has_bias=True, dimension=3))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[-1],
                                     self.nPlanes[-1],
                                     dimension=3,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)
        self.pred_active = nn.Sequential(*self.pred_active)


    def encoder1(self, x):
        '''
        Vanilla UResNet Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        '''
        x = self.input_layer1(x)
        for i, layer in enumerate(self.encoding_block1):
            x = self.encoding_block1[i](x)
            x = self.encoding_conv1[i](x)
        return x


    def encoder2(self, x):
        '''
        Vanilla UResNet Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        '''
        x = self.input_layer2(x)
        for i, layer in enumerate(self.encoding_block2):
            x = self.encoding_block2[i](x)
            x = self.encoding_conv2[i](x)
        return x


    def encoder(self, input_xy, input_xz):

        # Encoder
        encode_xy = self.encoder1(input_xy)
        encode_xz = self.encoder2(input_xz)

        # FC Layers
        z_xy = self.global_pool1(encode_xy)
        z_xz = self.global_pool2(encode_xz)
        z = ME.cat(z_xy, z_xz)
        latent = self.linear_enc(z)
        return latent


    def decoder(self, latent, target_key=None):
        '''
        Vanilla UResNet Decoder
        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''
        z = self.linear_dec(latent)
        x = self.global_unpool(z)
        # print("Global Unpool = ", x)
        decoderTensors = []
        voxel_predictions = []
        targets = []
        for i, layer in enumerate(self.decoding_conv):
            x = layer(x)
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
            pred = self.pred_active[i](x)
            mask = (pred.F > self.thresholding_score).cpu().squeeze()
            if self.training:
                target = get_target(x, target_key)
                targets.append(target)
                mask += target
                # print("Depth = {}".format(i), x)
            x = self.pruning(x, mask)
            # print("Depth = {}".format(i), x)
            voxel_predictions.append(pred)
        return decoderTensors, voxel_predictions, targets


    def forward(self, input):

        coords_xy = input[0][:, [0, 1, 2]].cpu().int()
        coords_xz = input[0][:, [0, 1, 3]].cpu().int()

        coords = input[0][:, 0:self.D+1].cpu().int()
        features = input[0][:, -1].view(-1, 1).float()

        input_xy = ME.SparseTensor(features, coords=coords_xy)
        input_xz = ME.SparseTensor(features, coords=coords_xz, 
            coords_manager=input_xy.coords_man, force_creation=True)

        latent = self.encoder(input_xy, input_xz)
        latent_coords, latent_feats = latent.C, latent.F
        latent_coords = torch.cat(
            [latent_coords, torch.zeros(latent_coords.shape[0], 1).int()], dim=1)

        if self.training:
            truth_3d = ME.SparseTensor(features, coords=coords)
            cm = truth_3d.coords_man
            input_generator = ME.SparseTensor(
                latent_feats, coords=latent_coords, 
                tensor_stride=self.spatial_size, coords_manager=cm)
            target_key = cm.create_coords_key(
                coords,
                force_creation=True,
                allow_duplicate_coords=True)
            decoderTensors, voxel_predictions, targets = \
                self.decoder(input_generator, target_key)
        else:
            input_generator = ME.SparseTensor(
                latent_feats, coords=latent_coords, 
                tensor_stride=self.spatial_size)
            decoderTensors, voxel_predictions, _ = self.decoder(latent)
        out = self.output_layer(decoderTensors[-1])

        res = {
            'voxel_predictions': [voxel_predictions],
            'latent': [latent],
            'out': [out],
        }

        if self.training:
            out_key = ME.CoordsKey(cm.D)
            ins, outs = cm.get_union_map(
                (out.coords_key, truth_3d.coords_key), out_key)
            N = cm.get_coords_size_by_coords_key(out_key)
            diff = torch.zeros((N, out.F.size(1)), dtype=out.dtype).cuda()
            diff[outs[0]] = out.F[ins[0]]
            diff[outs[1]] -= truth_3d.F[ins[1]]
            res['targets'] = [targets]
            res['diff'] = [diff]

        # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/output3d', out.F.detach().cpu().numpy())
        # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/outputCoords', out.C.detach().cpu().numpy())
        # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/input', input[0].detach().cpu().numpy())

        # for i, t in enumerate(voxel_predictions):
        #     np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/intermedFeats_{}.npy'.format(i), t.F.detach().cpu().numpy())
        #     np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/intermedCoords_{}.npy'.format(i), t.C.detach().cpu().numpy())

        return res


class AELoss(nn.Module):

    def __init__(self, cfg, name='aeloss'):
        super(AELoss, self).__init__()
        self.loss_config = cfg['modules'][name]
        print(self.loss_config)
        self.crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_mode = 'lovasz'
        self.training_mode = self.loss_config.get('training', True)

    def forward(self, outputs, input_data):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        energy = input_data[0][:, -1]
        voxel_predictions = outputs['voxel_predictions'][0]
        latent = outputs['latent'][0]
        final = outputs['out'][0]
        if self.training_mode:
            targets = outputs['targets'][0]
            diff = outputs['diff'][0]
            loss, num_layers = [], len(targets)
            accuracy = []
            for target, pred in zip(targets, voxel_predictions):
                # print(target.cuda())
                # print((pred.F > 0).squeeze())
                # print(iou_binary(target.cuda(), (pred.F > 0).squeeze(), per_image=False))
                # print(target.squeeze().cuda())
                print("Target = ", target.shape)
                print("Target Counts = ", torch.sum(target))
                print("Prediction = ", pred.F.shape)
                print("Prediction Counts = ", torch.sum(pred.F > 0))
                # curr_loss = lovasz_hinge_flat(pred.F.squeeze(), target.type(pred.F.dtype).cuda())
                curr_loss = self.crit(pred.F.squeeze(), target.type(pred.F.dtype).cuda())
                loss.append(curr_loss)
                acc = iou_binary(target.cuda(), (pred.F > 0).squeeze(), per_image=False)
                print(acc)
                accuracy.append(acc)

            loss = sum(loss) / len(loss)
            accuracy = sum(accuracy) / len(accuracy)
            loss += torch.pow(diff, 2).mean()
        else:
            loss, accuracy = 0, 0
        return {
            'accuracy': accuracy,
            'loss': loss
        }