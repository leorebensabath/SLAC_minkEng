import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from mlreco.nn.layers.factories import activations_dict, activations_construct
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


class SparseGenerator(NetworkBase):
    '''
    MinkowskiNet Generator Module for sparse tensor reconstruction.
    '''
    def __init__(self, cfg, name='sparse_generator'):
        super(SparseGenerator, self).__init__(cfg)
        self.model_config = cfg[name]
        print(name, self.model_config)
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 3)
        self.thresholding_score = self.model_config.get('thresholding_score', 0.0)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        print("Final Tensor Shape = ", final_tensor_shape)

        # Active site prediction layers
        self.pred_active = []

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[-1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[-1],
                out_channels=self.nPlanes[-1],
                kernel_size=2,
                generate_new_coords=True,
                stride=2,
                dimension=self.D
                ))
            self.pred_active.append(ME.MinkowskiConvolution(
                self.nPlanes[-1], 1, kernel_size=1, has_bias=True, dimension=3))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ConvolutionBlock(self.nPlanes[-1],
                                     self.nPlanes[-1],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)
        self.pred_active = nn.Sequential(*self.pred_active)

        self.output_layer = ConvolutionBlock(
            self.nPlanes[-1], 1, dimension=self.D, 
            activation=self.activation_name, activation_args=self.activation_args)

        self.pruning = ME.MinkowskiPruning()

        self.linear2 = nn.Sequential(
            ME.MinkowskiBatchNorm(self.latent_size),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolution(
                in_channels=self.latent_size, out_channels = self.nPlanes[-1],
                kernel_size=1, stride=1, dimension=self.D
            )
        )

        self.global_unpool = nn.Sequential(
            ME.MinkowskiBatchNorm(self.nPlanes[-1]),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[-1],
                out_channels=self.nPlanes[-1],
                kernel_size=final_tensor_shape,
                generate_new_coords=True,
                stride=final_tensor_shape,
                dimension=self.D))

        self.unpooling_pred = ME.MinkowskiConvolution(
                self.nPlanes[-1], 1, kernel_size=1, has_bias=True, dimension=3)


    def get_target(self, out, target_key, kernel_size=1):
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

        
    def forward(self, latent_features, gen_coords, input_coords=None):

        gen_input = ME.SparseTensor(
            latent_features, 
            coords=gen_coords, 
            tensor_stride=self.spatial_size,
            allow_duplicate_coords=True)

        print("Training = ", self.training)

        cm = gen_input.coords_man

        if self.training:
            target_key = cm.create_coords_key(
                input_coords,
                force_creation=True,
                allow_duplicate_coords=True)

        z = self.linear2(gen_input)

        decoderTensors = []
        voxel_predictions = []
        targets = []
        # Decoder
        # print(z.C)
        x = self.global_unpool(z)
        # print(x.C, x.tensor_stride)
        # print(x)
        pred_unpool = self.unpooling_pred(x)
        mask = (pred_unpool.F > self.thresholding_score).cpu().squeeze()
        if self.training:
            target = self.get_target(x, target_key)
            targets.append(target)
            mask += target
        x = self.pruning(x, mask)
        voxel_predictions.append(pred_unpool)
        # print("Decoder 0: ", x.tensor_stride)
        for i, layer in enumerate(self.decoding_conv):
            x = layer(x)
            # print("Decoder {}: ".format(i), x.tensor_stride)
            x = self.decoding_block[i](x)
            if self.training:
                target = self.get_target(x, target_key)
                # print(target)
                targets.append(target)
            decoderTensors.append(x)
            pred = self.pred_active[i](x)
            mask = (pred.F > self.thresholding_score).cpu().squeeze()
            # print(mask)
            if self.training:
                mask += target
            x = self.pruning(x, mask)
            voxel_predictions.append(pred)

        out = self.output_layer(x)

        # print('---------------Generator----------------')
        # for i, p in enumerate(voxel_predictions):
        #     print(p.C)
        # print(targets)

        res = {
            'targets': [targets],
            'voxel_predictions': [voxel_predictions],
            'out': [out]
        }

        # for i, p in enumerate(voxel_predictions):
        #     np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/event_dump/autoenc/coords_{}'.format(i), p.C.detach().cpu().numpy())
        #     np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/event_dump/autoenc/layer_{}'.format(i), p.F.detach().cpu().numpy())
        # assert False
        return res


class SparseEncoder(NetworkBase):
    '''
    Minkowski Net Autoencoder for sparse tensor reconstruction. 
    '''
    def __init__(self, cfg, name='sparse_encoder'):
        super(SparseEncoder, self).__init__(cfg)
        self.model_config = cfg[name]
        print(name, self.model_config)
        self.reps = self.model_config.get('reps', 2)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = self.model_config.get('input_kernel', 7)
        self.latent_size = self.model_config.get('latent_size', 512)
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        self.coordConv = self.model_config.get('coordConv', False)
        print("Final Tensor Shape = ", final_tensor_shape)

        # Initialize Input Layer
        if self.coordConv:
            self.input_layer = ME.MinkowskiConvolution(
                in_channels=self.num_input + self.D,
                out_channels=self.num_filters,
                kernel_size=self.input_kernel, stride=1, dimension=self.D)
        else:
            self.input_layer = ME.MinkowskiConvolution(
                in_channels=self.num_input,
                out_channels=self.num_filters,
                kernel_size=self.input_kernel, stride=1, dimension=self.D)

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ConvolutionBlock(F, F,
                    dimension=self.D,
                    activation=self.activation_name,
                    activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.encoding_block.append(m.cuda())
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m.cuda())

#        self.encoding_conv = nn.Sequential(*self.encoding_conv)
#        self.encoding_block = nn.Sequential(*self.encoding_block)

        self.global_pool = ME.MinkowskiConvolution(
            in_channels=self.nPlanes[-1], out_channels=self.nPlanes[-1],
            kernel_size=final_tensor_shape, stride=final_tensor_shape, dimension=self.D)

        self.max_pool = ME.MinkowskiGlobalPooling()

        self.linear1 = nn.Sequential(
            ME.MinkowskiBatchNorm(self.nPlanes[-1]),
            activations_construct(
                self.activation_name, **self.activation_args),
            ME.MinkowskiConvolution(
                in_channels=self.nPlanes[-1], out_channels = self.latent_size,
                kernel_size=1, stride=1, dimension=self.D
            )
        )

        self.union = ME.MinkowskiUnion()
        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def encoder(self, x):
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
        x = self.input_layer(x)
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            # print(x.tensor_stride)
            # print("Encoder {}: ".format(i), x.tensor_stride)
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)

        # for i, p in enumerate(encoderTensors):
            # np.save('/gpfs/slac/staas/fs1/g/neutrino/koh0207/event_dump/autoenc/target_{}'.format(i), p.C.detach().cpu().numpy())

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result


    def forward(self, input_tensor):

        # Encoder
        encoderOutput = self.encoder(input_tensor)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        # print('---------------Encoder----------------')
        # for i, p in enumerate(encoderTensors):
        #     print(p.C)

        # FC Layers
        # print(finalTensor.C.shape)
        # print(finalTensor.C, finalTensor.tensor_stride)
        z = self.global_pool(finalTensor)
        # z = self.max_pool(z)
        latent = self.linear1(z)

        return [latent]


class SparseAutoEncoder(NetworkBase):

    def __init__(self, cfg, name='sparse_autoencoder'):
        super(SparseAutoEncoder, self).__init__(cfg)
        self.model_config = cfg[name]
        self.generator = SparseGenerator(cfg)
        self.encoder = SparseEncoder(cfg)
        self.center_coords = self.model_config.get('center_coords', 256)
        self.coordConv = self.encoder.coordConv


    def forward(self, input):

        coords = input[0][:, 0:self.D+1].cpu().int()
        # coords[:, 1:] = coords[:, 1:] - (self.spatial_size // 2)
        features = input[0][:, -1].view(-1, 1).float()
        if self.coordConv:
            features = torch.cat([features, coords[:, 1:]], dim=1)

        input_tensor = ME.SparseTensor(features, coords=coords)
        latent = self.encoder(input_tensor)[0]
        # print(latent.C, latent.tensor_stride)

        gen_coords = latent.C
        gen_coords[:, 1:] += self.center_coords

        if self.training:
            gen_res = self.generator(latent.F, gen_coords, input_coords=coords)
        else:
            gen_res = self.generator(latent.F, gen_coords)

        res = {}
        res.update(gen_res)
        res['latent'] = [latent]

        return res


class AELoss(nn.Module):

    def __init__(self, cfg, name='aeloss'):
        super(AELoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_config = cfg[name]
        self.train_mode = self.loss_config.get('training', True)
        self.level = self.loss_config.get('level', 3)
        self.energy_reco_weight = self.loss_config.get('energy_reco_weight', 0)
        self.union = ME.MinkowskiUnion()

    def forward(self, outputs, inputs):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        device = inputs[0].device
        energy = inputs[0][:, -1]
        targets = outputs['targets'][0]
        voxel_predictions = outputs['voxel_predictions'][0]
        latent = outputs['latent'][0]
        final = outputs['out'][0]

        coords = final.C.detach().cpu().numpy()
        perm = np.lexsort((coords[:, 1], coords[:, 2], coords[:, 3], coords[:, 0]))

        num_layers = len(targets)
        accuracy = []
        loss, deltaE = [], 0
        if self.train_mode:
            count = 0
            for target, pred in zip(targets, voxel_predictions):
                if count < self.level:
                    # print(pred.F)
                    curr_loss = self.crit(pred.F.squeeze(), target.type(pred.F.dtype).cuda())
                    loss.append(curr_loss)
                    with torch.no_grad():
                        acc = float(iou_binary(target.cuda(), (pred.F > 0).squeeze(), per_image=False))
                    print(count, acc)
                    accuracy.append(acc)
                count += 1
                # acc = torch.sum((pred.F > 0).cpu() & target) / target.shape[0]
                # accuracy.append(acc)

            loss = sum(loss) / len(loss)
            accuracy = sum(accuracy) / len(accuracy)
            # print(final.C, final.tensor_stride)
            input_sparse = ME.SparseTensor(
                inputs[0][:, -1].view(-1, 1), 
                coords=inputs[0][:, :4], 
                coords_manager=final.coords_man, force_creation=True)
            zeros_final = ME.SparseTensor(
                torch.zeros(final.C.shape[0]).to(device).view(-1, 1), 
                coords=final.C, coords_manager=final.coords_man, force_creation=True)
            zeros_input = ME.SparseTensor(
                torch.zeros(inputs[0].shape[0]).to(device).view(-1, 1), 
                coords=inputs[0][:, :4], coords_manager=final.coords_man, force_creation=True)
            input_expanded = self.union(input_sparse, zeros_final)
            final_expanded = self.union(final, zeros_input)
            diff = input_expanded - final_expanded
            deltaE = torch.pow(diff.F, 2).mean()
            loss += self.energy_reco_weight * deltaE
        else:
            loss, accuracy = 0, 0

        return {
            'accuracy': accuracy,
            'loss': loss,
            'deltaE': float(deltaE)
        }
