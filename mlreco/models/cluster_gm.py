import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.cluster.gaussian_mixtures import GaussianMixture
from mlreco.nn.loss.factories import *
from mlreco.nn.loss.segmentation import SegmentationLoss
from mlreco.nn.layers.misc import normalize_coords
from mlreco.nn.layers.network_base import NetworkBase
from collections import defaultdict


class ClusterGM(NetworkBase):
    '''
    GM stands for Gaussian Mixtures
    '''
    def __init__(self, cfg, name='cluster_gm'):
        super(ClusterGM, self).__init__(cfg)
        self.model_cfg = cfg['modules']
        self.net = GaussianMixture(cfg, name='gaussian_mixtures')

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.margins_fn = nn.Sigmoid()

        print(self)

        print('Total Number of Trainable Parameters = {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        out = defaultdict(list)
        num_gpus = len(input)
        for igpu, x in enumerate(input):
            res = self.net(x)
            features = res['embeddings'].F
            coords = normalize_coords(res['embeddings'].C)
            seed = res['seediness']
            seed = self.sigmoid(seed.F)
            embeddings = features[:, :3]
            margins = features[:, 3:]
            embeddings = self.tanh(embeddings)
            margins = torch.clamp(2 * self.margins_fn(margins), min=1e-6)
            out['seediness'].append(seed)
            out['embeddings'].append(embeddings + coords)
            out['margins'].append(margins)

        return out


class GaussianMixtureLoss(NetworkBase):
    '''
    Module defining the various losses available for the
    gaussian mixture clustering model.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(GaussianMixtureLoss, self).__init__(cfg)
        cluster_loss_cfg = cfg['modules']['clustering_loss']
        self.cluster_loss_name = cluster_loss_cfg.get(
            'name', 'lovasz_hinge_inter')
        segment_loss_cfg = cfg['modules']['segmentation_loss']
        self.seg_loss_name = segment_loss_cfg.get(
            'name', 'cross_entropy')
        constructor = gm_loss_construct(self.cluster_loss_name)
        self.cluster_loss = constructor(cfg)
        self.segment_loss = SegmentationLoss(cfg, self.seg_loss_name)

        self.segmentation_weight = cluster_loss_cfg.get(
            'segmentation_weight', 1.0)

    def forward(self, out, segment_label, input_data, weight=None):

        num_gpus = len(segment_label)
        loss = defaultdict(list)
        accuracy = defaultdict(list)
        assert len(segment_label) == num_gpus

        for i in range(num_gpus):

            slabels = segment_label[i][:, -1]
            coords = segment_label[i][:, :3].float()
            if torch.cuda.is_available():
                coords = coords.cuda()
            slabels = slabels.int()
            clabels = input_data[i][:, 4]
            batch_idx = segment_label[i][:, 3]
            embedding = out['embeddings'][i]
            seediness = out['seediness'][i]
            margins = out['margins'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]
                seed_batch = seediness[batch_idx == bidx]
                margins_batch = margins[batch_idx == bidx]
                coords_batch = coords[batch_idx == bidx] / self.spatial_size

                loss_class, acc_class = self.cluster_loss.combine_multiclass(
                    embedding_batch, margins_batch, \
                    seed_batch, slabels_batch, clabels_batch, coords_batch)
                for key, val in loss_class.items():
                    loss[key].append(sum(val) / len(val))
                for key, val in acc_class.items():
                    accuracy[key].append(val)
                # batch_index = batch_idx == bidx
                # event_segmentation = segmentation[i][batch_index]
                # event_label = segment_label[i][:, -1][batch_index]
                # event_label = torch.squeeze(event_label, dim=-1).long()
                # loss_seg = torch.mean(self.segment_loss.loss_fn(
                #     event_segmentation, event_label))
                # loss['segmentation_loss'].append(
                #     self.segmentation_weight * loss_seg)
                # if weight is not None:
                #     event_weight = weight[i][batch_index]
                #     event_weight = torch.squeeze(event_weight, dim=-1).float()
                #     loss['segmentation'].append(torch.mean(loss_seg * event_weight))
                # else:
                #     loss['segmentation'].append(torch.mean(loss_seg))
                # # Accuracy
                # predicted_labels = torch.argmax(event_segmentation, dim=-1)
                # acc = (predicted_labels == event_label).sum().item() \
                #     / float(predicted_labels.nelement())
                # accuracy['accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        # loss_avg['loss'] += loss_avg['segmentation_loss']
        # loss_avg['segmentation_loss'] = float(loss_avg['segmentation_loss'])
        acc_total = 0
        counts = 0
        for key, val in accuracy.items():
            counts += 1
            acc_avg[key] = sum(val) / len(val)
            acc_total += sum(val) / len(val)

        res = {}
        res['accuracy'] = acc_total / counts
        res.update(loss_avg)
        res.update(acc_avg)
        # print("Mask loss = {}".format(res['mask_loss']))
        # print("Smoothing loss = {}".format(res['smoothing_loss']))
        # print("Segmentation loss = {}".format(res['segmentation_loss']))

        return res
