import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from collections import defaultdict
from .lovasz import lovasz_softmax_flat, StableBCELoss

# -------------------------------------------------------------
# Various Loss Functions for Semantic Segmentation

class LovaszSoftmaxLoss(nn.Module):

    def __init__(self, name='lovasz_softmax'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax()
        
    def forward(self, input, target):
        x = self.log_softmax(input)
        x = torch.clamp(torch.exp(x), min=0, max=1)
        return lovasz_softmax_flat(x, target, classes='all')


class FocalLoss(nn.Module):
    '''
    Multi-class focal loss for semantic segmentation. 
    '''
    def __init__(self, name='focal'):
        super(FocalLoss, self).__init__()
    def forward(self, input, target):
        pass

# -------------------------------------------------------------
# Segmentation Loss Function Factory

def segmentation_loss_dict():
    losses = {
        'cross_entropy': nn.CrossEntropyLoss(reduction='none'),
        'lovasz_softmax': LovaszSoftmaxLoss()
        # 'focal': segmentation.FocalLoss(reduce=False),
        # 'weighted_cross_entropy': segmentation.WeightedFocalLoss(reduce=False),
    }
    return losses

def segmentation_loss_construct(name):
    losses = segmentation_loss_dict()
    print("Using segmentation loss: {}".format(name))
    if name not in losses:
        raise Exception("Unknown loss function name provided")
    return losses[name]


class SegmentationLoss(nn.Module):
    '''
    Module for computing semantic segmentation loss.
    '''
    def __init__(self, cfg, name='segmentation_loss'):
        super(SegmentationLoss, self).__init__()
        loss_cfg = cfg['modules']['segmentation_loss']
        self.loss_name = loss_cfg.get('name', 'cross_entropy')
        self.loss_fn = segmentation_loss_construct(self.loss_name)

    def forward(self, outputs, label, weight=None):
        '''
        segmentation[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        '''
        # TODO Add weighting
        segmentation = outputs['segmentation']
        assert len(segmentation) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        batch_ids = [d[:, -2] for d in label]
        total_loss = 0
        total_acc = 0
        count = 0
        # Loop over GPUS
        for i in range(len(segmentation)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b
                event_segmentation = segmentation[i][batch_index]
                event_label = label[i][:, -1][batch_index]
                event_label = torch.squeeze(event_label, dim=-1).long()
                loss_seg = self.loss_fn(event_segmentation, event_label)
                if weight is not None:
                    event_weight = weight[i][batch_index]
                    event_weight = torch.squeeze(event_weight, dim=-1).float()
                    total_loss += torch.mean(loss_seg * event_weight)
                else:
                    total_loss += torch.mean(loss_seg)
                # Accuracy
                predicted_labels = torch.argmax(event_segmentation, dim=-1)
                acc = (predicted_labels == event_label).sum().item() \
                    / float(predicted_labels.nelement())
                total_acc += acc
                count += 1

        return {
            'accuracy': total_acc/count,
            'loss': total_loss/count
        }