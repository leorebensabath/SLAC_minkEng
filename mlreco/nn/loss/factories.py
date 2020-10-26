import torch
import torch.nn as nn

# Semantic Segmentation Loss Function Constructor

def segmentation_loss_dict():
    from . import segmentation
    losses = {
        'cross_entropy': nn.CrossEntropyLoss(reduction='none'),
        'lovasz_softmax': segmentation.LovaszSoftmaxLoss(),
        'focal': segmentation.FocalLoss(reduce=False),
        'weighted_cross_entropy': segmentation.WeightedFocalLoss(reduce=False),
    }
    return losses

def segmentation_loss_construct():
    losses = segmentation_loss_dict()
    if name not in losses:
        raise Exception("Unknown loss function name provided")
    return losses[name]

# Gaussian Mixture Loss Function Constructor

def gm_loss_dict():
    from . import gaussian_mixtures as gm
    losses = {
        'bce': gm.MaskBCELoss2,
        'bce_aa_ellipse': gm.MaskBCELossBivariate,
        'lovasz_hinge': gm.MaskLovaszHingeLoss,
        'lovasz_hinge_inter': gm.MaskLovaszInterLoss,
        # 'lovasz_aa_ellipse:': MaskLovaszEllipsoidalLoss,
        'focal': gm.MaskFocalLoss,
        'weighted_focal': gm.MaskWeightedFocalLoss,
        'lovasz_ellipse': gm.MaskLovaszEllipsoidal,
        'ce_lovasz': gm.CELovaszLoss
    }
    return losses

def gm_loss_construct(name):
    losses = gm_loss_dict()
    print("Using segmentation loss: {}".format(name))
    if name not in losses:
        raise Exception("Unknown loss function name provided")
    return losses[name]

# Hyperspace Embedding Loss Function Constructor
