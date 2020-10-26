def backbone_dict():
    from . import uresnet
    from . import uresnext
    from . import fpn
    from . import deeplab
    networks = {
        'uresnet': uresnet.UResNet,
        'uresnet_cascade': uresnet.ASPPUNet,
        'fpn': fpn.FPN,
        'uresnext': uresnext.UResNeXt,
        'deeplab': deeplab.DeepLabUNet
    }
    return networks

def backbone_construct(name):
    networks = backbone_dict()
    if name not in networks:
        raise Exception("Unknown loss function name provided")
    return networks[name]