def model_dict():

    # Models
    from . import uresnet_chain
    from . import acnn_chain
    from . import uresnext_chain
    from . import fpn_chain
    from . import cluster_gm
    from . import autoencoder
    from . import cluster3d
    from . import particle_types
    from . import single_particle
    from . import single_particle_bis
    from . import single_particle_3
    from . import single_particle_4
    from . import single_particle_5
    
    # Losses
    from mlreco.nn.loss.segmentation import SegmentationLoss

    models = {
        # URESNET CHAIN
        "uresnet_chain": (uresnet_chain.UResNet_Chain, uresnet_chain.SegmentationLoss),
        "fpn_chain": (fpn_chain.FPN_Chain, SegmentationLoss),
        "acnn_chain": (acnn_chain.ACNN_Chain, SegmentationLoss),
        "uresnext_chain": (uresnext_chain.UResNeXt_Chain, uresnet_chain.SegmentationLoss),
        # CLUSTERING
        "cluster_gm": (cluster_gm.ClusterGM, cluster_gm.GaussianMixtureLoss),
        "sparse_autoencoder": (autoencoder.SparseAutoEncoder, autoencoder.AELoss),
        "cluster3d": (cluster3d.Cluster3d, cluster3d.AELoss),
        "cluster3d_resnet": (cluster3d.Cluster3dResidual, cluster3d.AELoss),
        # Particle ID and Flow
        'particle_type': (particle_types.ParticleImageClassifier, particle_types.ParticleTypeLoss),
        'particle_type_and_einit': (particle_types.ParticleTypesAndEinit, particle_types.ParticleTypeAndEinitLoss),
        'particle_kinematics': (particle_types.ParticleTypesAndKinematics, particle_types.ParticleKinematicsLoss),
        'single_particle_classifier': (single_particle.singleParticleNetwork, single_particle.singleParticleLoss),
        'single_particle_classifier_bis': (single_particle_bis.SingleParticleNetwork, single_particle_bis.SingleParticleLoss), 
        'single_particle_classifier_3': (single_particle_3.singleParticleNetwork, single_particle_3.singleParticleLoss),
        'single_particle_classifier_4': (single_particle_4.singleParticleNetwork, single_particle_4.singleParticleLoss), 
        'single_particle_classifier_5': (single_particle_5.singleParticleNetwork, single_particle_5.singleParticleLoss)
    }
    return models

def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
