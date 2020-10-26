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
    from . import single_particle_res
    from . import single_particle_bis
    from . import single_particle_3
    from . import single_particle_4
    from . import single_particle_5
    from . import single_particle_6
    from . import particle_type_2
    from . import particle_type_2_custKer
    from . import uresnet
    from . import uresnet_2
    from . import uresnet_2bis
    from . import uresnet_2_custKer
    from . import uresnet_2bis_custKer
    from . import uresnet_3
    from . import clustering
    from . import clustering_test
    from . import clustering_2
    from . import particle_type_2_same
    from . import particle_type_2_same2
    
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
        'particle_image_classifier': (particle_types.ParticleImageClassifier, particle_types.ParticleTypeLoss),
        'particle_type_and_einit': (particle_types.ParticleTypesAndEinit, particle_types.ParticleTypeAndEinitLoss),
        'particle_kinematics': (particle_types.ParticleTypesAndKinematics, particle_types.ParticleKinematicsLoss),
        'single_particle_classifier_res': (single_particle_res.SingleParticleNetwork, single_particle_res.SingleParticleLoss),
        'single_particle_classifier_bis': (single_particle_bis.SingleParticleNetwork, single_particle_bis.SingleParticleLoss), 
        'single_particle_classifier_3': (single_particle_3.SingleParticleNetwork, single_particle_3.SingleParticleLoss),
        'single_particle_classifier_4': (single_particle_4.SingleParticleNetwork, single_particle_4.SingleParticleLoss), 
        'single_particle_classifier_5': (single_particle_5.SingleParticleNetwork, single_particle_5.SingleParticleLoss),
        'single_particle_classifier_6': (single_particle_6.SingleParticleNetwork, single_particle_6.SingleParticleLoss),
#        'particle_image_classifier': (particle_type_2.ParticleImageClassifier, particle_type_2.ParticleTypeLoss),
        'particle_image_classifier_custKer': (particle_type_2_custKer.ParticleImageClassifier, particle_type_2_custKer.ParticleTypeLoss),
        'particle_image_classifier_same': (particle_type_2_same.ParticleImageClassifier, particle_type_2_same.ParticleTypeLoss),
        'particle_image_classifier_same2': (particle_type_2_same2.SingleParticleNetwork, particle_type_2_same2.SingleParticleLoss),
        'uresnet': (uresnet.UResNet, uresnet.SegmentationLoss),
        'uresnet_2bis': (uresnet_2bis.UResNet, uresnet_2bis.SegmentationLoss), 
        'uresnet_2bis_custKer': (uresnet_2bis_custKer.UResNet, uresnet_2bis_custKer.SegmentationLoss),
        'uresnet_3': (uresnet_3.UResNet, uresnet_3.SegmentationLoss),
        'uresnet_lonely': (uresnet_2.UResNet, uresnet_2.SegmentationLoss),
        'uresnet_lonely_custKer': (uresnet_2_custKer.UResNet, uresnet_2_custKer.SegmentationLoss),
        'clustering_uresnet': (clustering.ClusteringUresnet, clustering.ClusteringLoss),
        'clustering_uresnet_test': (clustering_test.ClusteringUresnet, clustering_test.ClusteringLoss), 
        'clustering_uresnet_2': (clustering_2.ClusteringUresnet, clustering_2.ClusteringLoss)
    }
    return models

def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
