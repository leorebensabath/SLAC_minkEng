import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn

from sklearn.metrics import adjusted_rand_score
from mlreco.lm_utils.misc import distance_matrix

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

##########################################################################
#
#   Loss Function Modules for Hyperspace embedding clustering.
#
##########################################################################

def push_enemies(cls, sphere=False):
    '''
    Class decorator for pushing enemies away from each centroid.
    '''
    if not sphere:
        def new_intra_loss(self, features, labels, cluster_means,
                            ally_margin=0.5, enemy_margin=1.0):
            '''
            Intra-cluster loss, with per-voxel weighting and enemy loss.
            This variant of intra-cluster loss penalizes the distance
            from the centroid to its enemies in addition to pulling
            ally points towards the center.

            INPUTS:
                - ally_margin (float): centroid pulls all allied points
                inside this margin.
                - enemy_margin (float): centroid pushs all enemy points
                inside this margin.
                - weight:
            '''
            intra_loss = 0.0
            ally_loss, enemy_loss = 0.0, 0.0
            n_clusters = len(cluster_means)
            cluster_labels = labels.unique(sorted=True)
            for i, c in enumerate(cluster_labels):
                index = (labels == c)
                allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                                    p=self.norm, dim=1)
                allies = torch.clamp(
                    allies - self.loss_params['ally_margin'], min=0)
                x = self.loss_params['ally_weight'] * \
                    torch.mean(torch.pow(allies, 2))
                intra_loss += x
                # ally_loss += float(x)
                if index.all():
                    continue
                enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                                    p=self.norm, dim=1)
                enemies = torch.clamp(
                    self.loss_params['enemy_margin'] - enemies, min=0)
                x = self.loss_params['enemy_weight'] * \
                    torch.sum(torch.pow(enemies, 2))
                intra_loss += x
                # enemy_loss += float(x)

            intra_loss /= n_clusters
            return intra_loss
            # ally_loss /= n_clusters
            # enemy_loss /= n_clusters
            # return intra_loss, ally_loss, enemy_loss
    else:
        def new_intra_loss(self, features, labels, cluster_means,
                            ally_margin=0.5, enemy_margin=1.0):
            '''
            Intra-cluster loss, with per-voxel weighting and enemy loss.
            This variant of intra-cluster loss penalizes the distance
            from the centroid to its enemies in addition to pulling
            ally points towards the center.

            INPUTS:
                - ally_margin (float): centroid pulls all allied points
                inside this margin.
                - enemy_margin (float): centroid pushs all enemy points
                inside this margin.
                - weight:
            '''
            intra_loss = 0.0
            ally_loss, enemy_loss = 0.0, 0.0
            n_clusters = len(cluster_means)
            cluster_labels = labels.unique(sorted=True)
            for i, c in enumerate(cluster_labels):
                index = (labels == c)
                allies = 0.5 * (1 + self.cosine_similarity(features[index],
                    cluster_means[i].expand_as(features[index])))
                allies = torch.clamp(
                    self.loss_params['ally_margin'] - allies, min=0)
                x = self.loss_params['ally_weight'] * \
                    torch.mean(torch.pow(allies, 2))
                intra_loss += x
                # ally_loss += float(x)
                if index.all():
                    continue
                enemies = self.cosine_similarity(
                    features[~index],
                    cluster_means[i].expand_as(features[~index]))
                enemies = torch.clamp(
                    enemies - self.loss_params['enemy_margin'], min=0)
                x = self.loss_params['enemy_weight'] * \
                    torch.sum(torch.pow(enemies, 2))
                intra_loss += x
                # enemy_loss += float(x)

            intra_loss /= n_clusters
            return intra_loss
            # ally_loss /= n_clusters
            # enemy_loss /= n_clusters
            # return intra_loss, ally_loss, enemy_loss
    cls.intra_cluster_loss = new_intra_loss
    return cls


def embed_on_hypersphere(cls):
    '''
    Class decorator for embedding points on n-dimensional hypersphere.
    '''
    def new_intra_loss(self, features, labels, cluster_means, margin=0.8):
        '''
        Modified intra loss for spherical embeddings.
        '''
        intra_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            dists = 0.5 * (1 + self.cosine_similarity(features[index],
                cluster_means[i].expand_as(features[index])))
            # Similarities must be greater than at least <margin>
            hinge = torch.clamp(margin - dists, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            intra_loss += l
        intra_loss /= n_clusters
        return intra_loss

    def new_inter_loss(self, cluster_means, margin=0.7):
        '''
        Modified inter loss for spherical embeddings
        '''
        inter_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        dist = 0.5 * (1 + self.cosine_similarity(c1, c2))
                        # Similarities must be smaller than <margin>
                        hinge = torch.clamp(dist - margin, min=0)
                        inter_loss += torch.pow(hinge, 2)
            inter_loss /= float((n_clusters - 1) * n_clusters)
            return inter_loss

    def new_compute_accuracy(self, embedding, truth):
        '''
        Computes heuristic accuracy for spherical embeddings
        '''
        nearest = []
        with torch.no_grad():
            cmeans = self.find_cluster_means(embedding, truth)
            for c in cmeans:
                dists = self.cosine_similarity(
                    embedding, c.expand_as(embedding))
                dists = dists.view(-1, 1)
                nearest.append(dists)
            nearest = torch.cat(nearest, dim=1)
            nearest = torch.argmax(nearest, dim=1)
            pred = nearest.cpu().numpy()
            grd = truth.cpu().numpy()
            score = adjusted_rand_score(pred, grd)
        return score

    def new_combine(self, features, labels, **kwargs):
        '''
        Modified combine function for spherical embeddings
        '''

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(
            c_means, margin=kwargs['inter_margin'])
        intra_loss = self.intra_cluster_loss(
            features, labels, c_means, margin=kwargs['intra_margin'])
        reg_loss = self.regularization(c_means)

        loss = kwargs['intra_weight'] * intra_loss + kwargs['inter_weight'] \
            * inter_loss

        return {
            'loss': loss,
            'intra_loss': kwargs['intra_weight'] * float(intra_loss),
            'inter_loss': kwargs['inter_weight'] * float(inter_loss)
        }

    cls.intra_cluster_loss = new_intra_loss
    cls.inter_cluster_loss = new_inter_loss
    cls.compute_heuristic_accuracy = new_compute_accuracy
    cls.combine = new_combine
    return cls


class DiscriminativeLoss(nn.Module):
    '''
    Implementation of the Discriminative Loss Function in Pytorch.
    https://arxiv.org/pdf/1708.02551.pdf
    Note that there are many other implementations in Github, yet here
    we tailor it for use in conjuction with Sparse UResNet.
    '''

    def __init__(self, cfg, name='discriminative_loss'):
        super(DiscriminativeLoss, self).__init__()
        loss_config = cfg['modules']['clustering_loss']
        self.name = loss_config[name]
        self.args = loss_config[name]['args']

        # Discriminative Loss Hyperparameters
        self.num_classes = self.args.get('num_classes', 5)
        self.loss_params = {}
        self.loss_params['intra_weight'] = self.args.get('intra_weight', 1.0)
        self.loss_params['inter_weight'] = self.args.get('inter_weight', 1.0)
        self.loss_params['reg_weight'] = self.args.get('reg_weight', 0.001)
        self.loss_params['intra_margin'] = self.args.get('intra_margin', 0.5)
        self.loss_params['inter_margin'] = self.args.get('inter_margin', 1.5)
        self.loss_params['norm'] = self.args.get('norm', 2)

        self.dimension = loss_config.get('dimension', 3)
        self.use_segmentation = loss_config.get('use_segmentation', True)
        self.loss_config = loss_config


    def find_cluster_means(self, features, labels):
        '''
        For a given image, compute the centroids mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        clabels = labels.unique(sorted=True)
        cluster_means = []
        for c in clabels:
            index = (labels == c)
            mu_c = features[index].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        return cluster_means


    def intra_cluster_loss(self, features, labels, cluster_means, margin=0.5):
        '''
        Implementation of variance loss in Discriminative Loss.
        Inputs:
            features (torch.Tensor): pixel embedding, same as in find_cluster_means.
            labels (torch.Tensor): ground truth instance labels
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): constant used to specify delta_v in paper. Think of it
            as the size of each clusters in embedding space.
        Returns:
            intra_loss: (float) variance loss (see paper).
        '''
        intra_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            dists = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.loss_params['norm'], dim=1)
            hinge = torch.clamp(dists - margin, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            intra_loss += l
        intra_loss /= n_clusters
        return intra_loss


    def inter_cluster_loss(self, cluster_means, margin=1.5):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            inter_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        inter_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        dist = torch.norm(c1 - c2 + 1e-8,
                            p=self.loss_params['norm'])
                        hinge = torch.clamp(2.0 * margin - dist, min=0)
                        inter_loss += torch.pow(hinge, 2)
            inter_loss /= float((n_clusters - 1) * n_clusters)
            return inter_loss


    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = 0.0
        n_clusters, _ = cluster_means.shape
        for i in range(n_clusters):
            reg_loss += torch.norm(cluster_means[i, :] + 1e-8,
                                   p=self.loss_params['norm'])
        reg_loss /= float(n_clusters)
        return reg_loss


    def compute_heuristic_accuracy(self, embedding, truth):
        '''
        Compute Adjusted Rand Index Score for given embedding coordinates,
        where predicted cluster labels are obtained from distance to closest
        centroid (computes heuristic accuracy).

        Inputs:
            embedding (torch.Tensor): (N, d) Tensor where 'd' is the embedding dimension.
            truth (torch.Tensor): (N, ) Tensor for the ground truth clustering labels.
        Returns:
            score (float): Computed ARI Score
            clustering (array): the predicted cluster labels.
        '''
        nearest = []
        with torch.no_grad():
            cmeans = self.find_cluster_means(embedding, truth)
            for centroid in cmeans:
                dists = torch.sum((embedding - centroid)**2, dim=1)
                dists = dists.view(-1, 1)
                nearest.append(dists)
            nearest = torch.cat(nearest, dim=1)
            nearest = torch.argmin(nearest, dim=1)
            pred = nearest.cpu().numpy()
            grd = truth.cpu().numpy()
            score = adjusted_rand_score(pred, grd)
        return score


    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses.

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(
            c_means, margin=kwargs['inter_margin'])
        intra_loss = self.intra_cluster_loss(
            features, labels, c_means, margin=kwargs['intra_margin'])
        reg_loss = self.regularization(c_means)

        loss = kwargs['intra_weight'] * intra_loss + kwargs['inter_weight'] \
            * inter_loss + kwargs['reg_weight'] * reg_loss

        return {
            'loss': loss,
            'intra_loss': kwargs['intra_weight'] * float(intra_loss),
            'inter_loss': kwargs['inter_weight'] * float(inter_loss),
            'reg_loss': kwargs['reg_weight'] * float(reg_loss)
        }


    def combine_multiclass(self, features, slabels, clabels, **kwargs):
        '''
        Wrapper function for combining different components of the loss,
        in particular when clustering must be done PER SEMANTIC CLASS.

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud.

        INPUTS:
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class.
            loss[i] = computed DLoss for semantic class <i>.
            acc_segs (list): list of computed clustering accuracy for each semantic class.
        '''
        loss, acc_segs = defaultdict(list), defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            loss_blob = self.combine(features[index], clabels[index], **kwargs)
            for key, val in loss_blob.items():
                loss[key].append(val)
            acc = self.compute_heuristic_accuracy(features[index], clabels[index])
            acc_segs['accuracy_{}'.format(sc.item())] = acc
        return loss, acc_segs


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of backbone; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''
        num_gpus = len(semantic_labels)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = semantic_labels[i][:, -1]
            slabels = slabels.int()
            clabels = group_labels[i][:, -1]
            batch_idx = semantic_labels[i][:, 3]
            embedding = out['cluster_feature'][i]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]

                if self.use_segmentation:
                    loss_dict, acc_segs = self.combine_multiclass(
                        embedding_batch, slabels_batch,
                        clabels_batch, **self.loss_params)
                    for key, val in loss_dict.items():
                        loss[key].append(sum(val) / len(val))
                    for s, acc in acc_segs.items():
                        accuracy[s].append(acc)
                    acc = sum(acc_segs.values()) / len(acc_segs.values())
                    accuracy['accuracy'].append(acc)
                else:
                    loss["loss"].append(self.combine(
                        embedding_batch, clabels_batch, **self.loss_params))
                    acc, _ = self.compute_heuristic_accuracy(
                        embedding_batch, clabels_batch)
                    accuracy['accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


@embed_on_hypersphere
class HypersphereLoss(DiscriminativeLoss):
    '''
    Maps embeddings to n-dimensional hypersphere via cosine similarity loss.
    '''
    def __init__(self, cfg, name='cosine_embedding'):
        super(HypersphereLoss, self).__init__(cfg)
        self.cosine_similarity = nn.CosineSimilarity()
        self.loss_params = {}
        # No need for regularization since features are constrained on S^n.
        self.loss_params['intra_weight'] = self.args.get('intra_weight', 1.0)
        self.loss_params['inter_weight'] = self.args.get('inter_weight', 1.0)
        # Angles are in degrees
        intra_angle = self.args.get('intra_angle', 5)
        inter_angle = self.args.get('inter_angle', 20)
        self.loss_params['intra_margin'] = np.cos(np.radians(intra_angle)/2)**2
        self.loss_params['inter_margin'] = np.cos(np.radians(inter_angle)/2)**2


class HyperspaceMultiLoss(DiscriminativeLoss):
    '''
    Hyperspace Embedding loss, applied at all spatial resolutions.
    '''
    def __init__(self, cfg, name='hyperspace_multi'):
        super(HyperspaceMultiLoss, self).__init__(cfg)
        self.loss_config = cfg['modules']['clustering_loss']
        self.num_strides = self.loss_config.get('num_strides', 5)

        self.intra_margins = self.loss_config.get('intra_margins',
            [self.loss_params['intra_margin'] for i in range(self.num_strides)])
        self.inter_margins = self.loss_config.get('inter_margins',
            [self.loss_params['inter_margin'] for i in range(self.num_strides)])


    def compute_loss_layer(self, embedding, slabels, clabels, batch_idx, **kwargs):
        '''
        Compute the multi-class loss for a feature map on a given layer.
        We group the loss computation to a function in order to compute the
        clustering loss over the decoding feature maps.

        INPUTS:
            - embedding (torch.Tensor): (N, d) Tensor with embedding space
                coordinates.
            - slabels (torch.Tensor): (N, 5) Tensor with segmentation labels
            - clabels (torch.Tensor): (N, 5) Tensor with cluster labels
            - batch_idx (list): list of batch indices, ex. [0, 1, ..., 4]

        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding
                to calculated loss over a given layer.
        '''
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        coords = embedding.C

        for bidx in batch_idx:
            index = slabels[:, 3].int() == bidx
            embedding_batch = embedding[index]
            slabels_batch = slabels[index][:, -1]
            clabels_batch = clabels[index][:, -1]
            # Compute discriminative loss for current event in batch
            if self.use_segmentation:
                loss_dict, acc_segs = self.combine_multiclass(
                    embedding_batch, slabels_batch, clabels_batch, **kwargs)
                for key, val in loss_dict.items():
                    loss[key].append( sum(val) / float(len(val)) )
                for s, acc in acc_segs.items():
                    accuracy[s].append(acc)
                acc = sum(acc_segs.values()) / len(acc_segs.values())
                accuracy['accuracy'].append(acc)
            else:
                loss["loss"].append(self.combine(
                    embedding_batch, clabels_batch, **kwargs))
                acc = self.compute_heuristic_accuracy(
                    embedding_batch, clabels_batch)
                accuracy['accuracy'].append(acc)

        # Averaged over batch at each layer
        loss = { key : sum(l) / float(len(l)) for key, l in loss.items() }
        accuracy = { key : sum(l) / float(len(l)) for key, l in accuracy.items() }
        return loss, accuracy


    def forward(self, out, segment_label, cluster_label):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        num_gpus = len(segment_label)
        num_layers = len(out['clusterTensors'][0])

        for i_gpu in range(num_gpus):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['clusterTensors'][i_gpu]):
                delta_var, delta_dist = self.intra_margins[i], self.inter_margins[i]
                loss_i, acc_i = self.compute_loss_layer(em,
                    segment_label[i_gpu][i],
                    cluster_label[i_gpu][i],
                    batch_idx,
                    delta_var=delta_var,
                    delta_dist=delta_dist)
                for key, val in loss_i.items():
                    loss[key].append(val)
            # Compensate for final embedding
            final_embedding = out['final_embedding']
            loss_i, acc_i = self.compute_loss_layer(
                final_embedding,
                segment_label[igpu][0],
                cluster_label[igpu][0],
                batch_idx,
                delta_var=self.loss_params['intra_margin'],
                delta_dist=self.loss_params['inter_margin'])
            for key, val in loss_i.items():
                loss[key].append(val)
            acc_clustering = acc_i
            for key, acc in acc_clustering.items():
                # Batch Averaged Accuracy
                accuracy[key].append(acc)

        # Average over layers and num_gpus
        loss_avg = {}
        acc_avg = {}
        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


@embed_on_hypersphere
class HypersphereMultiLoss(HyperspaceMultiLoss):

    def __init__(self, cfg, name='hypersphere_multi'):
        super(HypersphereMultiLoss, self).__init__(cfg)
        self.cosine_similarity = nn.CosineSimilarity()
        self.loss_params = {}
        # No need for regularization since features are constrained on S^n.
        self.loss_params['intra_weight'] = self.args.get('intra_weight', 1.0)
        self.loss_params['inter_weight'] = self.args.get('inter_weight', 1.0)
        # Angles are in degrees
        intra_angle = self.args.get('intra_angle', 2)
        inter_angle = self.args.get('inter_angle', 5)
        self.loss_params['intra_margin'] = np.cos(np.radians(intra_angle)/2)**2
        self.loss_params['inter_margin'] = np.cos(np.radians(inter_angle)/2)**2


@push_enemies
class HyperspaceMultiLoss2(HyperspaceMultiLoss):

    def __init__(self, cfg, name='hyperspace_multi_2'):
        super(HyperspaceMultiLoss2, self).__init__(cfg, name=name)
        self.loss_params['enemy_margin'] = self.loss_config.get('enemy_margin', 1.0)
        self.loss_params['enemy_weight'] = self.loss_config.get('enemy_weight', 10.0)
        self.loss_params['ally_weight'] = self.loss_config.get('ally_weight', 1.0)
        self.loss_params['ally_margin'] = self.intra_margin


# @push_enemies(sphere=True)
# class HypersphereMultiLoss2(HypersphereMultiLoss):
#
#     def __init__(self, cfg, name='hypersphere_multi_2'):
#         super(HypersphereMultiLoss2, self).__init__(cfg, name=name)
#         enemy_angle = self.loss_config.get('enemy_angle', 3)
#         self.loss_params['enemy_margin'] = np.cos(enemy_angle/2)**2
#         self.loss_params['enemy_weight'] = self.loss_config.get('enemy_weight', 10.0)
#         self.loss_params['ally_weight'] = self.loss_config.get('ally_weight', 1.0)
#         self.loss_params['ally_margin'] = self.intra_margin


class HyperspaceDensityLoss(HyperspaceMultiLoss2):

    def __init__(self, cfg, name='hyperspace_multi_density '):
        self.density_radius = self.loss_config.get('density_radius', 0.1)
        self.sigma = self.density_radius / np.sqrt(2 * np.log(2))
        self.ally_density_weight = self.loss_config.get('ally_density_weight', 1.0)
        self.enemy_density_weight = self.loss_config.get('enemy_density_weight', 1.0)
        self.num_sample = self.loss_config.get('num_samples', 100)


##########################################################################
#
#   CLASS DECORATORS FOR MEMBER FUNCTION PATCHING
#
##########################################################################

# def add_density_loss(cls, sphere=True):

#     if not sphere:
#         def new_intra_loss(self, features, labels, cluster_means,
#                         ally_margin=0.5, enemy_margin=1.0):
#             '''
#             Smooth Density Maximization Loss
#             '''
#             intra_loss = 0.0
#             ally_loss, enemy_loss = 0.0, 0.0
#             densityAloss, densityEloss = 0.0, 0.0
#             n_clusters = len(cluster_means)
#             cluster_labels = labels.unique(sorted=True)
#             with torch.no_grad():
#                 dists = distance_matrix(features)
#             for i, c in enumerate(cluster_labels):
#                 # Intra-Pull Loss
#                 index = (labels == c)
#                 allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
#                                 p=self.norm, dim=1)
#                 allies = torch.clamp(allies - ally_margin, min=0)
#                 x = self.ally_weight * torch.mean(torch.pow(allies, 2))
#                 intra_loss += x
#                 # ally_loss += float(x)
#                 # Ally Density Loss (Check that at least k allies exist)
#                 if sum(index) < 5:
#                     k = sum(index)
#                 else:
#                     k = 5
#                 _, idx_ally = dists[index, :][:, index].topk(k, dim=1, largest=False)
#                 x = torch.sum(torch.pow(
#                     features[index].unsqueeze(1) - features[index][idx_ally], 2), dim=2)
#                 x = torch.mean(torch.clamp(torch.exp(-x / (2 * self.sigma**2)), max=0.5))
#                 intra_loss += x
#                 # densityAloss += float(x)
#                 if index.all():
#                     continue
#                 # Intra-Push Loss
#                 enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
#                         p=self.norm, dim=1)
#                 enemies = torch.clamp(enemy_margin - enemies, min=0)
#                 x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
#                 intra_loss += self.ally_density_weight * x
#                 # enemy_loss += float(x)
#                 # Enemy Density Loss (Check that at least k enemies exist)
#                 if sum(~index) < 5:
#                     k = sum(~index)
#                 else:
#                     k = 5
#                 _, idx_enemy = dists[~index, :][:, ~index].topk(k, dim=1, largest=False)
#                 x = torch.sum(torch.pow(
#                     features[~index].unsqueeze(1) - features[~index][idx_enemy], 2), dim=2)
#                 x = torch.mean(torch.clamp(torch.exp(-x / (2 * self.sigma**2)), max=0.5))
#                 intra_loss += self.enemy_density_weight * x
#                 # densityEloss += float(x)

#             intra_loss /= n_clusters
#             # ally_loss /= n_clusters
#             # enemy_loss /= n_clusters
#             # densityAloss /= n_clusters
#             # densityEloss /= n_clusters
#             # return intra_loss, ally_loss, enemy_loss, densityAloss, densityEloss
#             return intra_loss
#     else:
#         def new_intra_loss(self, features, labels, cluster_means,
#                 ally_margin=0.5, enemy_margin=1.0):
#             '''
#             Smooth Density Maximization Loss
#             '''
#             intra_loss = 0.0
#             ally_loss, enemy_loss = 0.0, 0.0
#             densityAloss, densityEloss = 0.0, 0.0
#             n_clusters = len(cluster_means)
#             cluster_labels = labels.unique(sorted=True)
#             with torch.no_grad():
#                 dists = distance_matrix(features)
#             for i, c in enumerate(cluster_labels):
#                 # Intra-Pull Loss
#                 index = (labels == c)
#                 allies = 0.5 * (1 + self.cosine_similarity(features[index],
#                     cluster_means[i].expand_as(features[index])))
#                 allies = torch.clamp(
#                     self.loss_params['ally_margin'] - allies, min=0)
#                 x = self.loss_params['ally_weight'] * \
#                     torch.mean(torch.pow(allies, 2))
#                 intra_loss += x
#                 # ally_loss += float(x)
#                 # Ally Density Loss (Check that at least k allies exist)
#                 if sum(index) < 5:
#                     k = sum(index)
#                 else:
#                     k = 5
#                 _, idx_ally = dists[index, :][:, index].topk(k, dim=1, largest=False)
#                 x = torch.sum(torch.pow(
#                     features[index].unsqueeze(1) - features[index][idx_ally], 2), dim=2)
#                 x = torch.mean(torch.clamp(torch.exp(-x / (2 * self.sigma**2)), max=0.5))
#                 intra_loss += x
#                 # densityAloss += float(x)
#                 if index.all():
#                     continue
#                 # Intra-Push Loss
#                 enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
#                         p=self.norm, dim=1)
#                 enemies = torch.clamp(enemy_margin - enemies, min=0)
#                 x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
#                 intra_loss += self.ally_density_weight * x
#                 # enemy_loss += float(x)
#                 # Enemy Density Loss (Check that at least k enemies exist)
#                 if sum(~index) < 5:
#                     k = sum(~index)
#                 else:
#                     k = 5
#                 _, idx_enemy = dists[~index, :][:, ~index].topk(k, dim=1, largest=False)
#                 x = torch.sum(torch.pow(
#                     features[~index].unsqueeze(1) - features[~index][idx_enemy], 2), dim=2)
#                 x = torch.mean(torch.clamp(torch.exp(-x / (2 * self.sigma**2)), max=0.5))
#                 intra_loss += self.enemy_density_weight * x
#                 # densityEloss += float(x)

#             intra_loss /= n_clusters
#             # ally_loss /= n_clusters
#             # enemy_loss /= n_clusters
#             # densityAloss /= n_clusters
#             # densityEloss /= n_clusters
#             # return intra_loss, ally_loss, enemy_loss, densityAloss, densityEloss
#             return intra_loss

    pass

def add_distance_estimation(cls):
    pass

def attention_weight(cls, valley=True):
    pass
