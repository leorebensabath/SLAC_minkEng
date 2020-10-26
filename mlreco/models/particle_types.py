import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.autoencoder import SparseEncoder
from mlreco.nn.layers.network_base import NetworkBase
from collections import defaultdict

from mlreco.nn.loss.lovasz import StableBCELoss

#Same network as the one coded with SSCN

class ParticleImageClassifier(NetworkBase):

    def __init__(self, cfg, name='particle_image_classifier'):
        super(ParticleImageClassifier, self).__init__(cfg)
        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)
        self.encoder = SparseEncoder(cfg).cuda()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.encoder.latent_size),
            nn.ELU(),
            nn.Linear(self.encoder.latent_size, self.num_classes)
        )
        self.global_pooling = ME.MinkowskiGlobalPooling(average=True)

    def forward(self, input):
        print(input)
        coords = input[0][:, 0:self.D+1].cuda().float()
        # coords[:, 1:] = coords[:, 1:] - (self.spatial_size // 2)
        features = input[0][:, -1].view(-1, 1).float()

        input_tensor = ME.SparseTensor(features, coords=coords)
        latent = self.encoder(input_tensor)[0]
        # perm = np.lexsort((latent.C.detach().cpu().numpy()[:, 3],
        #                    latent.C.detach().cpu().numpy()[:, 2],
        #                    latent.C.detach().cpu().numpy()[:, 1],
        #                    latent.C.detach().cpu().numpy()[:, 0]))
        # print(latent.C[perm], latent.F[perm])
        latent = self.global_pooling(latent)
        logits = self.classifier(latent.F)
        res = {
            'logits': [logits]
        }
        return res


class ParticleTypesAndEinit(ParticleImageClassifier):

    def __init__(self, cfg, name='particle_type_and_einit'):
        super(ParticleTypesAndEinit, self).__init__(cfg)
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(self.encoder.latent_size),
            nn.ELU(),
            nn.Linear(self.encoder.latent_size, self.encoder.latent_size),
            nn.BatchNorm1d(self.encoder.latent_size),
            nn.ELU(),
            nn.Linear(self.encoder.latent_size, 1)
        )

    def forward(self, input):

        coords = input[0][:, 0:self.D+1].cpu().int()
        # coords[:, 1:] = coords[:, 1:] - (self.spatial_size // 2)
        features = input[0][:, -1].view(-1, 1).float()

        input_tensor = ME.SparseTensor(features, coords=coords)
        latent = self.encoder(input_tensor)[0]
        # perm = np.lexsort((latent.C.detach().cpu().numpy()[:, 3],
        #                    latent.C.detach().cpu().numpy()[:, 2],
        #                    latent.C.detach().cpu().numpy()[:, 1],
        #                    latent.C.detach().cpu().numpy()[:, 0]))
        # print(latent.C[perm], latent.F[perm])
        latent = self.global_pooling(latent)
        logits = self.classifier(latent.F)
        e_init = self.regressor(latent.F)
        res = {
            'logits': [logits],
            'energy_init': [e_init]
        }
        return res


class ParticleTypesAndKinematics(ParticleTypesAndEinit):


    def __init__(self, cfg, name='particle_type_and_einit'):
        super(ParticleTypesAndKinematics, self).__init__(cfg)
        self.direction_estimator = nn.Sequential(
            nn.BatchNorm1d(self.encoder.latent_size),
            nn.ELU(),
            nn.Linear(self.encoder.latent_size, self.encoder.latent_size),
            nn.BatchNorm1d(self.encoder.latent_size),
            nn.ELU(),
            nn.Linear(self.encoder.latent_size, 3)
        )

    def forward(self, input):

        coords = input[0][:, 0:self.D+1].cpu().int()
        # coords[:, 1:] = coords[:, 1:] - (self.spatial_size // 2)
        features = input[0][:, -1].view(-1, 1).float()

        input_tensor = ME.SparseTensor(features, coords=coords)
        latent = self.encoder(input_tensor)[0]
        # perm = np.lexsort((latent.C.detach().cpu().numpy()[:, 3],
        #                    latent.C.detach().cpu().numpy()[:, 2],
        #                    latent.C.detach().cpu().numpy()[:, 1],
        #                    latent.C.detach().cpu().numpy()[:, 0]))
        # print(latent.C[perm], latent.F[perm])
        latent = self.global_pooling(latent)
        logits = self.classifier(latent.F)
        e_init = self.regressor(latent.F)
        p_vec = self.direction_estimator(latent.F)
        p_vec = p_vec / torch.norm(p_vec, dim=1).view(-1, 1)
        # print("p_vec = ", p_vec)
        res = {
            'logits': [logits],
            'energy_init': [e_init],
            'p_vec': [p_vec]
        }
        return res


class ParticleTypeLoss(nn.Module):

    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleTypeLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, out, type_labels):
        logits = out['logits'][0]
        device = logits.device
        labels = [t.view(-1, 1) for t in type_labels[0]]
        labels = torch.cat(labels, dim=1).view(-1)
        # labels = type_labels[0][:, 0].to(dtype=torch.long)

        # print(logits, logits.shape)
        # if logits.shape[0] != labels.shape[0]:
        #     ignore = -torch.ones(int(logits.shape[0] - labels.shape[0])).to(device, dtype=torch.long)
        #     labels = torch.cat([labels, ignore], dim=0)

        loss = self.xentropy(logits, labels)
        pred = torch.argmax(logits, dim=1)

        accuracy = float(torch.sum(pred == labels)) / float(labels.shape[0])

        res = {
            'loss': loss,
            'accuracy': accuracy
        }
        acc_types = {}
        for c in labels.unique():
            mask = labels == c
            acc_types['accuracy_{}'.format(int(c))] = \
                float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask))
        return res


class ParticleTypeAndEinitLoss(ParticleTypeLoss):

    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleTypeAndEinitLoss, self).__init__(cfg)
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, out, type_labels, energy_truth):
        logits = out['logits'][0]
        e_init = out['energy_init'][0]
        device = logits.device
        labels = [t.view(-1, 1) for t in type_labels[0]]
        labels = torch.cat(labels, dim=1).view(-1)
        energy_init = [t.view(-1, 1) for t in energy_truth[0]]
        energy_init = torch.cat(energy_init, dim=1).view(-1, 1)
        energy_init /= 1000.0
        # labels = type_labels[0][:, 0].to(dtype=torch.long)

        # print(logits, logits.shape)
        # if logits.shape[0] != labels.shape[0]:
        #     ignore = -torch.ones(int(logits.shape[0] - labels.shape[0])).to(device, dtype=torch.long)
        #     labels = torch.cat([labels, ignore], dim=0)

        loss = self.xentropy(logits, labels)
        loss_reg = self.mseloss(e_init, energy_init).mean()

        loss += loss_reg

        pred = torch.argmax(logits, dim=1)

        accuracy = float(torch.sum(pred == labels)) / float(labels.shape[0])

        res = {
            'loss': loss,
            'loss_reg': float(loss_reg),
            'accuracy': accuracy
        }
        acc_types = {}
        loss_types = {}
        with torch.no_grad():
            for c in labels.unique():
                mask = labels == c
                acc_types['accuracy_{}'.format(int(c))] = \
                    float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask))
                loss_types['loss_reg_{}'.format(int(c))] = \
                    float(self.mseloss(e_init[mask], energy_init[mask]).mean())
        res.update(acc_types)
        res.update(loss_types)
        return res


class ParticleKinematicsLoss(ParticleTypeAndEinitLoss):

    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleKinematicsLoss, self).__init__(cfg)
        self.cosine_similarity = nn.CosineSimilarity()
        self.bceloss = StableBCELoss()
        self.margin = np.pi / 180.0

    def forward(self, out, type_labels, energy_truth, p_truth):
        logits = out['logits'][0]
        e_init = out['energy_init'][0]
        p_vec = out['p_vec'][0]
        device = logits.device

        labels = [t.view(-1, 1) for t in type_labels[0]]
        labels = torch.cat(labels, dim=1).view(-1)

        energy_init = [t.view(-1, 1) for t in energy_truth[0]]
        energy_init = torch.cat(energy_init, dim=0).view(-1, 1)
        energy_init /= 1000.0

        p_init = [t.view(-1, 3) for t in p_truth[0]]
        # print(p_init)
        p_init = torch.cat(p_init, dim=0)
        p_init /= torch.norm(p_init, dim=1).view(-1, 1)

        # labels = type_labels[0][:, 0].to(dtype=torch.long)

        # print(logits, logits.shape)
        # if logits.shape[0] != labels.shape[0]:
        #     ignore = -torch.ones(int(logits.shape[0] - labels.shape[0])).to(device, dtype=torch.long)
        #     labels = torch.cat([labels, ignore], dim=0)

        loss = self.xentropy(logits, labels)
        loss_reg = self.mseloss(e_init, energy_init).mean()
        sim = self.cosine_similarity(p_vec, p_init)
        score = (1.0 + sim) / 2.0
        # score = torch.exp( - torch.sum(
        #     torch.pow(p_vec - p_init, 2), dim=1) / (2.0 * self.margin**2 + 1e-6))
        # print(score)
        loss_direction = self.bceloss(score, torch.ones((score.shape[0], 1)).to(device))
        # loss_direction = torch.mean(torch.sum(torch.pow(p_vec - p_init, 2), dim=1))
        with torch.no_grad():
            degree = torch.acos(torch.clamp(sim, min=-1+1e-6, max=1+1e-6)) / np.pi * 180
            degree = degree.mean()
            print('Angular Separation = ', float(degree))
            print('Direction loss = ', float(loss_direction))

        loss += loss_reg
        loss += loss_direction

        pred = torch.argmax(logits, dim=1)

        accuracy = float(torch.sum(pred == labels)) / float(labels.shape[0])

        res = {
            'loss': loss,
            'loss_reg': float(loss_reg),
            'loss_direction': float(loss_direction),
            'accuracy': accuracy,
            'acc_degree': float(degree)
        }
        acc_types = {}
        loss_types = {}
        with torch.no_grad():
            for c in labels.unique():
                mask = labels == c
                acc_types['accuracy_{}'.format(int(c))] = \
                    float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask))
                loss_types['loss_reg_{}'.format(int(c))] = \
                    float(self.mseloss(e_init[mask], energy_init[mask]).mean())
        res.update(acc_types)
        res.update(loss_types)
        return res
