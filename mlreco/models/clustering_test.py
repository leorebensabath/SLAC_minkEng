import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from torch.optim import SGD
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiModuleBase
import torch.nn.functional as F

from mlreco.nn.layers.nonlinearities import MinkowskiLeakyReLU
from mlreco.blocks.uresnet_blocks import Encoder, Decoder 
from mlreco.utils.metrics import ARI

import sys


class ClusteringUresnet(ME.MinkowskiNetwork) : 
    
    def __init__(self, cfg, name='clustering_uresnet_test'):
        
        self.model_config = cfg[name]
        self.D = self.model_config.get('D', 3)
        super(ClusteringUresnet, self).__init__(self.D)
        
        self.in_features = self.model_config.get('in_features', 1)
        self.depth = self.model_config.get('depth', 7)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [self.num_filters*i for i in range(1, self.depth+1)]
        self.spatial_size= self.model_config.get('spatial_size', 768)
        self.out_features_embedding = self.model_config.get('out_features_embedding', 4)
        self.out_features_seediness = self.model_config.get('out_features_seediness', 1)
        
        self.encoder = Encoder(cfg)
        self.decoder_embedding = Decoder(cfg)
        self.decoder_seediness = Decoder(cfg)
        
        self.input_layer = nn.Sequential(
            ME.MinkowskiBatchNorm(self.in_features),
            MinkowskiLeakyReLU(),
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.num_filters,
                kernel_size = 2, stride=1, dimension =self.D))
            
        self.linear_embedding = ME.MinkowskiLinear(self.num_filters, self.out_features_embedding)
        self.linear_seediness = ME.MinkowskiLinear(self.num_filters, self.out_features_seediness)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x) : 
        
        coords = x[0][:, 0:4]
        feats = x[0][:, 4].float().reshape([-1, 1])
        
        x = ME.SparseTensor(feats = feats, coords=coords)
        
        out = self.input_layer(x)
        out_encoder, encoding_features = self.encoder(out)
        
        out_embedding = self.decoder_embedding(out_encoder, encoding_features)
        out_seediness = self.decoder_seediness(out_encoder, encoding_features)
        out_embedding = self.linear_embedding(out_embedding)
        out_seediness = self.linear_seediness(out_seediness)
        
        out = {'logits_embedding' : torch.cat([out_embedding.F[:, :3], self.tanh(out_embedding.F[:, 3:4])], dim = 1), 
               'logits_seediness' : self.sigmoid(out_seediness.F)}

        return(out)
    
class ClusteringLoss(nn.Module) : 
    
    def __init__(self, cfg, name = 'clustering_uresnet_test') :
#        print("cfg for loss : ", cfg)
        super(ClusteringLoss, self).__init__()
        self.loss_config = cfg[name]
        self.spatial_size = self.loss_config.get('spatial_size', 768)
        
    def forward(self, out, labels) :
        labels = labels[0]
        batch_ids = labels[:, 0].unique().to(dtype=torch.long)
        output_embedding = out['logits_embedding']
        output_seediness = torch.reshape(out['logits_seediness'], (-1,))
        
        coords_embedding = (output_embedding[:, :3] + (labels[:, 1:4] - (self.spatial_size/2))/(self.spatial_size/2)).cuda()
        sigma_embedding = output_embedding[:, 3]
        instance_labels = (labels[:, 5].to(dtype=torch.long))

        clustering_loss = 0.0
        clustering_acc = 0.0
        clustering_acc_xentropy = 0.0
        
        count = 0
        for b in batch_ids :
            
            batch_index = labels[:, 0] == b
            batch_coords_embedding = coords_embedding[batch_index, :]
            batch_sigma_embedding = sigma_embedding[batch_index]
            batch_output_seediness = output_seediness[batch_index]
            batch_instance_labels = instance_labels[batch_index]
            batch_instance = batch_instance_labels.unique()
            num_instance = len(batch_instance)

            instance_centers = torch.zeros([num_instance, 3], dtype=torch.float).cuda()
            instance_sigmas = torch.zeros(num_instance).cuda()

            N = len(coords_embedding[batch_index, :])
            
            for i in range(num_instance) : 
                k = batch_instance[i]
                instance_mask = batch_instance_labels == k
                instance_centers[i, :] = torch.sum(batch_coords_embedding[instance_mask, :], dim = 0)/(float(instance_mask.sum()))
                instance_sigmas[i] = torch.sum(batch_sigma_embedding[instance_mask])/(float(instance_mask.sum()))
                
            maskLoss = 0.0
            smoothingLoss = 0.0
            seedinessLoss = 0.0
            
            for i in range(num_instance) :
                k = batch_instance[i]
                instance_mask = batch_instance_labels == k
                
                p_k = torch.exp(-torch.max(torch.min(torch.norm(batch_coords_embedding - instance_centers[i], dim = 1)**2/(2*(instance_sigmas[i])**2), torch.tensor(100.0).cuda()), torch.tensor(0.0000001).cuda()))
                #print("torch.norm(batch_coords_embedding - instance_centers[i], dim = 1)**2 : ", torch.norm(batch_coords_embedding - instance_centers[i], dim = 1)**2)
                #print("(2*(instance_sigmas[i])**2) : ", (2*(instance_sigmas[i])**2))
                #print("p_k : ", p_k)
                maskLoss += -torch.sum(instance_mask*torch.log(p_k) + (~instance_mask)*torch.log(1 - p_k))/N
                
                #print("batch_sigma_embedding[instance_mask] : ", batch_sigma_embedding[instance_mask])
                #print("instance_sigmas[i] : ", instance_sigmas[i])
                smoothingLoss += torch.sum(torch.abs(batch_sigma_embedding[instance_mask] - instance_sigmas[i]))/(float(instance_mask.sum()))               
                
                #print("batch_output_seediness : ", batch_output_seediness)
                seedinessLoss += torch.norm(batch_output_seediness[instance_mask] - p_k[instance_mask])**2/(float(instance_mask.sum()))

            maskLoss = maskLoss/num_instance
            smoothingLoss = smoothingLoss/num_instance
            seedinessLoss = seedinessLoss/num_instance
            
            interCLusterLoss = 0.0
            interCLusterParam = 0.5
            for i in range(num_instance) : 
                for j in range(i) :
                    interCLusterLoss += max(0, 2*interCLusterParam - torch.norm(instance_centers[i] - instance_centers[j]))**2
            interCLusterLoss = interCLusterLoss/(num_instance*(num_instance-1))
            
            print("mask loss : ", maskLoss)
            print("Smoothing loss : ", smoothingLoss)
            print("Inter cluster loss : ", interCLusterLoss)
            print("seedinessLoss : ", seedinessLoss)

            embeddingLoss = maskLoss + smoothingLoss + interCLusterLoss
            with torch.no_grad():
                batch_pred = torch.zeros(N)
                for i in range(N) : 
                    batch_pred[i] = batch_instance[torch.argmin(torch.norm(batch_coords_embedding[i, :] - instance_centers,  dim = 1))]
#                    print("pred : ", batch_pred[i], " ------ ", "truth : ", batch_instance_labels[i])
                
                clustering_acc += ARI(batch_pred.cpu(), batch_instance_labels.cpu())
                
                #print("batch_pred : ", batch_pred)
                #print("batch_instance_labels : ", batch_instance_labels)
                clustering_acc_xentropy += torch.sum(batch_pred.cpu() == batch_instance_labels.cpu())/float(N)
            #print(seedinessLoss)
            clustering_loss += (embeddingLoss + seedinessLoss)
            
            #clustering_loss += embeddingLoss
            count+=1
        print("clustering_acc_xentropy : ", clustering_acc_xentropy/count)
        res = {'loss' : clustering_loss/count, 
               'accuracy' : clustering_acc/count}
        
        return(res)
    
    '''
    def forward(self, out, labels) :
        batch_ids = labels[:, 0].unique().to(dtype=torch.long)
        output_embedding = out['logits_embedding']
        output_seediness = torch.reshape(out['logits_seediness'], (-1,))
        coords_embedding = (output_embedding[:, :3] + (labels[:, 1:4] - (self.spatial_size/2))/(self.spatial_size/2)).cuda()
        sigma_embedding = output_embedding[:, 3]

        instance_labels = (labels[:, 5].to(dtype=torch.long))
        semantic_labels = (labels[:, 9].to(dtype=torch.long))    
        
        clustering_loss = 0.0
        clustering_acc = 0.0
        clustering_acc_xentropy = 0.0
        
        count = 0
        
        p0 = 0.5
        s0 = 
        for b in batch_ids :
            
            batch_index = labels[:, 0] == b
            batch_coords_embedding = coords_embedding[batch_index, :]
            batch_sigma_embedding = sigma_embedding[batch_index]
            batch_output_seediness = output_seediness[batch_index]
    
            batch_instance_labels = instance_labels[batch_index]
            batch_semantic_labels = semantic_labels[batch_index]
            
            batch_instance = batch_instance_labels.unique()
            batch_semantic = batch_semantic_labels.unique()

            num_instance = len(batch_instance)
            num_semantic = len(batch_semantic)

            for i in range(num_semantic) : 
                semantic_class = batch_semantic[i]
                mask_semantic_class = batch_semantic_labels == semantic_class
                semantic_seediness = batch_output_seediness[mask_semantic_class]
                semantic_coords_embedding = batch_coords_embedding[mask_semantic_class, :]
                semantic_sigma_embedding = batch_sigma_embedding[mask_semantic_class]
                
                while (torch.sum(mask_instance) != 0)&(torch.sum(semantic_seediness > s0)!=0): 
                    argmax_semantic_seediness = torch.argmax(semantic_seediness)
                    max_centroid = semantic_coords_embedding[argmax_semantic_seediness, :]
                    max_sigma = semantic_sigma_embedding[argmax_semantic_seediness]

                    p_k = torch.exp(-torch.max(torch.min(torch.norm(semantic_coords_embedding - max_centroid, dim = 1)**2/(2*(max_sigma)**2), torch.tensor(100.0).cuda()), torch.tensor(0.0000001).cuda()))
                    
                    mask_instance = p_k < p0
                    mask_semantic_class = mask_instance
                    semantic_seediness = semantic_seediness[mask_semantic_class]
                    semantic_coords_embedding = semantic_coords_embedding[mask_semantic_class, :]
                    semantic_sigma_embedding = semantic_sigma_embedding[mask_semantic_class]
    '''