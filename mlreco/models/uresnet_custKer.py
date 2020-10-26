import MinkowskiEngine as ME
from torch import nn
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiModuleBase
import torch 

from random import randint

class ConcatTable(nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)

    def forward(self, input):
        return [module(input) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self
    
class AddTable(nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)

    def forward(self, input):
        coordinates = input[0].coords
        features = sum([i.feats for i in input])
        r = input[0]+input[1]
        #return(ME.SparseTensor(coords = coordinates, feats = features, tensor_stride = input[0].tensor_stride))
        return r
    
    def input_spatial_size(self, out_size):
        return out_size
    
    
class Identity(ME.MinkowskiNetwork):
    def forward(self, input):
        return input

def RandomCustomKernel() :
    offset = []
    for i in range(3) : 
        for j in range(3) : 
            r = randint(0, 2)
            for k in range(3) :
                if k != r : 
                    offset.append([i, j, k])
    custom_kernel = ME.KernelGenerator(kernel_size = 3, stride = 1, region_type=ME.RegionType.HYPERCROSS, dimension = 3, region_offsets = torch.IntTensor(offset))
    
    return(custom_kernel)    
    
class MinkowskiLeakyReLU(MinkowskiModuleBase):
    MODULE = nn.LeakyReLU
    
class UResNet(ME.MinkowskiNetwork):

    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (float,), (3, 1)]
    ]

    MODULES = ['uresnet_lonely']

    def __init__(self, cfg, name="uresnet_lonely_custKer"):
        self._model_config = cfg[name]
        self._dimension = self._model_config.get('data_dim', 3)
        
        super(UResNet, self).__init__(self._dimension)
        
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        spatial_size = self._model_config.get('spatial_size', 768)
        num_classes = self._model_config.get('num_classes', 5)

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None
        leakiness = 0

        def block(m, a, b, num):  # ResNet style blocks
            
            module = nn.Sequential(ConcatTable(Identity(self._dimension) if a == b else ME.MinkowskiLinear(a, b), \
nn.Sequential( \
ME.MinkowskiBatchNorm(num_features = a), MinkowskiLeakyReLU(), \
ME.MinkowskiConvolution(a, b, kernel_generator=RandomCustomKernel(), stride=1, dimension=self._dimension), \
ME.MinkowskiBatchNorm(num_features = b), MinkowskiLeakyReLU(), \
ME.MinkowskiConvolution(b, b, kernel_generator=RandomCustomKernel(), stride=1, dimension=self._dimension)) \
), AddTable())
            m.add_module(f'block_{num}', module)
            
        self.input = ME.MinkowskiConvolution(nInputFeatures, m, kernel_size=3, stride=1, dimension=self._dimension)
                  
        # Encoding
        self.bn = nn.Sequential(ME.MinkowskiBatchNorm(num_features = nPlanes[0]), MinkowskiLeakyReLU())
        '''
        self.encoding_block = nn.Sequential()
        self.encoding_conv = nn.Sequential()
        
        for i in range(num_strides):
            module = nn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            self.encoding_block.add_module(f'encod_block_{i}', module)
            
            module2 = nn.Sequential()
            if i < num_strides-1:
                module2 = nn.Sequential(ME.MinkowskiBatchNorm(num_features = nPlanes[i]), \
                            MinkowskiLeakyReLU(), \
                            ME.MinkowskiConvolution(nPlanes[i], nPlanes[i+1], \
                        kernel_size = downsample[0], stride = downsample[1], dimension = self._dimension))
                
            self.encoding_conv.add_module(f'encod_block_conv_{i}', module2)
        '''
        
        self.encoding_block = []
        self.encoding_conv = []
        
        
        for i in range(num_strides):
            mod = nn.Sequential()
            for _ in range(reps) : 
                block(mod, nPlanes[i], nPlanes[i], _)
            self.encoding_block.append(mod)
        
            m_list = []
            if i < num_strides-1:
                m_list.append(ME.MinkowskiBatchNorm(nPlanes[i]))
                
                m_list.append(MinkowskiLeakyReLU())
                
                m_list.append(ME.MinkowskiConvolution(
                    in_channels=nPlanes[i],
                    out_channels=nPlanes[i+1],
                    kernel_size=downsample[0], stride=2, dimension=self._dimension))
                
            m_list= nn.Sequential(*m_list)
            self.encoding_conv.append(m_list)    
            
        self.encoding_block = nn.Sequential(*self.encoding_block)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)

        # Decoding
        
        self.decoding_conv, self.decoding_blocks = nn.Sequential(), nn.Sequential()
        for i in range(num_strides-2, -1, -1):
            module1 = nn.Sequential(ME.MinkowskiBatchNorm(num_features = nPlanes[i+1]), \
                    MinkowskiLeakyReLU(), \
                    ME.MinkowskiConvolutionTranspose(in_channels=nPlanes[i+1], out_channels=nPlanes[i], \
                    kernel_size=downsample[0], stride=downsample[1], generate_new_coords = False, \
                    dimension=self._dimension))
            self.decoding_conv.add_module(f'decod_block_deconv_{i}', module1)
            module2 = nn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i] * (2 if j == 0 else 1), nPlanes[i], j)
            self.decoding_blocks.add_module(f'decod_block_{i}', module2)

        self.bnr = nn.Sequential(ME.MinkowskiBatchNorm(num_features = m), ME.MinkowskiReLU())
        self.linear = ME.MinkowskiLinear(m, num_classes)

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        coords = input[0][:, 0:4].float()
        feats = input[0][:, 4].float().reshape([-1, 1])
        x = ME.SparseTensor(feats = feats, coords=coords)
                  
        feature_maps = [x]
        x = self.input(x) 
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)
        
        # U-ResNet decoding
        for i, layer in enumerate(self.decoding_conv):
            encoding_block = feature_maps[-i-2]
            x = layer(x)
            x = ME.cat(encoding_block, x)
            x = self.decoding_blocks[i](x)

        x = self.bnr(x)
        x_seg = self.linear(x)  # Output of UResNet

        res = {
            'segmentation': [x_seg.F],
        }
        return res
                
                  

class SegmentationLoss(nn.modules.loss._Loss):
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (int,), (3, 1)]
    ]

    def __init__(self, cfg, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['uresnet_lonely_custKer']
        self._num_classes = self._cfg.get('num_classes', 5)
        self._alpha = self._cfg.get('alpha', 1.0)
        self._beta = self._cfg.get('beta', 1.0)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')


    def forward(self, result, label, weights=None):
#        print("result : ", result)
#       print("label : ", label)

        assert len(result['segmentation']) == len(label)
        batch_ids = [d[:, 0] for d in label]
        uresnet_loss, uresnet_acc = 0., 0.
        uresnet_acc_class = [0.] * self._num_classes
        count_class = [0.] * self._num_classes
        count = 0
        class_acc_list = [[0 for i in range(self._num_classes)] for j in range(self._num_classes)]
        class_count_list = [0 for i in range(self._num_classes)]
        
        for i in range(len(label)):
            for b in batch_ids[0].unique():
                batch_index = batch_ids[i] == b

                event_segmentation = result['segmentation'][0][batch_index]  # (N, num_classes)
                event_label = label[i][batch_index][:, -1][:, None]  # (N, 1)
                event_label = torch.squeeze(event_label, dim=-1).long()

                # check and warn about invalid labels
                unique_label,unique_count = torch.unique(event_label,return_counts=True)
                if (unique_label >= self._num_classes).long().sum():
                    print('Invalid semantic label found (will be ignored)')
                    print('Semantic label values:',unique_label)
                    print('Label counts:',unique_count)
                # Now mask to compute the rest of UResNet loss
                mask = event_label < self._num_classes
                event_segmentation = event_segmentation[mask]
                event_label = event_label[mask]

                if event_label.shape[0] > 0:  # FIXME how to handle empty mask?
                    # Loss for semantic segmentation
                    loss_seg = self.cross_entropy(event_segmentation, event_label)
                    uresnet_loss += torch.mean(loss_seg)

                    # Accuracy for semantic segmentation
                    with torch.no_grad():
                        predicted_labels = torch.argmax(event_segmentation, dim=-1)
                        #print("predicted_labels : ", predicted_labels)
                        #print("event_label : ", event_label)
                        acc = predicted_labels.eq_(event_label).sum().item() / float(predicted_labels.nelement())
                        uresnet_acc += acc

                        # Class accuracy
                        for c in range(self._num_classes):
                            class_mask = event_label == c
                            class_count = class_mask.sum().item()
                            if class_count > 0:
                                uresnet_acc_class[c] += predicted_labels[class_mask].sum().item() / float(class_count)
                                count_class[c] += 1

                    count += 1
                    

                
                # Accuracy for semantic segmentation
                with torch.no_grad():
                    predicted_labels = torch.argmax(event_segmentation, dim=-1)

                    acc = float((predicted_labels == event_label).sum()/float(len(event_label)))
                    uresnet_acc += acc
                    count += 1
                    for c1 in range(self._num_classes):
                        class_mask = event_label == c1
                        if float(class_mask.sum()) != 0.0 : 
                            class_count_list[c1] += 1
                            for c2 in range(self._num_classes):
                                class_acc_list[c1][c2] += (predicted_labels[class_mask] == c2).sum()/float((class_mask.sum()))
                                
                            
        results = {
            'accuracy': uresnet_acc/count,
            'loss': uresnet_loss/count
        }
        for c in range(self._num_classes):
            if count_class[c] > 0:
                results['accuracy_class_%d' % c] = uresnet_acc_class[c]/count_class[c]
            else:
                results['accuracy_class_%d' % c] = -1.
                
        for i in range(self._num_classes) : 
            for j in range(self._num_classes) : 
                results[f'class_acc_list_{i}_{j}'] = class_acc_list[i][j]/class_count_list[i]
                
        return results
    """
    def __init__(self, cfg, name = 'segmentation_loss') : 
        super(SegmentationLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss(reduction = 'none')
        
    def forward(self, out, labels) : 
        
        out = out['segmentation'][0]
        assert len(out) == len(labels[0])
        labels = labels[0]
        labels = torch.tensor(labels).to(dtype=torch.long)
        
        batch_ids = labels[:, 0].unique()
        
        uresnet_loss = 0.0
        uresnet_acc = 0.0
        
        num_class = len(labels[:, 4].unique())
        count = 0
        
        class_acc = [[0 for i in range(num_class)] for j in range(num_class)]
        class_count = [0 for i in range(num_class)]
        
        for b in batch_ids :
            
            batch_index = labels[:, 0] == b
            batch_labels = labels[batch_index, 4]
            batch_predictions = out[batch_index, :]
            
            loss_seg = self.xentropy(batch_predictions, batch_labels)
            uresnet_loss += torch.mean(loss_seg)
            
            # Accuracy for semantic segmentation
            with torch.no_grad():
                predicted_labels = torch.argmax(batch_predictions, dim=-1)
                acc = float((predicted_labels == batch_labels).sum()/float(len(batch_labels)))
                uresnet_acc += acc
                count += 1

                for c1 in range(num_class):
                    class_mask = batch_labels == c1
                    if float(class_mask.sum()) != 0.0 : 
                        class_count[c1] += 1
                        for c2 in range(num_class):
                            class_acc[c1][c2] += (predicted_labels[class_mask] == c2).sum()/float((class_mask.sum()))

                            
        res = {
            'loss': uresnet_loss/count,
            'accuracy': uresnet_acc/count
        }

        '''
        for i in range(num_class) : 
            res[f'class_count_{i}'] = class_count[i]
            
        for i in range(num_class) : 
            for j in range(num_class) : 
                res[f'class_acc_cum_{i}_{j}'] = class_cum_acc[i][j]
        '''
        
        for i in range(num_class) : 
            for j in range(num_class) : 
                res[f'class_acc_{i}_{j}'] = class_acc[i][j]/class_count[i]
                                            
        return(res)