from collections import OrderedDict
import torch.nn as nn
from .tools import Identity, Linear_fw
class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
        '''
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append((f'linear_{i}', Linear_fw(input_dim, layers[i])))
            self.all_layers.append((f'relu_{i}' ,nn.ReLU()))
            if use_bn:
                self.all_layers.append((f'bn_{i}', nn.BatchNorm1d(layers[i])))
            if dropout > 0:
                self.all_layers.append((f'drop_{i}', nn.Dropout(dropout)))
            input_dim = layers[i]
        
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        
        self.fc_out = Linear_fw(layers[-1], output_dim)
        self.module = nn.Sequential(OrderedDict(self.all_layers))
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat
