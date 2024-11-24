"""
script containing the ResNet18 model
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d, ZeroPad2d, Dropout, ReLU


# define some base classes: MainPath, IdentityBlock, ConvolutionalBlock

class MainPath(Module):

    def __init__(self, in_filters, out_filters, stride, padding='valid'):
        # out_filters are the output filters for the 2 convolutional layers
        # kernel_size is the kernel for the 2nd convolutional layer
        super().__init__()
        F1, F2 = out_filters 
        self.main_path = Sequential(
            Conv2d(in_filters, F1, kernel_size=3, stride=stride, padding=padding),
            BatchNorm2d(F1),
            ReLU(),
            Conv2d(F1, F2, kernel_size=3, stride=1, padding='same'), # since stride=1 asymmetric padding is allowed
            BatchNorm2d(F2)            
            #ReLU(),
            #Conv2d(F2, F3, kernel_size=1, stride=1, padding='valid'),
            #BatchNorm2d(F3),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, Conv2d):
            t.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        y = self.main_path(x)
        return y


class IdentityBlock(MainPath):
    # by construction in the IdentityBlock the output and input dimensions are the same

    def __init__(self, in_channels, filters):
        super().__init__(in_channels, filters, stride=1, padding='same')
        self.relu = ReLU()

    def forward(self, x):
        y = self.relu(self.main_path(x) + x)
        return y
    

class ConvolutionalBlock(MainPath):
    # block for downsampling

    def __init__(self, in_channels, filters, stride=2):
        super().__init__(in_channels, filters, stride=stride) # NB: this has stride=2
        self.relu = ReLU()
        self.shortcut_path = Sequential(
            Conv2d(in_channels, filters[1], kernel_size=3, stride=stride), # th shortcut path has the same stride and filter dimension as the 2nd convolutional layer
            BatchNorm2d(filters[1])
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, t.nn.Linear):
            t.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, t.nn.Conv2d):
            t.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        y = self.relu(self.main_path(x) + self.shortcut_path(x))
        return y

# ResNet18 model

class ResNet18(Module):

    def __init__(self, hidden_dimension):
        super().__init__()
        self.network = Sequential(
            # STAGE 1
            Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding='valid'), # DOWNSAMPLING 167*47
            BatchNorm2d(64), # CHECK: MUST BE APPLIED TO THE CHANNEL AXIS
            MaxPool2d(kernel_size=3, stride=2), # DOWNSAMPLING 83*23
            # STAGE 2
            ConvolutionalBlock(64, [64, 64], stride=1), # DOWNSAMPLING 
            IdentityBlock(64, [64, 64]),
            # STAGE 3
            ConvolutionalBlock(64, [128, 128]), # DOWNSAMPLING
            Dropout(0.2),
            IdentityBlock(128, [128, 128]),
            # STAGE 4
            ConvolutionalBlock(128, [256, 256]), # DOWNSAMPLING
            Dropout(0.2),
            IdentityBlock(256, [256, 256]),
            ZeroPad2d((0,1,0,1)), # UPSAMPLING
            # STAGE 5
            ConvolutionalBlock(256, [512, 512]), # DOWNSAMPLING 9*2?
            Dropout(0.2),
            IdentityBlock(512, [512, 512]),

            AvgPool2d(kernel_size=2, stride=2, ceil_mode=True) # DOWNSAMPLING 5*1?
        )
        self.classification_layer = Linear(2560, 4*hidden_dimension)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.network(x)
        #x = x.reshape(x.shape[0], -1, 512).mean(axis=1) # axis=1 ha size 2560/512 = 5
        #x = x.reshape(x.shape[0], -1, 512)
        y = t.flatten(x, start_dim=1)
        y = self.classification_layer(y)
        return y

    def _init_weights(self, module):
        if isinstance(module, t.nn.Linear):
            t.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, t.nn.Conv2d):
            t.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
