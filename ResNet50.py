"""
script containing the ResNet50 model
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d, ZeroPad2d, Dropout, ReLU


# define some base classes: MainPath, IdentityBlock, ConvolutionalBlock

class MainPath(Module):

    def __init__(self, in_filters, out_filters, kernel_size, stride=1):
        # out_filters are the output filters for the 3 convolutional layers
        # kernel_size is the kernel for the 2nd convolutional layer
        super().__init__()
        F1, F2, F3 = out_filters 
        self.main_path = Sequential(
            Conv2d(in_filters, F1, kernel_size=1, stride=stride, padding='valid'),
            BatchNorm2d(F1),
            ReLU(),
            Conv2d(F1, F2, kernel_size=kernel_size, stride=1, padding='same'), # since stride=1 asymmetric padding is allowed
            BatchNorm2d(F2),
            ReLU(),
            Conv2d(F2, F3, kernel_size=1, stride=1, padding='valid'),
            BatchNorm2d(F3),
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

    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size, stride=1)
        self.relu = ReLU()

    def forward(self, x):
        y = self.relu(self.main_path(x) + x)
        return y
    

class ConvolutionalBlock(MainPath):
    # block for downsampling

    def __init__(self, in_channels, filters, kernel_size, stride=2):
        super().__init__(in_channels, filters, kernel_size, stride=stride) # NB: this has stride=2
        self.relu = ReLU()
        self.shortcut_path = Sequential(
            Conv2d(in_channels, filters[2], kernel_size=1, stride=stride), # th shortcut path has the same stride and filter dimension as the 2nd convolutional layer
            BatchNorm2d(filters[2])
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

# ResNet50 model

class ResNet50(Module):

    def __init__(self):
        super().__init__()
        self.network = Sequential(
            # STAGE 1
            Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding='valid'), # DOWNSAMPLING 167*47
            BatchNorm2d(64), # CHECK: MUST BE APPLIED TO THE CHANNEL AXIS
            MaxPool2d(kernel_size=3, stride=2), # DOWNSAMPLING 83*23
            # STAGE 2
            ConvolutionalBlock(64, [64, 64, 256], 3, stride=1), #81*21
            IdentityBlock(256, [64, 64, 256], 3),
            IdentityBlock(256, [64, 64, 256], 3),
            # STAGE 3
            ConvolutionalBlock(256, [128, 128, 512], 3), # DOWNSAMPLING 40*10
            Dropout(0.2),
            IdentityBlock(512, [128, 128, 512], 3),
            IdentityBlock(512, [128, 128, 512], 3),
            IdentityBlock(512, [128, 128, 512], 3),
            # STAGE 4
            ConvolutionalBlock(512, [256, 256, 1024], 3), # DOWNSAMPLING 19*4
            Dropout(0.2),
            IdentityBlock(1024, [256, 256, 1024], 3),
            IdentityBlock(1024, [256, 256, 1024], 3),
            IdentityBlock(1024, [256, 256, 1024], 3),
            IdentityBlock(1024, [256, 256, 1024], 3),
            IdentityBlock(1024, [256, 256, 1024], 3),
            ZeroPad2d((0,1,0,1)), # UPSAMPLING 20*5
            # STAGE 5
            ConvolutionalBlock(1024, [512, 512, 2048], 3), # DOWNSAMPLING 9*2
            Dropout(0.2),
            IdentityBlock(2048, [512, 512, 2048], 3),
            IdentityBlock(2048, [512, 512, 2048], 3),

            AvgPool2d(kernel_size=2, stride=2, ceil_mode=True) # DOWNSAMPLING 4*2, but apparently is a 6*2
        )
        self.classification_layer = Linear(24576, 5)
        self.apply(self._init_weights)

    def forward(self, x):
        #print(f"out dimensions before flatten: {self.network(x).shape}")
        # FLATTENS all dimensions except the first (batch size)
        y = t.flatten(self.network(x), start_dim=1)
        #print(f"out dimensions after flatten: {y.shape}")
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
