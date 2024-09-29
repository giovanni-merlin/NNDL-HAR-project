"""
This script contains the definition of the Inception-v4 model.
The structure of the model is:
- Stem
- 4 x Inception A
- Reduction A
- 7 x Inception B
- Reduction B
- 3 x Inception C
- Avg Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d, ZeroPad2d, Dropout, ReLU

class Conv2d_bn(Module):

    def __init__(self, in_filters, out_filters, kernel_size, strides, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=strides, padding=padding)
        self.relu = nn.ReLU()

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)
    
# Stem

class StemBlock(Module):

    def __init__(self):
        super().__init__()
        self.first_block = Sequential(
            Conv2d_bn(in_filters=1, out_filters=32, kernel_size=3, strides=2, padding="valid"), #(169,49)
            Conv2d_bn(in_filters=32, out_filters=32, kernel_size=3, strides=1, padding="valid"), # 167,47
            Conv2d_bn(in_filters=32, out_filters=64, kernel_size=3, strides=1, padding="same"), # 167,47
        )
        self.first_left = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.first_right = Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=2, padding="valid") # 83,23
        self.second_left =  Sequential(
            Conv2d_bn(in_filters=160, out_filters=64, kernel_size=1, strides=1, padding="same"), # 83,23 
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="valid"), # 81,21
        )
        self.second_right =  Sequential(
            Conv2d_bn(in_filters=160, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=64, kernel_size=(7, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=64, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="valid"),
        )
        self.third_left = Conv2d_bn(in_filters=192, out_filters=192, kernel_size=3, strides=2, padding="valid") # 40,10
        self.third_right = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.first_block(x)
        x_l = self.first_left(x)
        x_r = self.first_right(x)
        x = torch.cat([x_l, x_r], axis=1)
        x_l = self.second_left(x)
        x_r = self.second_right(x)
        x = torch.cat([x_l, x_r], axis=1)
        x_l = self.third_left(x)
        x_r = self.third_right(x)
        x = torch.cat([x_l, x_r], axis=1)
        return x
    
# Inception A

class InceptionA(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=96, kernel_size=1, strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=96, kernel_size=1, strides=1, padding="same")
        self.three_by_three_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="same"),
        )
        self.five_by_five =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="same"),
            Conv2d_bn(in_filters=96, out_filters=96, kernel_size=3, strides=1, padding="same"),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, x):
        x_1 = self.avg_block(x)
        x_2 = self.one_by_one_block(x)
        x_3 = self.three_by_three_block(x)
        x_4 = self.five_by_five(x)
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        return x

# Inception B

class InceptionB(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=128, kernel_size=1, strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=1, strides=1, padding="same")

        self.seven_by_seven_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(7, 1), strides=1, padding="same"),
        )

        self.thirteen_by_thirteen_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=192, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(7, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=224, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(7, 1), strides=1, padding="same"),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x_1 = self.avg_block(x)
        x_2 = self.one_by_one_block(x)
        x_3 = self.seven_by_seven_block(x)
        x_4 = self.thirteen_by_thirteen_block(x)
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        return x

# Inception C

class InceptionC(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=1, strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=1, strides=1, padding="same")

        self.branch_a =  Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=1, strides=1, padding="same")
        self.branch_a_left = Conv2d_bn(in_filters=384, out_filters=256, kernel_size=(1, 3), strides=1, padding="same")
        self.branch_a_right = Conv2d_bn(in_filters=384, out_filters=256, kernel_size=(3, 1), strides=1, padding="same")

        self.branch_b =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=384, out_filters=448, kernel_size=(1, 3), strides=1, padding="same"),
            Conv2d_bn(in_filters=448, out_filters=512, kernel_size=(3, 1), strides=1, padding="same"),
        )

        self.branch_b_left = Conv2d_bn(in_filters=512, out_filters=256, kernel_size=(1, 3), strides=1, padding="same")
        self.branch_b_right = Conv2d_bn(in_filters=512, out_filters=256, kernel_size=(3, 1), strides=1, padding="same")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x_1 = self.avg_block(x)
        x_2 = self.one_by_one_block(x)
        x_a = self.branch_a(x)
        x_3 = self.branch_a_left(x_a)
        x_4 = self.branch_a_right(x_a)
        x_b = self.branch_b(x)
        x_5 = self.branch_b_left(x_b)
        x_6 = self.branch_b_right(x_b)
        x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)
        return x

# Reduction A

class ReductionA(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=0) # 19,4
        self.central_block = Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=3, strides=2, padding="valid")
        self.right_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"), 
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=3, strides=1, padding="same"),  
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=3, strides=2, padding="valid"), #for the mthe padding is same also here?!
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x_1 = self.max_pool(x)
        x_2 = self.central_block(x)
        x_3 = self.right_block(x)
        x = torch.cat([x_1, x_2, x_3], axis=1)
        return x

# Reduction B

class ReductionB(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=0) # 9,
        self.central_block = Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=192, kernel_size=3, strides=2, padding="valid"),
        )
        self.right_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=256, out_filters=256, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=256, out_filters=320, kernel_size=(7, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=320, out_filters=320, kernel_size=3, strides=2, padding="valid"),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x_1 = self.max_pool(x)
        x_2 = self.central_block(x)
        x_3 = self.right_block(x)
        x = torch.cat([x_1, x_2, x_3], axis=1)
        return x
    
class Inceptionv4(Module):

    def __init__(self, hidden_dimension):
        super().__init__()
        self.network = Sequential(
            StemBlock(), # 40,10
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384), # 40,10
            ReductionA(384), # 19,4
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024), # 19,4
            ZeroPad2d((0,1,0,1)), # 20,5
            ReductionB(1024), # 9,2
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536), # 9,2
        )
        self.drop = Dropout(0.2)
        self.classification_layer = nn.Sequential(
            Linear(27648, 4*hidden_dimension),
            ReLU(inplace=True),
            Linear(4*hidden_dimension, hidden_dimension)
        )
        # forse è un po' troppo? Ridurre la dimensione x?
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.network(x)
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        y = self.classification_layer(x)
        return y

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
