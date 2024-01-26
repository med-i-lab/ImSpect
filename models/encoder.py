"""
Code Reference:
This script is adapted from simclr-pytorch at https://github.com/AndrewAtanov/simclr-pytorch.

For the original version of the code, please refer to the mentioned repository.
"""

import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os
import pdb

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self, hparams, param='net'):
        super().__init__()
        self.param = param

        if hparams.arch == 'ResNet50':
            cifar_head = (hparams.data == 'cifar')
            self.convnet = models.resnet.ResNet50(cifar_head=cifar_head, hparams=hparams)
            self.encoder_dim = 2048
        elif hparams.arch == 'resnet18':
            self.convnet = models.resnet.ResNet18(cifar_head=(hparams.data == 'cifar'), param=param)
            self.encoder_dim = 512
        else:
            raise NotImplementedError

        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)

        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

    def forward(self, x, out='z'):
        if self.param == 'cam':
            h, d = self.convnet(x)
            return h, d
        else:
            h = self.convnet(x)
            if out == 'h':
                return h
            return self.projection(h)
