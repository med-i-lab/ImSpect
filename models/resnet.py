"""
Code Reference:
This script is adapted from simclr-pytorch at https://github.com/AndrewAtanov/simclr-pytorch.

For the original version of the code, please refer to the mentioned repository.
"""

import torch.nn as nn
import torchvision.models as models
import torch


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers, cifar_head=False, param='net', hparams=None):
        super().__init__(block, layers)
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
            self.relu = nn.ReLU(inplace=True)
        self.hparams = hparams
        self.param = param

        print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        con = self.layer4(x)

        x = self.avgpool(con)
        x = torch.flatten(x, 1)
        if self.param == 'cam':
            return x, con
        return x

class ResNet18(ResNetEncoder):
    def __init__(self, cifar_head=True, param='net'):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], cifar_head=cifar_head, param=param)


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head, hparams=hparams)
        

model_dict = {
    'resnet18':  512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
}
       
class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet18', num_classes=2):
        super(LinearClassifier, self).__init__()
        feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
