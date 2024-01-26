"""
Code Reference:
This script is adapted from simclr-pytorch at https://github.com/AndrewAtanov/simclr-pytorch.

For the original version of the code, please refer to the mentioned repository.
"""

from models import encoder
from models import losses
from models import resnet
from models import resnet_lc
from models import ssl

REGISTERED_MODELS = {
    'sim-clr': ssl.SimCLR,
    'eval': ssl.SSLEval,
    'semi-supervised-eval': ssl.SemiSupervisedEval,
}
