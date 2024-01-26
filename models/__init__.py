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
