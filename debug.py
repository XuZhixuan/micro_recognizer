from torchsummary import summary
from Modules import *


def debug():
    net = Network()
    summary(net, input_size=(1, 974, 974), device='cuda')
    pass


if __name__ == '__main__':
    debug()
