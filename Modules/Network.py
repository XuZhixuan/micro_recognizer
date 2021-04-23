import torch.nn as nn

from typing import List, Dict


class Network(nn.Module):
    """ Neural Network Class
    Neural Network definition & compilation
    With Keras API used

    Attributes:
        net: Neural Network model with Keras API
    """

    def __init__(self, config: List[Dict]):
        super(Network, self).__init__()
        self.net = None
        self._make_layers(config)

    def _make_layers(self, configs: list):
        layer_class = {
            'conv2d': nn.Conv2d,
            'maxPool2d': nn.MaxPool2d,
            'bn2d': nn.BatchNorm2d,
            'gap': nn.AdaptiveAvgPool2d,
            'dense': nn.Linear,
            'flatten': nn.Flatten,
            'relu': nn.ReLU,
            'softMax': nn.Softmax
        }
        layers = []
        for config in configs:
            layer = layer_class[
                config['class']
            ](
                **config['args']
            )
            layers.append(layer)

        self.net = nn.Sequential(*layers).cuda()

    def forward(self, x):
        return self.net(x)
