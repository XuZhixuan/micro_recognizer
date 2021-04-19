import torch
import torch.nn as nn
import torch.nn.functional as f


class Network(nn.Module):
    """ Neural Network Class
    Neural Network definition & compilation
    With Keras API used

    Attributes:
        net: Neural Network model with Keras API
    """

    def __init__(self):
        super(Network, self).__init__()
        configs = [
            # Conv block 1
            {'class': 'conv2d', 'args': {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3}},
            {'class': 'relu', 'args': {'inplace': True}},
            {'class': 'conv2d', 'args': {'in_channels': 16, 'out_channels': 16, 'kernel_size': 3}},
            {'class': 'relu', 'args': {'inplace': True}},
            {'class': 'maxPool2d', 'args': {'kernel_size': 2}},
            # Conv block 2
            {'class': 'conv2d', 'args': {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3}},
            {'class': 'relu', 'args': {'inplace': True}},
            {'class': 'conv2d', 'args': {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3}},
            {'class': 'relu', 'args': {'inplace': True}},
            {'class': 'maxPool2d', 'args': {'kernel_size': 2}},
            # # Conv block 3
            # {'class': 'conv2d', 'args': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3}},
            # {'class': 'relu', 'args': {'inplace': True}},
            # {'class': 'conv2d', 'args': {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3}},
            # {'class': 'relu', 'args': {'inplace': True}},
            # {'class': 'maxPool2d', 'args': {'kernel_size': 2}},
            {'class': 'gap', 'args': {'output_size': 20}},
            {'class': 'flatten', 'args': {}},
            # Dense block
            {'class': 'dense', 'args': {'in_features': 12800, 'out_features': 20}},
            {'class': 'relu', 'args': {'inplace': True}},
            {'class': 'dense', 'args': {'in_features': 20, 'out_features': 20}},
            {'class': 'relu', 'args': {'inplace': True}},
            {'class': 'dense', 'args': {'in_features': 20, 'out_features': 1}}
        ]
        self.net = None
        self._make_layers(configs)

    def _make_layers(self, configs: list):
        layer_class = {
            'conv2d': nn.Conv2d,
            'maxPool2d': nn.MaxPool2d,
            'bn2d': nn.BatchNorm2d,
            'gap': nn.AdaptiveAvgPool2d,
            'dense': nn.Linear,
            'flatten': nn.Flatten,
            'relu': nn.ReLU
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
