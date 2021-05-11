from typing import List, Dict

from torch.nn import Module, Sequential

from container import Container


class Network(Module):
    """ Neural Network Class
    Neural Network definition & compilation
    With Keras API used

    Attributes:
        net: Neural Network model with Keras API
    """

    def __init__(self, app: Container, config: List[Dict]):
        """
        Initializer of Network
        Args:
            config:
        """
        super(Network, self).__init__()
        self.net = None
        self._make_layer = app.nn_make
        self._make_layers(config)

    def _make_layers(self, configs: list):
        """
        Make layers from config list
        Args:
            configs:

        Returns:

        """
        layers = []
        for config in configs:
            layer = self._make_layer(config['class'], config['args'])
            layers.append(layer)

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)
