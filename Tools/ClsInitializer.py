from torch import nn
from torch import optim


class ClsInitializer:
    def __init__(self):
        pass

    def __call__(self, name: str, args: dict):
        return self.init(name, args)

    @staticmethod
    def init(name: str, args: dict = None):
        """
        Find a class related to reference and instantiate it
        Args:
            name: The name of class in torch.nn
            args: The args used to instantiate

        Returns:
            An object
        """
        cls = None
        if args is None:
            args = {}

        if name in nn.modules.__all__:
            cls = getattr(nn, name)
        else:
            try:
                cls = getattr(optim, name)
            except AttributeError:
                exit('No module named %s found in torch' % name)

        return cls(**args)
