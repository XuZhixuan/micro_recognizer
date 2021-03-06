from abc import abstractmethod

import torch

from container import Container


class ServiceProvider:
    """
    Base abstract Service Provider class
    The Service provider class is used to register a service to the app container
    """

    def __init__(self, app: Container):
        self.app = app

    @abstractmethod
    def register(self):
        """
        Register method
        Register & bind service instance to specify name or class reference
        """
        raise NotImplementedError

    @abstractmethod
    def boot(self):
        """
        Boot method
        Called after all services registered, to perform action required other services
        """
        raise NotImplementedError


class HelperFunctionsServiceProvider(ServiceProvider):
    def register(self):
        import helper
        self.app.singleton('helper', helper)

    def boot(self):
        pass


class LayerMakeServiceProvider(ServiceProvider):
    def register(self):
        from Tools import ClsInitializer
        self.app.singleton('nn_make', ClsInitializer())

    def boot(self):
        pass


class SummaryServiceProvider(ServiceProvider):
    def register(self):
        from torch.utils.tensorboard import SummaryWriter
        import os

        dir_name = self.app.config('logs.summary.dir') + self.app.helper.time_name() + '/'
        os.mkdir(dir_name)

        self.app.singleton('train_summary', SummaryWriter(
            dir_name
        ))

        from torchsummary import summary
        self.app.singleton('network_summary', summary)

    def boot(self):
        pass


class ImageLoaderServiceProvider(ServiceProvider):
    def register(self):
        from Tools import ImageLoader
        self.app.singleton(ImageLoader, ImageLoader(self.app, size=(256, 256)))

    def boot(self):
        pass


class DataServicesProvider(ServiceProvider):
    def register(self):
        pass

    def boot(self):
        from Source import Source
        self.app.singleton(
            Source,
            self.app.config('data.source')(
                **dict({'app': self.app}, **self.app.config('data.kwargs'))
            )
        )

        self.app.set_alias(Source, 'source')


class NetworkServiceProvider(ServiceProvider):
    def register(self):
        from Modules import Network
        # model = Network(
        #     self.app,
        #     self.app.config('training.define'),
        # )
        # model = DataParallel(model, device_ids=[0, 1, 2, 3])

        # from torchvision import models
        # from torch import nn
        # from torch.nn import DataParallel
        # model = models.resnet18(pretrained=True)
        # model.conv1 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=model.conv1.out_channels,
        #     kernel_size=model.conv1.kernel_size,
        #     stride=model.conv1.stride,
        #     padding=model.conv1.padding,
        #     bias=model.conv1.bias
        # )
        # model.fc = nn.Linear(model.fc.in_features, 1)
        # model.cuda()
        # model = DataParallel(model, device_ids=[0, 1, 2, 3])

        model = torch.load('./storage/bin/model-best.pth').cpu()  # .cuda()
        # model = torch.nn.DataParallel(model, [0, 1])

        self.app.singleton(Network, model)
        self.app.set_alias(Network, 'model')

    def boot(self):
        pass


class TrainingServiceProvider(ServiceProvider):
    def register(self):
        self.app.singleton('loss_function', None)
        self.app.singleton('optimizer', None)

    def boot(self):
        loss = self.app.nn_make(
            self.app.config('training.loss_function.name'),
            self.app.config('training.loss_function.args')
        )

        optimizer = self.app.nn_make(
            self.app.config('training.optimizer.name'),
            dict({'params': self.app.model.parameters()}, **self.app.config('training.optimizer.args'))
        )

        self.app.singleton('loss_function', loss)
        self.app.singleton('optimizer', optimizer)


class RedirectPrintServiceProvider(ServiceProvider):
    def register(self):
        from Tools import UnbufferedLogger
        self.app.singleton('redirect_print', UnbufferedLogger(self.app))

    def boot(self):
        import sys
        sys.stdout = self.app.redirect_print


class LRScheduleServiceProvider(ServiceProvider):
    def register(self):
        pass

    def boot(self):
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.ExponentialLR(self.app.optimizer, 0.95)

        self.app.singleton('lr_scheduler', scheduler)
