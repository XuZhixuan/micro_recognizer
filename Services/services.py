from abc import abstractmethod

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
        self.app.singleton('train_summary', SummaryWriter(
            self.app.config('logs.summary.dir')
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
        from torch.nn import DataParallel

        model = Network(self.app, self.app.config('training.define')).cuda(0)
        # model = DataParallel(model, device_ids=[0, 1, 2])

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
