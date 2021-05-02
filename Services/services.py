from container import Container

from abc import abstractmethod


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


class LayerMakeServiceProvider(ServiceProvider):
    def register(self):
        from Tools import ClsInitializer
        self.app.singleton('nn_make', ClsInitializer())

    def boot(self):
        pass


class SummaryServiceProvider(ServiceProvider):
    def register(self):
        from torch.utils.tensorboard import SummaryWriter
        self.app.singleton('summary', SummaryWriter(
            self.app.config('logs.summary.dir')
        ))

    def boot(self):
        pass


class DataServicesProvider(ServiceProvider):
    def register(self):
        from Source import Source

        self.app.singleton(
            Source,
            self.app.config('data.source')(
                **self.app.config('data.kwargs')
            )
        )

        self.app.set_alias(Source, 'source')

    def boot(self):
        pass


class NetworkServiceProvider(ServiceProvider):
    def register(self):
        from Modules import Network
        from helper import load_json
        self.app.singleton(Network, Network(
            self.app,
            load_json(
                self.app.config('training.define')
            )
        ))
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
