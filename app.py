from Services import *
from container import Container
from handle import Handler
from helper import check_dir


class Application(Container):
    """
    Base app core class
    """

    def __init__(self):
        """
        Init the application instance
        """
        self.register_base_bindings()
        self.register_base_services()

        self.register_user_services()
        self.singleton('handler', Handler(self))

    def register_base_bindings(self):
        """
        Set the basic bindings of useful components
        """
        from config import config

        self.singleton('app', self)
        self.singleton(Container, self)
        self.singleton('config', config)

    def register_base_services(self):
        """
        Set the auxiliary services for training
        """
        self.register(HelperFunctionsServiceProvider(self))
        self.register(LayerMakeServiceProvider(self))
        self.register(SummaryServiceProvider(self))

    def register_user_services(self):
        """
        Set the main services for training
        """
        for service in self._providers:
            self.register(service(self))

        for service in self._providers:
            self.boot(service(self))

    _providers = [
        ImageLoaderServiceProvider,
        DataServicesProvider,
        NetworkServiceProvider,
        TrainingServiceProvider,
        RedirectPrintServiceProvider
    ]

    @staticmethod
    def register(service: ServiceProvider):
        """
        Call the register method of service
        Args:
            service: The service to register to container
        """
        service.register()

    @staticmethod
    def boot(service: ServiceProvider):
        """
        Call the boot method of service
        Args:
            service: The service to boot
        """
        service.boot()

    def handle(self):
        self.handler.run()


def run():
    check_dir()
    app = Application()
    app.handle()


if __name__ == '__main__':
    run()
