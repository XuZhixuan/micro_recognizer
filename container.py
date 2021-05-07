from typing import Union, Any
from warnings import warn


class Container:
    """
    Base container class
    store & resolve instance reference to class
    """
    _bindings = {}
    _alias = {}

    def singleton(self, abstract: Union[type, str], concrete: object):
        """
        Set a instance reference to a class
        Args:
            abstract: The class to reference
            concrete: The instance to bind
        """
        self._bindings[abstract] = concrete

    def resolve(self, abstract: Union[type, str]) -> Any:
        """
        Resolve the instance bind to class
        Args:
            abstract: The class that instance bind to
        Returns:
            concrete: The instance found
        """
        if abstract in self._bindings.keys():
            return self._bindings[abstract]
        elif abstract in self._alias:
            return self._bindings[
                self._alias[abstract]
            ]

    def set_alias(self, abstract: Union[type, str, dict], alias: Union[type, str] = None):
        """
        Set alias for exists binds
        Args:
            abstract: The original class reference or name or the dict of original name to alias
            alias: The alias name of it
        """
        if isinstance(abstract, dict):
            for ab, alias in abstract.items():
                if abstract in self._bindings.keys():
                    self._alias[alias] = abstract
                else:
                    warn('Reference to abstract %s not found in registered bindings' % abstract)
                    continue
            return

        if abstract in self._bindings.keys():
            self._alias[alias] = abstract
        else:
            warn('Reference to abstract %s not found in registered bindings' % abstract)

    def __getattr__(self, item):
        return self.resolve(item)
