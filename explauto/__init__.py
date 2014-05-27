import logging

from ._version import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())


class ExplautoError(Exception):
    pass


class ExplautoEnvironmentUpdateError(ExplautoError):
    pass


class ExplautoBootstrapError(ExplautoError):
    pass


class ExplautoNoTestCasesError(ExplautoError):
    pass
