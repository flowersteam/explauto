class ExplautoError(Exception):
    pass


class ExplautoEnvironmentUpdateError(ExplautoError):
    pass


class ExplautoBootstrapError(ExplautoError):
    pass


class ExplautoNoTestCasesError(ExplautoError):
    pass
