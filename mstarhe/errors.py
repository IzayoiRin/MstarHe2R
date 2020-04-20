class LoggingConfigError(Exception):
    pass


class ConfigureError(Exception):
    pass


class DataInitialError(Exception):
    pass


class ObjectTypeError(TypeError):
    pass


class AnalysisRuntimeError(RuntimeError):
    pass


class ParametersError(ValueError):
    pass
