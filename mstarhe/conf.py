import os
import importlib

ENVIRONMENT_VARIABLE = "MSTARHE_SETTING_MODULE"
empty = object()


class Settings(object):

    def __init__(self, settings_module):
        self.SETTINGS_MODULE = settings_module
        mod = importlib.import_module(self.SETTINGS_MODULE)
        for setting in dir(mod):
            if setting.isupper():
                setattr(self, setting, getattr(mod, setting))


class LazySettings(object):

    _wrapped = None

    def __init__(self):
        self._wrapped = empty

    def _setup(self):
        settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
        self._wrapped = Settings(settings_module)

    def __repr__(self):
        if self._wrapped is empty:
            return '<LazySettings [Unevaluated]>'
        return '<LazySettings "%(settings_module)s">' % {
            'settings_module': self._wrapped.SETTINGS_MODULE,
        }

    def __getattr__(self, name):
        """
        Return the value of a setting and cache it in self.__dict__.
        """
        if self._wrapped is empty:
            self._setup()
        val = getattr(self._wrapped, name)
        self.__dict__[name] = val
        return val

    def __setattr__(self, key, value):
        if key == '_wrapped':
            self.__dict__.clear()
            self.__dict__[key] = value
        else:
            self.__dict__.pop(key, None)
            if self._wrapped is empty:
                self._setup()
            setattr(self._wrapped, key, value)

    def __delattr__(self, name):
        if name == '_wrapped':
            raise AttributeError("can't delete _wrapped.")
        if self._wrapped is empty:
            self._setup()
        delattr(self._wrapped, name)
        self.__dict__.pop(name, None)

    def configure(self):
        self._setup()

    @property
    def configured(self):
        return self._wrapped is not empty


setting = LazySettings()
