import logging
from logging import handlers as H

from mstarhe.errors import LoggingConfigError


class Logger(object):

    def __init__(self, name: str, formatters: dict, handlers: dict):
        self.name = name
        self.logger = None
        self.formatters = formatters
        self.handlers = handlers

    def get_format(self, fmt, obj=True):
        try:
            fmt = self.formatters[fmt]["format"]
        except IndexError:
            raise LoggingConfigError("Could not config <%s> format" % fmt)
        return logging.Formatter(fmt) if obj else fmt

    def set_basic_configure(self, level, fmt='simple'):
        logging.basicConfig(level=level, format=self.get_format(fmt, obj=False))
        self.logger = logging.getLogger(self.name)

    def setup(self):
        for handler, kwargs in self.handlers.items():
            func = getattr(self, "handle_%s" % handler, None)
            if func is None:
                raise LoggingConfigError("Could not config <%s> logger" % handler)
            handler = func(**kwargs)
            self.logger.addHandler(handler)

    def handle_file(self, level, logclass, filename, fmt, **params):
        """

        :param fmt: logger.Formatter
        :param level: logger.INFO
        :param logclass: logging.handlers.RotatingFileHandler
        :param filename:
        :return:
        """
        fhandler_class = getattr(H, logclass, None)
        if fhandler_class is None:
            raise LoggingConfigError()
        fhandler = fhandler_class(filename, **params)  # type: H.RotatingFileHandler
        fhandler.setLevel(level)
        fhandler.setFormatter(self.get_format(fmt))
        return fhandler

    def handle_console(self):
        pass


def configure_logging(setting):
    pass
