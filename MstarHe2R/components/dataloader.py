from mstarhe.conf import LazySettings
from mstarhe.core.data_etc.dataloader import MstarTextDataLoader
from mstarhe.errors import ConfigureError

from apps.mstarDataInitiation import __is_initial__

__is_initial__()
setting = LazySettings()

from components.dataset import Mstar2RDataSet


class Mstar2RDataLoader(MstarTextDataLoader):

    mstar = Mstar2RDataSet()

    def __call__(self, batch_size, shuffle=None, split=None, size=None, flush=None):
        try:
            split = split or bool(setting.VALIDATE_RATE)
            size = size or (1 - setting.VALIDATE_RATE)
        except AttributeError as e:
            raise ConfigureError(e)
        if size < 0:
            raise ConfigureError("VALIDATE_RATE must be Positive Float")
        shuffle = shuffle or getattr(setting, "LOADER_SHUFFLE", True)
        flush = flush or getattr(setting, "FLUSH_MSTAR", False)
        return super(Mstar2RDataLoader, self).__call__(batch_size, shuffle, split, size, flush)


if __name__ == '__main__':
    m2r = Mstar2RDataLoader(train=True)
    ld = m2r(batch_size=64)
    print(ld)
    for i, j in ld[0]:
        print(i.size(), j.size())
