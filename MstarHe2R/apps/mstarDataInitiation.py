import os

from mstarhe.conf import LazySettings
from mstarhe.errors import DataInitialError


setting = LazySettings()


def __is_initial__():
    if os.environ.get("MSTARHE_SETTING_MODULE", None) is None:
        import mstarhe as mh
        mh.setup()
    cached = os.path.join(setting.DATA_DIR, setting.MSTAR_SAVE_DIR)
    if os.path.exists(cached) and os.listdir(cached):
        return os.listdir(cached)
    else:
        raise DataInitialError("Mstar Initial Fail")


def main(mod="w"):
    from components.dataset import Mstar2RDataSet
    MSTAR = Mstar2RDataSet
    if mod == "w":
        m1 = MSTAR()(train=True, flush=True)
        m0 = MSTAR()(train=False, flush=True)
        print(m1)
        print(m0)
    elif mod == "train":
        print(MSTAR()(train=True, flush=True))
    elif mod == "test":
        print(MSTAR()(train=False, flush=True))
    else:
        print("Wrong parameters, see help")
