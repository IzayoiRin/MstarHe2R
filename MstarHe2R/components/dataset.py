from mstarhe.core.data_etc.dataset import MstarTextDataSet
from mstarhe.conf import LazySettings


setting = LazySettings()


class Mstar2RDataSet(MstarTextDataSet):

    DATA_DIR = setting.TENSOR_DATA_DIR
    SAVE_DIR = setting.MSTAR_SAVE_DIR

    prefixed = setting.MSTAR_PREFIXED_DIR

    TARGETS = setting.MSTAR_DATA_TARGETS

    def __init__(self, root_dir=None, img_format=None):
        root_dir = root_dir or setting.DATA_DIR
        img_format = img_format or getattr(setting, "IMG_STD_TENSOR_FORMAT", None)
        super(Mstar2RDataSet, self).__init__(root_dir, img_format=img_format)
        if hasattr(setting, "MSTAR_DATA_MODE_DIR"):
            self.setting_mode = setting.MSTAR_DATA_MODE_DIR
