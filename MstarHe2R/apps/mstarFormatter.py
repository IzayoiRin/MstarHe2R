from mstarhe.conf import LazySettings
from mstarhe.utils.transformat.mstar_trans import MstarTransFormatter

setting = LazySettings()


def main():
    MstarTransFormatter.DATA_DIR = setting.RAW_DATA_DIR
    MstarTransFormatter.SAVE_DIR = setting.TRANS_OUT_DIR
    mstar = MstarTransFormatter(setting.DATA_DIR)
    mstar.to(*setting.TRANS2FORMATS, **setting.TRANS2PARAMS)


if __name__ == '__main__':
    main()
