import sys

# Project Setting Model
__DEFAULT_SETTING_MODEL__ = "dev"

if __name__ == '__main__':
    try:
        import mstarhe as mh
        mh.SETTING_MODEL = __DEFAULT_SETTING_MODEL__
        mh.setup()
    except ImportError:
        raise ImportError("Could not import MstarHe, make sure <mstarhe> under %s " % mh.PRJ_DIR)
    settings = mh.conf.LazySettings()
    settings.configure()
    sys.path.insert(0, settings.BASE_DIR)
    if len(sys.argv) > 1 and sys.argv[1] in settings.COMMANDS:
        mh.project_information()
        mh.fire_resoluter(settings.COMMANDS)
    else:
        print("Look at readme.txt for Further Helping")
