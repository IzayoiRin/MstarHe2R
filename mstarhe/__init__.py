import os
import sys
from importlib import import_module


"""Base Project Information"""
SETTING_MODEL = "dev"  # Project Setting Model
ENTRY_FUNC = "main"  # Project Uni Entry Function Name

PRJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project Root Path
PRJ_NAME = os.path.split(PRJ_DIR)[-1]  # Project Name

# Create Core Application Dir
if not os.path.exists(os.path.join(PRJ_DIR, PRJ_NAME)):
    os.makedirs(os.path.join(PRJ_DIR, PRJ_NAME))


def project_information():
    """Print project information"""
    msg = "***Welcome to MstarHe Project***\n" \
          "$Core Application: {name}\n" \
          "$Environment: {model}\n" \
          "\t\t@Copyright Iz@yoiRin".format(name=PRJ_NAME, model=SETTING_MODEL.upper())
    print(msg)


def recursion_import(module, entry=None, callback=True):
    """
    recursion import entry function from module, <module>.<entry_func>
    :param module: target module name
    :param entry: target entry function name, default "main"
    :param callback: recursion searching flag
    :return: target entry function
    """
    try:
        # import from module
        mod = import_module(module)
    # dose not a module
    except ModuleNotFoundError:
        # should recursion search
        if callback:
            # target module split to module and calling object <module>.<calling>
            module, calling = module.rsplit('.', 1)
            # recursion import calling from module and will stop at this time
            return recursion_import(module, entry=calling, callback=False)
        # search at recursion end, still can't find legal module
        else:
            raise ImportError("Could not import <%s>" % module)
    # success
    else:
        try:
            # get entry function
            entry_func = getattr(mod, entry or ENTRY_FUNC)
        # dose not have legal entry function
        except AttributeError:
            raise ImportError("Could not import <%s> from <%s>" % (entry, module))
        # success
        else:
            return entry_func


def fire_resoluter(cmapping):
    # receive shell args
    cmd_args = sys.argv
    # get execution file name
    startf = cmd_args[0]
    # get command name
    cmd = cmapping[cmd_args[1]]
    try:
        # get entry function
        entry = recursion_import(cmd)
    except ImportError as e:
        print(e)
        return

    import fire
    # mapping fire shell
    fire.Fire({cmd_args[1]: entry})


def setup():
    """
    Configure the settings (this happens as a side effect of accessing the
    first setting), configure logging and populate the app registry.
    Set the thread-local urlresolvers script prefix if `set_prefix` is True.
    """
    from mstarhe import conf
    from mstarhe.utils.logs import configure_logging

    os.environ.setdefault("MSTARHE_SETTING_MODULE", '.'.join([PRJ_NAME, "settings", SETTING_MODEL]))

    # configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
    # if set_prefix:
    #     set_script_prefix(
    #         '/' if settings.FORCE_SCRIPT_NAME is None else force_text(settings.FORCE_SCRIPT_NAME)
    #     )
    # apps.populate(settings.INSTALLED_APPS)


def upload_mstar_env():
    import os
    if os.environ.get("MSTARHE_SETTING_MODULE", None) is None:
        setup()
