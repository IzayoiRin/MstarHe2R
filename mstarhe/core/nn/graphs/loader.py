import re
from importlib import import_module

from mstarhe.conf import LazySettings


settings = LazySettings()


def loader_graph_from_module(module):
    graph_mod_pattern = re.compile(r"graph\d+")
    graph_pattern = re.compile(r'NetGraph')
    mod_ = import_module(module)
    graph_mods = [m for attr, m in mod_.__dict__.items() if re.match(graph_mod_pattern, attr)]
    graphs = list()

    def loader(mod):
        pakage = mod.__name__.rsplit(".", 1)[-1]
        params = settings.HYPER_PARAMETERS.get(pakage, dict())
        for attr, g in mod.__dict__.items():
            if re.search(graph_pattern, attr) and getattr(g.Meta, "abstract", False) is False:
                graphs.append([g, params.get(attr, settings.HYPER_PARAMETERS["default"])])

    for graph_mod in graph_mods:
        loader(graph_mod)

    return graphs
