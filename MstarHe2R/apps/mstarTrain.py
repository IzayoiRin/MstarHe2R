from mstarhe.conf import LazySettings
from mstarhe.core.nn.graphs.loader import loader_graph_from_module
from components.models import MSTARNet


settings = LazySettings()
G = loader_graph_from_module(settings.COMPUTATION_GRAPHS)


def main(d=None):
    Net = MSTARNet
    # set training input img size H * W
    if getattr(settings, "IMG_SIZE", None):
        from components import models
        models.__IMG_SIZE__ = settings.IMG_SIZE
    # set device for training
    if not getattr(settings, "CUDA_DEVICE_AVAILABLE", None):
        Net.device = None
    elif d in ("c", 'g') and d == 'c':
        Net.device = None
    # set data loader params
    Net.loader_params = settings.MODEL_LOADER_PARAMS
    # training on all loaded computation graphs
    for g, params in G:
        Net.model_graph_class = g
        Net.alpha = params["aph"]
        Net.step = params["stp"]
        net = Net(3, reg=None, dropout=False)
        print(net.graph.__class__.__name__)
        net.train(params['n'], 'PQ', checkpoint=params['cp'])
