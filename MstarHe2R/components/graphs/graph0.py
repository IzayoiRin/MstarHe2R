"""Create Custom Computation Graph, Only named with NetGraph can be load to Models"""
import logging
import copy

import torch as th
import torch.nn as nn

from mstarhe.core.nn.graphs.fc import NLayerFeedForwardNet
from mstarhe.core.nn.graphs.cnn import V1Inception, InceptionBaseRoute, MSTARMetaClass

logger = logging.getLogger('mstarhe')

IncMapping = dict()
MSTARMetaClass.IncMapping = IncMapping


class MSTARaseV1Inception(V1Inception, metaclass=MSTARMetaClass):
    __constants__ = ['inchanel', ]

    def __init__(self, inchanel):
        self.inchanel = inchanel
        super(MSTARaseV1Inception, self).__init__()
        self.assemble(inchanel)

    def extra_repr(self):
        return 'inchanel={}'.format(self.inchanel)


class MSTARV1Inception3a(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[32], [48, 64], [8, 16], [None, 16]]


class MSTARV1Inception3b(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[64], [64, 96], [16, 48], [None, 32]]


class MSTARV1Inception4a(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[96], [48, 104], [8, 24], [None, 32]]


class MSTARV1Inception4b(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[80], [56, 112], [12, 32], [None, 32]]


class MSTARV1Inception4c(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[64], [64, 128], [12, 32], [None, 32]]


class MSTARV1Inception4d(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[56], [72, 144], [16, 32], [None, 32]]


class MSTARV1Inception4e(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[128], [80, 160], [16, 64], [None, 64]]


class MSTARV1Inception5a(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[128], [80, 160], [16, 64], [None, 64]]


class MSTARV1Inception5b(MSTARaseV1Inception):
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[192], [96, 192], [24, 64], [None, 64]]


class MSTARConvRoute(InceptionBaseRoute):
    """
        a = MSTARConvRoute().assemble(1)
        print(a.graph)
        e = th.randn(8, 1, 171, 171)
        print("input: batch %s, C%s, W%s, H%s" % e.size())
        e = a(e)
        print("output: batch %s, C%s, W%s, H%s" % e.size())
    """

    Route = [
        ('conv', {'kernel': 7, 'stride': 2}),
        ('maxp', {'kernel_size': 3, 'stride': 2}),
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': 3, 'padding': 1}),
        ('maxp', {'kernel_size': 3, 'stride': 2}),
        ('inc3a', dict()),
        ('inc3b', dict()),
        ('maxp', {'kernel_size': 3, 'stride': 2, 'padding': 1}),
        ('inc4a', dict()),
        ('inc4b', dict()),
        ('inc4c', dict()),
        ('inc4d', dict()),
        ('inc4e', dict()),
        ('maxp', {'kernel_size': 3, 'stride': 2, 'padding': 1}),
        ('inc5a', dict()),
        ('inc5b', dict()),
        ('avgp', {'kernel_size': 5}),
        ('fln', dict())
    ]

    Chanel = [32, None, 64, 96, None, 128, 240, None, 256, 256, 256, 264, 416, None, 416, 512, None, None]

    def __init__(self, dropout=False):
        self.FMapping = copy.deepcopy(self.FMapping)
        for k, v in IncMapping.items():
            self.FMapping[k] = v
        if dropout:
            self.FMapping['dp'] = nn.Dropout2d
            self.NON_WEIGHTS.append('dp')
        super(MSTARConvRoute, self).__init__()

    def cond_(self, func_name: str, dims):
        if func_name.startswith('inc'):
            return True
        else:
            return False

    def costume_assemble(self, func, dims, i, **p):
        return func(dims[i], **p)


class MSTARCNNetGraph(NLayerFeedForwardNet):

    inc = MSTARConvRoute(dropout=False)

    N_HIDDEN = 2
    # N_HIDDEN = 3
    HIDDEN_CELL = [256, 32]
    # HIDDEN_CELL = [4096, 256, 12]

    __inchanel__ = 1
    __ifea__ = 171

    def add_inception(self):
        if self._ginit:
            print('Graph assembled, flush() first')
            return
        self.graph.add_module('v1inc', self.inc.assemble(self.__inchanel__))

    def assemble(self, ifea=521, ofea=4):
        self.add_inception()
        super(MSTARCNNetGraph, self).assemble(512, ofea=ofea)
        return self

    def forward(self, X, dim=-1):
        flag = isinstance(X, th.Tensor) and len(X.size()) is 4 \
               and X.size(1) == self.__inchanel__ and X.size(-1) == X.size(-2) == self.__ifea__
        if not flag:
            raise ValueError('X must be shape like: [B, C, W, H] == [N, %s, %s, %s]'
                             % (self.__inchanel__, self.__ifea__, self.__ifea__))
        return super().forward(X, dim=-1)


def _example():
    g = MSTARCNNetGraph(dropout=True).assemble()
    print(g.graph)
    e = th.randn(8, 1, 171, 171)
    print(e.device)
    print("input: batch %s, C%s, W%s, H%s" % e.size())
    e = g(e)
    print("output: batch %s, C * W * H = %s" % e.size())


if __name__ == '__main__':
    _example()
