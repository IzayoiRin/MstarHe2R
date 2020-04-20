import logging
import re
import torch as th
from torch import nn

from mstarhe.errors import ConfigureError, ObjectTypeError


logger = logging.getLogger('mstarhe')

CONV_DIM = 2
CONV_BACTH_NOR = False


def __conv_setup__(conv_dim=None, conv_bn=None):
    mapping = {
        1: (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d, nn.AvgPool1d),
        2: (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d),
        3: (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, nn.AvgPool3d)
    }

    conv_dim = conv_dim or CONV_DIM
    if conv_dim in mapping.keys():
        global Conv, BatchNorm, MaxPool, AvgPool
        Conv, BatchNorm, MaxPool, AvgPool = mapping[conv_dim]

        global CONV_BACTH_NOR
        CONV_BACTH_NOR = CONV_BACTH_NOR if conv_bn is None else conv_bn
    else:
        raise ConfigureError("set legal conv dim <{}>".format(list(mapping.keys())))


__conv_setup__()


class Fallen(nn.Module):

    def __init__(self):
        super(Fallen, self).__init__()

    def forward(self, X):
        size = X.size()
        if len(size) > 1:
            fallen = 1
            for s in size[1:]:
                fallen *= s
            return X.view(-1, fallen)
        else:
            return X


class ConvRelu(nn.Module):

    def __init__(self, in_chanel, out_chanel, kernel, stride=1, padding=0, bacth_nor=None, eps=1e-3):
        super(ConvRelu, self).__init__()
        self.convr = nn.Sequential()
        self.convr.add_module('conv_cr', Conv(in_chanel, out_chanel, kernel, stride, padding))
        bacth_nor = bacth_nor or CONV_BACTH_NOR
        if bacth_nor:
            self.convr.add_module('bn_cr', BatchNorm(out_chanel, eps=eps))
        self.convr.add_module('act_cr', nn.ReLU(True))

    def forward(self, X):
        return self.convr(X)


class InceptionBaseRoute(nn.Module):
    """
    Route = [
        ('conv', {'kernel': 1, 'padding': 0}),
        ('conv', {'kernel': 3, 'padding': 0}),
        ]
    """
    Route = list()
    Chanel = list()

    FMapping = {
        'conv': ConvRelu,
        'maxp': MaxPool,
        'avgp': AvgPool,
        'fln': Fallen
    }

    NON_WEIGHTS = ['maxp', 'avgp', 'fln']

    def __init__(self):
        super(InceptionBaseRoute, self).__init__()
        self.graph = nn.Sequential()
        self._ginit = False
        self._placeholder = None

    def flush(self):
        self.graph = nn.Sequential()
        self._ginit = False
        return self

    def assemble(self, in_chanel):
        if self._ginit:
            print('Graph assembled, flush() first')
            return
        n = len(self.Route)
        if isinstance(self.Chanel, int):
            dims = [self.Chanel for _ in range(n)]
        elif isinstance(self.Chanel, list) and len(self.Chanel) == n:
            dims = self.Chanel.copy()
        else:
            raise ObjectTypeError("DIMENSION must be an int or "
                                  "list object which length equal to Route(%s)" % n)
        dims.insert(0, in_chanel)

        for i in range(n):
            func_n, p = self.Route[i]
            if func_n not in self.FMapping.keys():
                raise ConfigureError("set legal function name")
            func = self.FMapping[func_n]

            if func_n in self.NON_WEIGHTS and dims[i+1] == self._placeholder:
                dims[i+1] = dims[i]
                m = func(**p)
            elif self.cond_(func_n, dims):
                m = self.costume_assemble(func, dims, i, **p)
            else:
                m = func(dims[i], dims[i + 1], **p)
            self.graph.add_module("slayer%s" % i, m)
        self._ginit = True
        return self

    def cond_(self, func_name, dims):
        return False

    def costume_assemble(self, func, dims, i, **p):
        raise NotImplementedError

    def forward(self, X):
        if not self._ginit:
            raise ConfigureError('assemble computational graph first')
        return self.graph(X)


class IncRoute15(InceptionBaseRoute):

    Route = [
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': 5, 'padding': 2})
    ]


class IncRoute13(InceptionBaseRoute):

    Route = [
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': 3, 'padding': 1})
    ]


class IncRoute11(InceptionBaseRoute):

    Route = [
        ('conv', {'kernel': 1}),
        ]


class IncRouteMP1(InceptionBaseRoute):

    Route = [
        ('maxp', {'kernel_size': 3, 'stride': 1, 'padding': 1}),
        ('conv', {'kernel': 1})
    ]


class IncRoute133(InceptionBaseRoute):

    Route = [
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': 3}),
        ('conv', {'kernel': 3}),
    ]


class IncRoute1NN(InceptionBaseRoute):

    N = 1
    Route = [
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': (1, N)}),
        ('conv', {'kernel': (N, 1)}),
    ]


class IncRoute1NN2(InceptionBaseRoute):

    N = 1
    Route = [
        ('conv', {'kernel': 1}),
        ('conv', {'kernel': (1, N)}),
        ('conv', {'kernel': (N, 1)}),
        ('conv', {'kernel': (1, N)}),
        ('conv', {'kernel': (N, 1)}),
    ]


class Inception(nn.Module):

    Routes = list()
    chanelArr = list()

    def __init__(self):
        super(Inception, self).__init__()
        self.routes_ = list()
        self._ginit = False

    def flush(self):
        self.routes_ = list()
        self._ginit = False
        return self

    def assemble(self, in_chanel):
        if self._ginit:
            print('Graph assembled, flush() first')
            return
        if len(self.Routes) != len(self.chanelArr):
            raise ConfigureError("Chanels' length(%s) equal to Routes'(%s)"
                              % (len(self.chanelArr), len(self.Routes)))
        i = 0
        for route, chanel in zip(self.Routes, self.chanelArr):
            route.Chanel = chanel
            setattr(self, 'rt%s' % i, route().assemble(in_chanel))
            self.routes_.append(getattr(self, 'rt%s' % i))
            i += 1
        return self

    def forward(self, X):
        out = [r(X) for r in self.routes_]
        return th.cat(out, dim=1)


class MSTARMetaClass(type):
    pattern = r'Inception(\d+[a-z]*)'
    IncMapping = dict()

    def __new__(cls, name, bases, attrs):
        ret = re.search(cls.pattern, name)
        class_ = type.__new__(cls, name, bases, attrs)
        if ret:
            key = 'inc%s' % ret.group(1)
            cls.IncMapping[key] = class_
            logger.debug('load %s to IncMapping, call %s' % (name, key))
        return class_


class V1Inception(Inception):
    Routes = [IncRoute11, IncRoute13, IncRoute15, IncRouteMP1]
    # conv1, conv1-3, conv1-5, convM-1
    chanelArr = [[64], [16, 32], [96, 128], [None, 32]]


class V2Inception(Inception):

    N = 3
    IncRoute1NN.N = N
    IncRoute1NN2.N = N

    Routes = [IncRoute11, IncRoute1NN, IncRoute1NN2, IncRouteMP1]
    # conv1, conv1-n-n, conv1-n-n-n-n, convM-1
    chanelArr = [[64], [16, 16, 32], [96, 96, 96, 96, 128], [None, 32]]


if __name__ == '__main__':
    a = th.randn(8, 1, 171, 171)
    v = V2Inception().assemble(1)
    b = v(a)
    print(b.size())
    # p = nn.MaxPool1d(2, stride=2)
    # c = p(b)
    # print(c.size())
    # e = MaxPool(3, stride=2, padding=0)(e)
    # print(e.size())


