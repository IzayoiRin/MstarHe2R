import torch.nn as nn

from mstarhe.errors import ConfigureError, ObjectTypeError


class MSTARFCMetaClass(type):

    def __new__(cls, name, bases, attrs):
        meta = attrs.get("Meta", None)
        if meta is None:
            attrs["Meta"] = type("Meta", (object,), {"abstract": False})
        class_ = type.__new__(cls, name, bases, attrs)
        return class_


class NLayerFeedForwardNet(nn.Module, metaclass=MSTARFCMetaClass):

    activator = nn.ReLU
    out_activator = nn.LogSoftmax

    N_HIDDEN = 5
    HIDDEN_CELL = [100, 80, 50, 25, 15]
    B_INIT = 0.0

    mask = 0.5
    momentum = 0.5

    class Meta:
        abstract = False

    def __init__(self, dropout=False, batch_nor=False):
        super().__init__()

        self.dropout = dropout
        self.batch_nor = batch_nor

        if dropout and batch_nor:
            self.dropout = self.batch_nor = False

        self.graph = nn.Sequential()
        self._ginit = False

    def _set_init(self, layer):
        nn.init.normal_(layer.weight, mean=0., std=.1)
        nn.init.constant_(layer.bias, self.B_INIT)

    def flush(self):
        self.graph = nn.Sequential()
        self._ginit = False
        return self

    def assemble(self, ifea, ofea):
        if self._ginit:
            print('Graph assembled, flush() first')
            return
        if isinstance(self.HIDDEN_CELL, int):
            hidden_cells = [self.HIDDEN_CELL for _ in range(self.N_HIDDEN)]
        elif isinstance(self.HIDDEN_CELL, list) and len(self.HIDDEN_CELL) == self.N_HIDDEN:
            hidden_cells = self.HIDDEN_CELL.copy()
        else:
            raise ObjectTypeError("HIDDEN_CELL must be an int or "
                                  "list object which length equal to N_HIDDEN(%s)" % self.N_HIDDEN)

        hidden_cells.insert(0, ifea)

        for i in range(self.N_HIDDEN):
            # batch normalize layer
            if self.batch_nor:
                self.graph.add_module('bn%d' % i, nn.BatchNorm1d(hidden_cells[i], momentum=self.momentum))
            # ful combo layer
            fc = nn.Linear(hidden_cells[i], hidden_cells[i+1])
            self._set_init(fc)
            self.graph.add_module('fc%d' % i, fc)
            # activate layer
            self.graph.add_module('act%d' % i, self.activator())
            # dropout layer
            if self.dropout:
                self.graph.add_module('dp%d' % i, nn.Dropout(self.mask))

        fcout = nn.Linear(hidden_cells[-1], ofea)
        self._set_init(fcout)
        self.graph.add_module('fcout', fcout)

        self._ginit = True
        return self

    def forward(self, X, **kwargs):
        if not self._ginit:
            raise ConfigureError('assemble computational graph first')
        return self.out_activator(**kwargs)(self.graph(X))
