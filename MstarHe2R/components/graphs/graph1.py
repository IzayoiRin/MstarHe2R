from mstarhe.core.nn.graphs.fc import NLayerFeedForwardNet


class MSTARANNetGraph(NLayerFeedForwardNet):

    # hyper-parameters
    mask = 0.5      # dropout mask vector
    momentum = 0.5  # batch normalize momentum

    # This is a abstract class
    class Meta:
        abstract = True

    def forward(self, X, dim=-1):
        X = X.view(-1, 1*171*171)
        return super().forward(X, dim=-1)


class L4MSTARANNetGraph(MSTARANNetGraph):

    N_HIDDEN = 4
    HIDDEN_CELL = [1024, 512, 256, 15]


class L4MSTARANNetGraphV2(MSTARANNetGraph):

    N_HIDDEN = 4
    HIDDEN_CELL = [2048, 512, 256, 12]
    # HIDDEN_CELL = [4096, 512, 256, 12]


class L2MSTARANNetGraph(MSTARANNetGraph):

    N_HIDDEN = 2
    HIDDEN_CELL = [1024, 16]


class L5MSTARANNetGraph(MSTARANNetGraph):

    N_HIDDEN = 5
    HIDDEN_CELL = [1024, 512, 256, 128, 15]
