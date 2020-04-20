"""Create Custom Computation Graph, Only named with NetGraph can be load to Models"""
from mstarhe.core.nn.graphs.fc import NLayerFeedForwardNet


class MSTARANNetGraph(NLayerFeedForwardNet):

    # hyper-parameters
    mask = 0.5      # dropout mask vector
    momentum = 0.5  # batch normalize momentum

    # This is a abstract class, will not load to Models
    class Meta:
        abstract = True

    def forward(self, X, dim=-1):
        X = X.view(-1, 1*128*128)
        return super().forward(X, dim=-1)


class TestL4MSTARANNetGraph(MSTARANNetGraph):

    N_HIDDEN = 4
    HIDDEN_CELL = [1024, 512, 256, 15]
