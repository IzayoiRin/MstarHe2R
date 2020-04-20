from torch import nn


def min_max_normalize(tensor):
    a, b = tensor.min(), tensor.max()
    return (tensor - a) / (b - a)


class ImgMinMaxNormalize(nn.Module):

    def __init__(self, std_operator=255):
        super(ImgMinMaxNormalize, self).__init__()
        self.std_operator = float(std_operator)

    def __call__(self, tensor):
        return min_max_normalize(tensor) / self.std_operator

    def __repr__(self):
        return "{name}(operator={op})".format(
            name=self.__class__.__name__,
            op=self.std_operator
        )
