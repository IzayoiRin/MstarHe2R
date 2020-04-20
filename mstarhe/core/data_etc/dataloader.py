import numpy as np

from torch.utils.data import DataLoader


class GenericMstarDataLoader(object):

    mstar = None

    def __init__(self, train, **initkwargs):
        self.train = train
        self.initkwargs = initkwargs

    def get_mstar_dataset(self):
        return self.mstar

    def split_dataset(self, size, flush):
        """
        split mstar datset to training and validating set
        :return list[training_set, validating set]
        """
        raise NotImplementedError

    def __call__(self, batch_size, shuffle=True, split=False, size=1.0, flush=False):
        """
        train 0 -----------> (dataset(0))
            split 1 train 0 ---> (dataset(0))
            split 0 train 0 ---> (dataset(0))
        train 1
            split 0 train 1 ---> (dataset(1))
            split 1 train 1 ---> (dataset(1), dataset(-1))
        :param batch_size: loader batch size
        :param shuffle: whether shuffle during iter
        :param split: whether split to training and validating set
        :param size: split size for training set
        :param flush: whether flush data
        :return: torch.DataLoader
        """
        mstar = self.get_mstar_dataset()
        if not self.train:
            return DataLoader(
                dataset=mstar(train=False, flush=flush),
                batch_size=batch_size,
                shuffle=shuffle
            )
        if split:
            return [
                DataLoader(dataset=d, batch_size=batch_size, shuffle=shuffle)
                for d in self.split_dataset(flush=flush, size=size)
            ]
        return DataLoader(
                dataset=mstar(train=True, flush=flush),
                batch_size=batch_size,
                shuffle=shuffle
            )


class MstarTextMixin(object):

    def split_dataset(self, size, flush):
        """split mstar datset to training and validating set"""
        mstar = self.get_mstar_dataset()(train=self.train, flush=flush)
        indices = mstar.data.indices
        end = len(mstar) - 1
        slices = list()
        for idx, i in enumerate(indices):
            if idx < len(indices) - 1:
                slices.append((i, indices[idx+1]))
            else:
                slices.append((i, end))

        for s, e in slices:
            mc_indices = np.array(range(s, e))
            diff = mc_indices.shape[-1]
            training_indices = np.random.choice(mc_indices, size=int(np.ceil(size * diff)), replace=False)
            self.from_indices(training_indices, mc_indices)

        return [mstar[i] for i in self.tr_indices_], \
               [mstar[i] for i in self.vl_indices_]

    def from_indices(self, training, total):
        validating = np.setdiff1d(total, training)
        if validating.shape[-1] == 0:
            validating = [training[0], ]
        self.tr_indices_.extend(training)
        self.vl_indices_.extend(validating)


class MstarTextDataLoader(MstarTextMixin, GenericMstarDataLoader):

    def __init__(self, train, **initkwargs):
        super(MstarTextDataLoader, self).__init__(train=train, **initkwargs)
        self.tr_indices_, self.vl_indices_ = list(), list()


def _exmaple():
    from mstarhe.conf import LazySettings
    from mstarhe.core.data_etc.dataset import _example as _e

    setting = LazySettings()
    m, mc = _e()

    MstarTextDataLoader.mstar = mc()

    mdlc = MstarTextDataLoader(train=True)
    mdlt, mdlv = mdlc(batch_size=128, split=True, size=0.8)
    for bd, bl in mdlt:
        print(len(bd), len(bl))


if __name__ == '__main__':
    _exmaple()
