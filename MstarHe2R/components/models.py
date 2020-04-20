import os

import numpy as np
import pandas as pd
import torch as th

from mstarhe.core.nn.models import PrettyFeedForward
from MstarHe2R.components.dataloader import Mstar2RDataLoader


__IMG_SIZE__ = 128 * 128


class MSTARNet(PrettyFeedForward):

    data_loader_class = Mstar2RDataLoader
    # model_graph_class = ANNetGraph
    model_graph_class = None
    optimizer_class = th.optim.Adam
    loss_func_class = th.nn.NLLLoss

    loader_params = {
        "train": {},
        "test": {}
    }

    # hyper-parameters
    lr = 1e-3  # learning rate
    l1_lambda = 0.5  # l1-penalty coef
    l2_lambda = 0.01  # l2-penalty coef
    step = 10  # measure_progress step k
    patient = 3  # early stopping patient
    alpha = 0.5  # early stopping threshold

    def __init__(self, ofea, **kwargs):
        super(MSTARNet, self).__init__(ifea=__IMG_SIZE__, ofea=ofea, **kwargs)
        self.CHECK_POINT = 'cp{}ep%s.tar'.format(self.model_graph_class.__name__)
        self._acc = list()
        self.acc_curve = list()
        self._loss = list()
        self.vloss_curve = list()
        self.tloss_curve = list()

        self.eval_ret = list()
        self.pre_accuracy = None

        self.test_samples_ = list()

    def get_data_loader(self, train):
        p = self.loader_params['train'] if train else self.loader_params['test']
        loader_factory = self.data_loader_class(train=train)
        if train:
            p["split"] = True
            return loader_factory(**p)
        p["shuffle"] = False
        loader = loader_factory(**p)
        self.test_samples_ = np.array(loader_factory.mstar.samples).reshape(-1, 1)
        return loader

    @property
    def epoch_acc(self):
        return np.mean(self._acc)

    @property
    def epoch_loss(self):
        return np.mean(self._loss)

    def analysis(self, label, ypre, preP):
        """
        :param label: size(batch) true class
        :param ypre: size(batch) pre class
        :param preP: size(batch) pre prob
        :return:
        """
        self._acc.append(self.accuracy(ypre, label).item())
        if not getattr(self, 'validate', False):
            self.eval_ret.append(th.stack([label.float(), ypre.float(), preP], dim=1))

    def train_batch(self, dl):
        super(MSTARNet, self).train_batch(dl)
        self.tloss_curve.append(self.epoch_loss)

    def eval_batch(self, dl):
        self._acc = list()
        # eval testing or validating batch
        super(MSTARNet, self).eval_batch(dl)
        print('Average Accuracy: %s' % self.epoch_acc)
        if getattr(self, 'validate', False):
            self.acc_curve.append(self.epoch_acc)
            self.vloss_curve.append(self.epoch_loss)
        else:
            ret = th.cat(self.eval_ret, dim=0)
            self.pre_accuracy = self.accuracy(ret[0], ret[1])
            path = os.path.join(self.csv_path, 'EvalCurves%s.txt' % self.model_graph_class.__name__)
            pd.DataFrame(np.hstack([self.test_samples_, ret.cpu().numpy()]),
                         columns=['objects', 'labels', 'predict', 'prob'])\
                .to_csv(path, sep='\t', index=True, header=True)

    def model_persistence(self):
        super(MSTARNet, self).model_persistence()
        curves = {
            "Accaracy": self.acc_curve,
            "TrLoss": self.tloss_curve,
            "VaLoss": self.vloss_curve
        }
        path = os.path.join(self.csv_path, 'EpochCurves%s.txt' % self.model_graph_class.__name__)
        df = pd.DataFrame(curves.values()).T
        df.columns = curves.keys()
        df.to_csv(path, sep='\t', index=True, header=True)


def _example():
    Net = MSTARNet
    Net.device = None
    from components.graphs.graph2 import TestL4MSTARANNetGraph
    G = [TestL4MSTARANNetGraph]
    for g, params in G:
        Net.model_graph_class = g
        Net.alpha = params["aph"]
        Net.step = params["stp"]
        net = Net(3, reg=None, dropout=False)
        print(net.graph.__class__.__name__)
        # print(net.get_data_loader(False))
        # print(len(net.test_samples_))
        net.train(params['n'], 'PQ', checkpoint=params['cp'])


if __name__ == '__main__':
    _example()
