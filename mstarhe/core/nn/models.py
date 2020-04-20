import copy as cp
import time
import os

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from mstarhe.conf import LazySettings
from mstarhe.errors import ConfigureError, ObjectTypeError, AnalysisRuntimeError, ParametersError


settings = LazySettings()
TORCH_PATH_ = settings.PERSISTENCE_DAT_DIR


class GenericFeedForwardNet(object):

    # config torch data loader
    data_loader_class = None
    # config torch computational graph
    model_graph_class = None
    # config torch optimizer
    optimizer_class = None
    # config loss function
    loss_func_class = None

    # the device that model is running on
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    # hyper-parameters
    lr = 1e-3       # learning rate
    l1_lambda = 0.5     # l1-penalty coef
    l2_lambda = 0.01    # l2-penalty coef

    # persistence config
    ROOT_PATH = TORCH_PATH_
    ROOT_DIR = 'data'
    PKL_DIR = "pickles"
    CSV_DIR = 'texts'
    TAR_DIR = 'tar'
    CHECK_POINT = 'cp%s.tar'

    def __init__(self, ifea, ofea, **kwargs):
        """
        :param kwargs: {
            reg: l-norm penalty, [L1, L2]
            dropout: dropout, [bool]
            batch_nor: batch_normalize, [bool]
        }
        """
        # init object running kwargs
        self.kwargs = kwargs

        # init regulation params
        if kwargs.get('reg'):
            self.penalty_mapping = {
                'L1': lambda v: self.l1_lambda * th.abs(v).sum(),
                'L2': lambda v: self.l2_lambda * th.sqrt(th.pow(v, 2).sum())
            }

        # init graph and optimizer
        self.graph = self.get_graph(ifea, ofea)
        self.optimizer = self.get_optimizer()

        self._loss = list()
        self.last_epoch_ = 0

        # init persistence path config
        self.path = os.path.join(self.ROOT_PATH, self.ROOT_DIR)
        self.pkl_path = os.path.join(self.path, self.PKL_DIR)
        self.csv_path = os.path.join(self.path, self.CSV_DIR)
        self.tar_path = os.path.join(self.path, self.TAR_DIR)
        self._init_dir(self.pkl_path, self.csv_path, self.tar_path)

    @staticmethod
    def _init_dir(*path):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)

    def get_data_loader(self, train):
        return self.data_loader_class(train=train)

    def get_graph(self, ifea, ofea):
        model_graph = self.model_graph_class(dropout=self.kwargs.get('dropout', False),
                                             batch_nor=self.kwargs.get('batch_nor', False))\
            .assemble(ifea, ofea)
        return model_graph.to(self.device) if self.device else model_graph

    def get_optimizer(self):
        if hasattr(self.graph, 'parameters'):
            return self.optimizer_class(self.graph.parameters(), lr=self.lr)
        raise ConfigureError('Config a legal model graph first')

    def get_loss(self, x, y, **kwargs):
        """
        reg: l1 or l2 penalty regularization
        :param x: logP
        :param y: label
        :return: J(theta) = cross-entropy + l-norm penalty
        """
        loss = self.loss_func_class(**kwargs)(x, y)
        reg = self.kwargs.get('reg')
        if reg and reg in self.penalty_mapping.keys():
            penalty = self.penalty_mapping[self.kwargs['reg']]
            loss += penalty(y.float())
        return loss

    def dat2device(self, x, y):
        return [x.to(self.device), y.to(self.device)] if self.device else [x, y]

    def ave_loss(self, loss_arr):
        return np.mean(loss_arr)

    def accuracy(self, x, y):
        return x.eq(y).float().mean()

    def analysis(self, label, ypre, preP):
        raise NotImplementedError

    def train(self, iter_n, checkpoint=0, **kwargs):
        """
        train model from training data loader in several epoch, each use whole batch to train
        :param iter_n: max epoch
        """
        for epoch in range(iter_n):
            self.kwargs['epoch'] = epoch
            tr_dl = self.get_data_loader(train=True)
            self.train_batch(tr_dl)
            if checkpoint and (epoch+1) % checkpoint == 0:
                self.checkpoint()

    def train_batch(self, dl):
        """
        train model from data loader in whole batch, each use one batch to update parameters
        :param dl: training data loader
        """
        self._loss = list()
        self.graph.train()
        for batch, (X, label) in enumerate(dl):
            self.kwargs['batch'] = batch
            X, label = self.dat2device(X, label)
            # initial optimizer
            self.optimizer.zero_grad()
            # forward
            logP = self.graph(X)
            # calculate loss J(theta)
            loss = self.get_loss(logP, label)
            # backward
            loss.backward()
            # update parameters
            self.optimizer.step()

            # average loss
            self._loss.append(loss.item())
            # if batch % 100 == 0:
            #     print("Batch: %s loss: %s" % (batch, self.ave_loss(loss_arr)))
            self.ave_loss(self._loss)

    def eval(self):
        """
        eval model from testing data loader in several epoch, each use whole batch
        """
        te_dl = self.get_data_loader(train=False)
        with th.no_grad():
            self.eval_batch(te_dl)

    def eval_batch(self, dl):
        """
        eval model from testing data loader in whole batch, each use one batch to analysis eval coef.
        :param dl: testing data loader
        """
        self._loss = list()
        self.graph.eval()
        for batch, (X, label) in enumerate(dl):
            self.kwargs['batch'] = batch
            X, label = self.dat2device(X, label)
            # forward
            logP = self.graph(X)
            # calculate loss J(theta)
            loss = self.get_loss(logP, label)

            # average loss
            self._loss.append(loss.item())
            self.ave_loss(self._loss)

            # get predict out, predict probability and label
            preP, ypre = logP.max(dim=-1)

            try:
                self.analysis(label, ypre, preP)
            except NotImplementedError:
                pass
            except Exception as e:
                raise AnalysisRuntimeError(e)

    def save(self, obj, name):
        if not hasattr(obj, 'state_dict'):
            raise ObjectTypeError("%s can't save" % obj)
        path = os.path.join(self.pkl_path, name)
        th.save(obj.state_dict(), path)

    def checkpoint(self):
        cp = {
            'epoch': self.kwargs['epoch'],
            'model_state_dict': self.graph.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        th.save(cp, os.path.join(self.tar_path, self.CHECK_POINT % cp['epoch']))

    def load(self, gpath=None, optpath=None, ckpath=None):
        if gpath:
            self.graph.load_state_dict(th.load(os.path.join(self.pkl_path, gpath)))
        if optpath:
            self.optimizer.load_state_dict(th.load(os.path.join(self.pkl_path, optpath)))
        if ckpath:
            checkpoint = th.load(os.path.join(self.tar_path, ckpath))
            self.graph.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch_ = checkpoint['epoch']


class EarlyStoppingMixin(object):

    step = 10
    alpha = 1e-2
    patient = 3
    s = 0

    def __init__(self, ifea, ofea, **kwargs):
        super().__init__(ifea, ofea, **kwargs)
        self.standard = {
            'GL': lambda x, y: x > self.alpha,
            'PQ': lambda x, y: x / y > self.alpha,
        }
        self.optim_record = None
        self.coef_ = list()

    def get_data_loader(self, train=True):
        """
        must be overwritten to split data loader as two part, include training and validating set
        :param train:
        :return:
        """
        raise NotImplementedError

    def ave_loss(self, loss_arr):
        """
        on mod: validating, set ave loss to attr: vl_ave_loss
        :param loss_arr:
        :return: scale
        """
        ave_loss = np.mean(loss_arr)
        if getattr(self, 'validate', None):
            setattr(self, 'vl_ave_loss', ave_loss)
        return ave_loss

    def _judge_stopping(self, standard, gloss_arr, ploss_arr, pk):
        epoch = self.kwargs['epoch']
        # get ave loss on validating set
        vl_ave_loss = getattr(self, 'vl_ave_loss')

        # generic loss array
        gloss_arr.append(vl_ave_loss)
        # cal generic loss: eva_t / optim eva_lt_t - 1
        gl = self.generic_loss(gloss_arr)

        # measure progress
        ploss_arr.append(vl_ave_loss)
        # measure progress loss only be calculated at non GL std and in K step
        if (epoch + 1) % self.step == 0 and standard != 'GL':
            # cal measure progress: sigma(eva_k) / (k * optim eva_between_t-k+1_k) - 1
            pk = self.measure_progress(ploss_arr)
            # zero record array
            ploss_arr = list()

        print('*****ValEpoch: %s ave_loss: %s gl: %s, pk: %s last-patient: %s / %s *****'
              % (epoch, vl_ave_loss, gl, pk, self.s, self.patient))
        self.coef_.append([gl, pk if pk else np.nan])
        return gl, pk, ploss_arr

    def stop(self, std, gl, pk, nojud=True):
        """
        early stopping standards, GL, PQ, within max patient
        :param nojud: Don't judge PQ
        :param std: GL, PQ
        :param gl: generic loss
        :param pk: measure progress
        :return: bool
        """
        if std == 'PQ' and nojud:
            return False
        func = self.standard[std]
        self.s += int(func(gl, pk))
        return self.s >= self.patient

    def train(self, iter_n, standard, checkpoint=0):
        """
        train model from training data loader in several epoch, each use whole batch to train
        :param iter_n: max epoch
        :param standard: 'GL', 'PQ'
        :return:
        """
        if standard not in self.standard.keys():
            raise ParametersError('standard only choose from GL / PQ / UP')
        # generic loss array
        gloss_arr = list()
        # progress loss array
        ploss_arr = list()
        # measure progress array
        pk = None

        # get split data loader
        tr_dl, va_dl = self.get_data_loader(train=True)

        # training epoch, max iteration N
        for epoch in range(iter_n):
            self.kwargs['epoch'] = epoch
            # use the whole batch training set to train model
            self.train_batch(tr_dl)
            # turn on validate mod
            setattr(self, 'validate', True)
            with th.no_grad():
                # use the whole batch validating set to train model
                # function inner call self.ave_loss
                self.eval_batch(va_dl)
            # turn off validate mod
            setattr(self, 'validate', False)

            # analysis stopping eval coef
            gl, pk, ploss_arr = self._judge_stopping(standard, gloss_arr, ploss_arr, pk)

            # should be stop
            if self.stop(standard, gl, pk, nojud=bool(ploss_arr)):
                print('Stop at %sth Iteration' % (epoch-1))
                break

            self.optim_record = (cp.deepcopy(self.graph), cp.deepcopy(self.optimizer))

            if checkpoint and (epoch + 1) % checkpoint == 0:
                self.checkpoint()

        # haven't get the optim stop point before max iteration
        else:
            print('Stop at Max Iteration')

        self.model_persistence()

    @staticmethod
    def generic_loss(eva_arr):
        """calculate generic loss"""
        eopt = min(eva_arr)
        return 100 * (eva_arr[-1] / eopt - 1)

    @staticmethod
    def measure_progress(eva_arr):
        """calculate measure progress"""
        eopt = min(eva_arr)
        return 1000 * (np.mean(eva_arr) / eopt - 1)

    def model_persistence(self):
        if self.optim_record is None:
            return
        name = self.model_graph_class.__name__
        names = 'mod%s.pkl' % name, 'opt%s.pkl' % name
        for obj, nm in zip(self.optim_record, names):
            self.save(obj, nm)
        path = os.path.join(self.csv_path, 'EpochESCoef%s.txt' % self.model_graph_class.__name__)
        pd.DataFrame(self.coef_, columns=['gl', 'mp']).to_csv(path, sep='\t', header=True, index=True, na_rep=-127)


class ESFeedForwardNet(EarlyStoppingMixin, GenericFeedForwardNet):

    pass


class Pretty:

    _bar = None
    _step = 40

    @classmethod
    def bar(cls, func):
        def inner(ins, dl):
            cls._bar = tqdm(dl)
            return func(ins, cls._bar)
        return inner

    @classmethod
    def desc(cls, func):
        name = func.__name__

        def inner(ins, arr):
            ret = func(ins, arr)
            if ins.kwargs['batch'] % cls._step == 0:
                s = "TrMod[{mod}] Epoch:{e} Batch:{b} {stat}:{r}".format(mod=ins.graph.training,
                                                                         e=ins.kwargs.get('epoch', 0),
                                                                         b=ins.kwargs.get('batch', 0),
                                                                         stat=name,
                                                                         r=round(ret, 6))
                cls._bar.set_description(s)
            return ret
        return inner


class PrettyFeedForward(ESFeedForwardNet):

    @Pretty.desc
    def ave_loss(self, loss_arr):
        return super().ave_loss(loss_arr)

    @Pretty.bar
    def train_batch(self, dl):
        """
        train model from data loader in whole batch, each use one batch to update parameters
        :param dl: training data loader
        """
        super().train_batch(dl)

    @Pretty.bar
    def eval_batch(self, dl):
        super().eval_batch(dl)
        time.sleep(0.1)
