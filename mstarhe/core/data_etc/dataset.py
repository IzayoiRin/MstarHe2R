import os
import re
import time
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch as th
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mstarhe.conf import LazySettings
from mstarhe.core.nn import functions as F
from mstarhe.core.data_etc import MstarDS, empty


settings = LazySettings()


class MstarDataSetBase(Dataset):
    DATA_DIR = None
    SAVE_DIR = None

    prefixed = None
    setting_mode = ("test", "train")

    TRANSFORMS = list()

    __target_dir_pattern__ = "{dirPrefixed}.{settingMode}.{targets}"
    __img_formats__ = ["chanel_last", "chanel_first"]
    __chanel__ = 1
    __transforms_func__ = dict()
    __cache_file_name__ = "Mastar_Cached"

    def __init__(self, root_dir, train, img_format, transform):
        self.root = os.path.join(root_dir, self.DATA_DIR)
        self.save_root = os.path.join(root_dir, self.SAVE_DIR)
        self.img_format = img_format
        self.train = train
        self.transform = transform or self._default_transform
        self._data = list()
        self.data = empty
        self._samples = list()

    @property
    def samples(self):
        return self.data.samples

    @property
    def _default_transform(self):
        seq = list()
        for key in self.TRANSFORMS:
            if self.__transforms_func__.get(key, None):
                seq.append(self.__transforms_func__[key])
        if len(seq):
            return Compose(seq)

    def get_target_dir_pattern(self, targets):
        return self.__target_dir_pattern__.format(
            dirPrefixed=self.prefixed or "",
            settingMode=self.setting_mode[int(self.train)],
            targets=targets
        ).split("@")

    def _resolute_path(self, targets):
        pattern = self.get_target_dir_pattern(targets)
        pattern[0] = os.path.join(self.root, *pattern[0].split('.'))
        pattern.reverse()
        return pattern

    def _fetch_files(self, path_list, patterns=None):
        if isinstance(path_list, str):
            path_list = [path_list]
        ret = list()
        patterns = patterns.copy()
        if not len(patterns):
            return path_list
        pattern = patterns.pop()
        for p in path_list:
            if os.path.isdir(p):
                pl = [os.path.join(p, f) for f in os.listdir(p) if re.match(pattern, f)]
                ret.extend(self._fetch_files(pl, patterns))
            else:
                ret.append(p)
        return ret

    def set_label(self, file):
        raise NotImplementedError

    def set_sample(self, file):
        self._samples.append(file)

    def loader(self, file):
        raise NotImplementedError

    def _load2tensor(self, files, label=None):
        bar = tqdm(enumerate(files), total=len(files))
        for idx, f in bar:
            label = label or self.set_label(f)
            self.set_sample(f)
            value = self.to_std_img_tensor(self.loader(f))
            self._data.append((value, label))
            bar.set_description_str("{label}:{num}".format(label=label, num=idx + 1))

    def to_std_img_tensor(self, img_tensor: th.Tensor):
        if callable(self.transform):
            img_tensor = self.transform(img_tensor)
        try:
            fmt = self.__img_formats__.index(self.img_format)
        except ValueError:
            self.img_format = settings.IMG_TENSOR_FORMAT
            fmt = 1
        fmt -= 1
        if img_tensor.size(fmt) == self.__chanel__:
            return img_tensor
        return img_tensor.transpose(0, -1)

    def _initial_from_target_class(self, target, label=None):
        path_list = self._resolute_path(target)
        files = self._fetch_files(path_list.pop(), path_list)
        self._load2tensor(files, label)
        return len(files)

    def to_pickles(self, name):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        with open(os.path.join(self.save_root, "%s.pkl" % name), "wb") as buffer:
            pickle.dump(self.data, buffer)
            print("Saved")

    def load_data_from_pickle(self, name):
        with open(os.path.join(self.save_root, "%s.pkl" % name), 'rb') as buffer:
            return pickle.load(buffer)

    def __call__(self, *targets):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{name} with {length} records on mod-{mode}".format(
            name=self.__class__.__name__,
            length=len(self),
            mode=int(self.train)
        )


class MstarTextDataSet(MstarDataSetBase):

    TARGETS = dict()
    TRANSFORMS = ["min_max_norm", ]

    __transforms_func__ = {
        "min_max_norm": F.ImgMinMaxNormalize(std_operator=255.0)
    }

    def __init__(self, root_dir=None, train=True, img_format=None, verbose=True, transform=None):
        root_dir = root_dir or settings.DATA_DIR
        super(MstarTextDataSet, self).__init__(root_dir, train, img_format, transform)
        self.kwargs = dict()
        self._label_mapping = dict()
        self.verbose = verbose
        self.__cached = True

    @property
    def label_mapping_dict(self):
        return getattr(self.data, "lmapping", dict())

    def set_label(self, file: str):
        temps = file.rsplit("\\", 3)
        class_name = "-".join(temps[1:3]) if self.verbose else temps[1]
        if self._label_mapping.get(class_name, None) is None:
            self._label_mapping[class_name] = self.kwargs['class']
        return self._label_mapping[class_name]

    def set_sample(self, file: str):
        temps = file.rsplit("\\", 3)
        temps[-1] = temps[-1].split('.')[0]
        if not self.verbose:
            self._samples.append(temps[-1])
        else:
            self._samples.append("-".join(temps[1:]))

    def loader(self, file):
        """load data form one file, return th.Tensor"""
        df = pd.read_csv(file, header=None, sep='  ', engine='python')
        ts = th.tensor(df.to_numpy(dtype=np.float32))
        if self.__chanel__ == 1:
            h, w = ts.size()
            ts = ts.view(self.__chanel__, h, w)
        return ts

    def initial(self, *targets):
        indices = list()
        i = 0
        for idx, target in enumerate(targets):
            self.kwargs["class"] = idx
            indices.append(i)
            i += self._initial_from_target_class(target)
            time.sleep(0.1)
        self.data = MstarDS(self._data)
        self.data.samples = self._samples
        self.data.indices = indices
        setattr(self.data, "lmapping", self._label_mapping)

    @property
    def cached(self):
        return self.__cached

    def flush(self):
        self.__cached = False
        self.data = empty

    def __cache__(self, save=False):
        """cache core dataset, return whether cached"""
        name = "%s_%s" % (self.__cache_file_name__, int(self.train))
        # cached and try to load
        if self.__cached:
            try:
                self.data = self.load_data_from_pickle(name)
            # fail to load cache
            except Exception as e:
                print(e)
                # return no cached
                self.__cached = False
        # no cached and try to save
        elif save:
            try:
                self.to_pickles(name)
            # fail to save cache
            except Exception as e:
                print(e)
            # success
            else:
                # return cached
                self.__cached = True

        return self.__cached

    def __call__(self, *targets, train=None, flush=False):
        if train is not None:
            self.train = train
        if flush:
            self.flush()
        if self.__cache__():
            return self
        t = targets or self.TARGETS[bool(self.train)]
        self.initial(*t)
        self.__cache__(True)
        return self


def _example():
    from mstarhe.conf import LazySettings

    if os.environ.get("MSTARHE_SETTING_MODULE", None) is None:
        import mstarhe as mh
        mh.setup()

    setting = LazySettings()
    MstarTextDataSet.DATA_DIR = "TensorData"
    MstarTextDataSet.SAVE_DIR = "STDataSet"
    MstarTextDataSet.prefixed = "adjust3"
    MstarTextDataSet.setting_mode = ("TEST", "TRAIN")

    MstarTextDataSet.TARGETS = {
        1: [
            "17_DEG@BMP2@\D+9563@.*",
            "17_DEG@BTR70@.*@.*",
            "17_DEG@T72@\D+132@.*",
        ],
        0: [
            "15_DEG@BMP2@\D+9563@.*",
            "15_DEG@BTR70@.*@.*",
            "15_DEG@T72@\D+132@.*",
        ]
    }

    return MstarTextDataSet(), MstarTextDataSet


if __name__ == '__main__':
    m, mc = _example()
    print(m(train=True, flush=True))
    print(mc()(train=False, flush=True))
    # m(train=True, flush=False)
    # a = tuple([m[i][-1] for i in range(10)])
    # print(a)
    # print(m.label_mapping_dict)
