import sys
import os
import re

import struct
from functools import reduce

import numpy as np
from PIL import Image


class MstarTransFormatter(object):

    DATA_DIR = None
    SAVE_DIR = None
    FORMATS = tuple()
    CHANEL = 1
    zero_operator = 255

    def __init__(self, root_dir):
        self.root = os.path.join(root_dir, self.DATA_DIR)
        self.err = None
        if not os.path.exists(self.root):
            self.err = "Setting legal data path"
            return
        self.mstar_files = os.listdir(self.root)
        self.save_root = os.path.join(root_dir, self.SAVE_DIR)
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

        self.formats = None
        self.endian = sys.byteorder.capitalize()
        self.tensors_ = list()
        self.pils_ = list()

    def to(self, *formats, float_size=4, img_format='chanel_last', gamma=2.2):
        """
        Mstar file to other format file
        :return:
        """
        if self.err:
            raise RuntimeError(self.err)
        self.formats = formats if formats else self.FORMATS
        print('CURRENT ROM STATUS: %s Endian' % self.endian)
        print("Total Files: %s" % len(self.mstar_files))
        for mstar in self.mstar_files:
            self.mstar2pic(mstar, float_size, img_format, gamma)
        return self

    def mstar2pic(self, name, float_size, img_format, gamma):
        mstar = os.path.join(self.root, name)
        pic_name = name.split('.', 1)[0]
        with open(mstar, 'rb') as f:
            header_infos = {"Columns": 0, "Rows": 0}
            header_info_pattern = re.compile(r'NumberOf(?P<key>(Columns)|(Rows))= (?P<value>\d+)')
            header_end_pattern = re.compile(r'EndofPhoenixHeader')
            while True:
                header_line = bytes.decode(f.readline()).strip()
                if re.search(header_end_pattern, header_line):
                    break
                info = re.search(header_info_pattern, header_line)
                if info:
                    header_infos[info.group('key')] = int(info.group('value'))
            print("Pic {name} size: {Rows} * {Columns}".format(name=pic_name, **header_infos))

            size = reduce(lambda x, y: x * y, header_infos.values())
            psize = list(header_infos.values())
            dim = psize.copy()
            if img_format is 'chanel_last':
                dim.append(self.CHANEL)
            else:
                dim.insert(0, self.CHANEL)

            pic_arr = np.array(struct.unpack('>'+'f'*size, f.read(float_size * size)))
            pic_tensor = pic_arr.reshape(psize)
            pic_tensor = self.enhance(pic_tensor, gamma)

            formats = list(self.formats)  # type: list
            if 'tensor' in formats:
                tensor = pic_arr.reshape(dim)
                self.tensors_.append(tensor)
                formats.remove('tensor')

            pil = Image.fromarray(pic_tensor)
            self.pils_.append(pil)
            for fmt in formats:
                pil.save(os.path.join(self.save_root, '.'.join([pic_name, fmt])))

    def enhance(self, arr: np.ndarray, gamma=1):
        a, b = np.min(arr), np.max(arr)
        ret = np.power((arr - a) / (b - a), 1 / gamma) * self.zero_operator
        return ret.astype(np.uint8)
