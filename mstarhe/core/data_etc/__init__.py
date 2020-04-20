class MstarDS(object):

    def __init__(self, data: list):
        self.data = data
        self.samples = list()
        self.indices = list()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return self.data + other


empty = MstarDS(list())
