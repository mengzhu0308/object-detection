#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/2/1 11:35
@File:          sampler.py
'''

import random

class SequentialSampler:
    def __init__(self, dataset):
        super(SequentialSampler, self).__init__()
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

class RandomSampler:
    def __init__(self, dataset):
        super(RandomSampler, self).__init__()
        self.dataset = dataset

    def __iter__(self):
        indexs = list(range(len(self.dataset)))
        random.shuffle(indexs)
        return iter(indexs)

    def __len__(self):
        return len(self.dataset)

class BatchSampler:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        super(BatchSampler, self).__init__()
        self.batch_size = batch_size
        self.drop_last = drop_last
        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        batch_index = []
        for idx in self.sampler:
            batch_index.append(idx)
            if len(batch_index) == self.batch_size:
                yield batch_index
                batch_index = []
        if len(batch_index) > 0 and not self.drop_last:
            yield batch_index

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

