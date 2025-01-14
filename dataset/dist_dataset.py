#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from typing import List, Any
import warnings
import random
from itertools import cycle
import torch
import json
import os
from torch.utils.data import IterableDataset


class DistLineReadingDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """

    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = False,
                 repeat: bool = False):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.files = []
        for t in data_path:
            folder = t[0] # t[1] is image root
            if os.path.isdir(folder):
                self.files.extend([(os.path.join(folder, d), t[1]) for d in os.listdir(folder)])
            elif os.path.isfile(folder):
                self.files.append((folder, t[1]))
            else:
                print('Path {} is invalid'.format(folder))
                sys.stdout.flush()

        self.repeat = repeat
        print('[DATA]--all dataset containing {} files.'.format(len(self.files)))
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        sys.stdout.flush()

    def generate(self):
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(
                self.files, self.rank, self.world_size)

        while True:
            if self.shuffle:
                random.shuffle(cur_dataloader_files)
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0:
                    print('[DATA]--current dataloader %s file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))
                cur_worker_files = split_shard(
                    cur_dataloader_files, worker_info.id, worker_info.num_workers)
                if worker_info.id == 0:
                    print("[DataLoader] --> Rank:{}  Workers:[{} ~ {}][{}]  Size of process file:{}  ...".format(
                        self.rank, 0, worker_info.num_workers - 1, worker_info.id, len(cur_dataloader_files)))
            else:
                cur_worker_files = cur_dataloader_files

            if self.shuffle:
                random.shuffle(cur_worker_files)
            for filepath, img_root in cur_worker_files:
                with open(filepath, 'r') as reader:
                    data = json.load(reader)
                    for sample in data:
                        yield sample, img_root

            if not self.repeat:
                break

    def __iter__(self):
        return self.generate()  


def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]
