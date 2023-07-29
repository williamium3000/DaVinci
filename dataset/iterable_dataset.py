import torch
import numpy as np
import torch.distributed as dist

class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, map_dataset, rank, world_size, repeat=10000):
        super().__init__()
        self.map_dataset = map_dataset
        self.length = len(self.map_dataset)
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.repeat = repeat
        
    def __len__(self,):
        return self.length // self.world_size
    
        
    def __iter__(self):
        self.epoch += 1
        permuted_idx = np.concatenate([np.random.default_rng(self.epoch + 10 * i).permutation(self.length) for i in range(self.repeat)])
        for i, index in enumerate(permuted_idx):
            if (i + self.rank) % self.world_size == 0:
                yield self.map_dataset.__getitem__(index)