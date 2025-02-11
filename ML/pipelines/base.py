import os
from abc import ABC, abstractmethod

import torch
from torch import distributed as dist


class BaseRunner(ABC):
    def __init__(self, distributed):
        self.distributed = distributed
        if distributed:
            dist.init_process_group(backend="nccl")
            self.device = torch.device(int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device=self.device)
            dist.barrier()
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    @abstractmethod
    def run(self):
        pass

    def end(self):
        if self.distributed:
            dist.destroy_process_group()

    def __call__(self):
        self.run()
        self.end()
