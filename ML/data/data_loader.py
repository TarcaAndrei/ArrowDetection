from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_data_loader(*, dataset, batch_size, distributed, collate_fn, shuffle, drop_last=True):
    if distributed:
        return DataLoader(
            dataset,
            sampler=DistributedSampler(dataset, shuffle=shuffle),
            batch_size=max(batch_size // get_world_size(), 1),
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=drop_last,
        shuffle=shuffle,
    )
