import torch
from torch.utils.data import DataLoader
import lightning as ltn
from lhotse.dataset import DynamicBucketingSampler, make_worker_init_fn
from lhotse.dataset import SimpleCutSampler as SingleCutSampler
import torch.distributed as dist


class Datamodule(ltn.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, test_dataset=None, use_bucket=True,
                 max_duration=100, num_buckets=10, shuffle=True, show_statistic=False,
                 buffer_size=100, shuffle_buffer_size=1000, seed=42,
                 prefetch_factor=2, num_workers=8, batch_size=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.shuffle = shuffle
        self.max_duration = max_duration
        self.num_buckets = num_buckets
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.show_statistic = show_statistic
        self.use_bucket = use_bucket
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        if self.show_statistic:
            self.train_dataset.cuts_data.describe()
            self.valid_dataset.cuts_data.describe()

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        train_cuts_data = self.train_dataset.cuts_data
        train_sampler = DynamicBucketingSampler(
            train_cuts_data,
            shuffle=self.shuffle,
            drop_last=False,
            max_duration=self.max_duration,
            num_buckets=self.num_buckets,
            buffer_size=self.buffer_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        )

        return DataLoader(self.train_dataset,
                          sampler=train_sampler,
                          batch_size=None,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=False,
                          )

    def val_dataloader(self):
        valid_cuts_data = self.valid_dataset.cuts_data
        val_sampler = DynamicBucketingSampler(
                valid_cuts_data,
                shuffle=False,
                drop_last=True,
                max_duration=self.max_duration,
                num_buckets=self.num_buckets,
                buffer_size=self.buffer_size,
                shuffle_buffer_size=self.shuffle_buffer_size,
                seed=self.seed,
            )
        return DataLoader(self.valid_dataset,
                          sampler=val_sampler,
                          batch_size=None,
                          # For faster dataloading, use num_workers > 1
                          num_workers=self.num_workers,
                          # Note: Lhotse offers its own "worker_init_fn" that helps properly
                          #       set the random seeds in all workers (also with multi-node training)
                          #       and randomizes the shard order across different workers.
                          # worker_init_fn=make_worker_init_fn(self.rank, self.num_replicas, seed=self.seed),
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=False,
                          )


    def test_dataloader(self):
        test_cuts_data = self.test_dataset.cuts_data
        test_sampler = DynamicBucketingSampler(
                test_cuts_data,
                shuffle=False,
                drop_last=True,
                max_duration=self.max_duration,
                num_buckets=self.num_buckets,
                buffer_size=self.buffer_size,
                shuffle_buffer_size=self.shuffle_buffer_size,
                seed=self.seed,
            )
        return DataLoader(self.valid_dataset,
                          sampler=test_sampler,
                          batch_size=None,
                          # For faster dataloading, use num_workers > 1
                          num_workers=self.num_workers,
                          # Note: Lhotse offers its own "worker_init_fn" that helps properly
                          #       set the random seeds in all workers (also with multi-node training)
                          #       and randomizes the shard order across different workers.
                          # worker_init_fn=make_worker_init_fn(self.rank, self.num_replicas, seed=self.seed),
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=False,
                          )

    def predict_dataloader(self):
        return None
