import torch
from datautils import MyTrainDataset
from torch.utils.data import Dataset, DataLoader

# Additional imports
import torch.multiprocessing as mp  # Wrapper around Python's multiprocessing
from torch.utils.data.distributed import DistributedSampler  # Distributes data over all GPUs
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP wrapper
from torch.distributed import init_process_group, destroy_process_group  # Functions to initialize and destroy process groups
import os

# World size is the total number of processes in a group
# Group is created to maintain communication between GPUs
# Each GPU hosts one group
# Rank is a unique identifier assigned to each process

def ddp_setup(rank, world_size):
    """
    Initializes the Distributed Data Parallel (DDP) setup.
    Args:
        rank (int): Unique identifier for the process.
        world_size (int): Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_dataset))[0])  # Correct way to get batch size
        print(
            f"[GPU{self.gpu_id}] Epoch{epoch} | Batchsize:{b_sz} | Steps:{len(self.train_dataset)}"
        )
        for source, target in self.train_dataset:
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(source, target)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch{epoch} | Training checkpoint saved at checkpoint.pt")

    def train(self, epochs):
        for epoch in range(epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    """
    Loads training objects: dataset, model, and optimizer.
    """
    train_set = MyTrainDataset(1594)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
    Prepares a DataLoader for distributed training.
    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Batch size for the DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(rank: int, world_size: int, total_epochs: int, save_every: int):
    """
    Main training function for a single process.
    Args:
        rank (int): Rank of the process.
        world_size (int): Total number of processes.
        total_epochs (int): Number of epochs to train.
        save_every (int): Frequency of saving checkpoints.
    """
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=100)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()

    mp.spawn(
        main,
        args=(world_size, total_epochs, save_every),
        nprocs=world_size,
        join=True,  # Ensures the processes are joined before exiting
    )
