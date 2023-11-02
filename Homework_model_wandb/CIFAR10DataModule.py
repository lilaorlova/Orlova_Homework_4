import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 4



class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dims = (3, 32, 32)
        self.num_classes = 10


    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True,
                                     download=True, transform=self.transform)

        torchvision.datasets.CIFAR10(self.data_dir, train=False,
                                     download=True, transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            cifar10_full = torchvision.datasets.CIFAR10(self.data_dir, train=True,
                                                download=False, transform=self.transform)
            self.trainset, self.valset = random_split(cifar10_full, [0.7, 0.3])


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.testset = torchvision.datasets.CIFAR10(self.data_dir, train=False,
                                                        download=False, transform=self.transform)



    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.testtest, batch_size=BATCH_SIZE)
