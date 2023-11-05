from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
import os

from lightning.pytorch.loggers import WandbLogger
import lightning as L
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from CIFAR10DataModule import CIFAR10DataModule
from CIFAR10Model import CIFAR10Model

dm = CIFAR10DataModule()

wandb_logger = WandbLogger(project='wandb-cifar10', job_type='train')

model = CIFAR10Model()
trainer = L.Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
)
trainer.fit(model, dm)
trainer.test()
wandb.finish()
