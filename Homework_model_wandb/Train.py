from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
import os

import lightning as L
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms

dm = CIFAR10DataModule()

wandb_logger = WandbLogger(project='wandb-cifar10', job_type='train')

model = CIFAR10Model()
trainer = L.Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1,
)
trainer.fit(model, dm)
trainer.test()
wandb.finish()