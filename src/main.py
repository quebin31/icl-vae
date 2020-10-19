import os
import train
import torch
import wandb

from argparse import ArgumentParser
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from modules.vicl import Vicl
from utils import split_classes_in_tasks

config = {
    "batch_size": 16,
    "learning_rate": 0.000003,
    "task": 0,
    "epochs": 10,
}

wandb.init(project="icl-vae", config=config)
torch.manual_seed(2)
torch.autograd.set_detect_anomaly(True)

transforms = transforms.Compose([
    transforms.ToTensor(),
])

data_train = CIFAR100(root="./data", train=True,
                      download=True, transform=transforms)
data_test = CIFAR100(root="./data", train=False,
                     download=True, transform=transforms)

vgg19_weights = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"

model = Vicl(vgg_weigths=vgg19_weights)
wandb.watch(model)
model = train.train(model, data_train, data_test,
                    task=config["task"], epochs=config["epochs"], batch_size=config["batch_size"], lr=config["learning_rate"], log_interval=50)
