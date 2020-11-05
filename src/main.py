import os
import train
import test
import torch
import wandb
import argparse
import random

from argparse import Namespace
from config import Config
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from modules.vicl import Vicl
from utils import split_classes_in_tasks


def valid_args(args: Namespace):
    valid = args.train or args.test
    if not valid:
        print('error: --train and/or --test should be provided')
        valid = False

    if args.train and args.config is None:
        print('error: --config should be provided when training')
        valid = False

    if args.test and (args.run_id is None and not args.train):
        print('error: --test should be used together with --train and/or --run-id')
        valid = False

    return valid


parser = argparse.ArgumentParser(description='Train/test vicl model')
parser.add_argument('-c', '--config', type=str, help='Hyperparameters config')
parser.add_argument('-t', '--task', type=int, required=True,
                    help='Task number (starts at 0)')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--run-id', type=str, help='Optional run id')

args = parser.parse_args()
if not valid_args(args):
    exit(1)

config = Config.load(args.config)
wandb.init(project='icl-vae', entity='kdelcastillo',
           resume=args.run_id if args.run_id else False, config=config.to_dict())
wandb.save('*.pt')

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

transforms = transforms.Compose([transforms.ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

vgg19_weights = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
model = Vicl(vgg_weights=vgg19_weights).to(device)

if args.train:
    data_train = CIFAR100(root='./data', train=True,
                          download=True, transform=transforms)

    if config.task != 0:
        print(f'Loading previous task ({config.task - 1}) model for training')
        model.load(wandb.restore(f'vicl_task_{config.task - 1}.pt'))

    wandb.watch(model)
    model = train.train(model, data_train, task=args.task, config=config)

if args.test:
    if not args.train:
        model.load(wandb.restore(f'vicl-task-{config.task}.pt'))

    data_test = CIFAR100(root='./data', train=False,
                         download=True, transform=transforms)
    model = test.test(model, data_test, task=args.task,  batch_size=16)
