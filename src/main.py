import os
import train
import test
import torch
import wandb
import argparse
import random
import yaml

from argparse import Namespace
from config import load_config_dict, Config
from halo import Halo
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from modules.vicl import Vicl
from utils import split_classes_in_tasks
from yaml import Loader


def valid_args(args: Namespace):
    valid = args.train or args.test
    if not valid:
        print('error: --train and/or --test should be provided')
        valid = False

    if args.train and args.task == 0 and not args.config:
        print('error: --config should be provided when training task 0')
        valid = False

    if args.train and args.task != 0 and not args.id:
        print('error: --id should be provided when training task != 0')

    if args.test and (not args.id and not args.train):
        print('error: --test should be used together with --train and/or --run-id')
        valid = False

    return valid


parser = argparse.ArgumentParser(description='Train/test vicl model')
parser.add_argument('-c', '--config', type=str, help='Hyperparameters config')
parser.add_argument('-t', '--task', type=int, required=True,
                    help='Task number (starts at 0)')
parser.add_argument('--id', type=str, help='Optional run id (resuming)')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--train', action='store_true', help='Train the model')

args = parser.parse_args()
if not valid_args(args):
    exit(1)

resume = 'must' if args.id else None
config = load_config_dict(args.config)
wandb.init(project='icl-vae',
           entity='kdelcastillo',
           id=args.id,
           resume=resume,
           config=config)

config = Config(wandb.config)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.autograd.set_detect_anomaly(False)

transforms = transforms.Compose([transforms.ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = Vicl().to(device)

if args.train:
    data_train = CIFAR100(root='./data', train=True,
                          download=True, transform=transforms)

    if args.task != 0:
        prev = args.task - 1
        text = f'Loading model for task {prev}'
        halo = Halo(text=text, spinner='dots').start()
        try:
            handler = wandb.restore(f'vicl-task-{prev}.pt', replace=True)
        except:
            handler = None

        if handler:
            model.load(handler.name)
            halo.succeed(f'Successfully loaded model for task {prev}')
        else:
            halo.fail(f'Failed to load model for task {prev}')
            exit(1)

    wandb.watch(model)
    model = train.train(model, data_train, task=args.task, config=config)

if args.test:
    if not args.train:
        text = f'Loading model for task {args.task}'
        halo = Halo(text=text, spinner='dots').start()
        try:
            handler = wandb.restore(f'vicl-task-{args.task}.pt', replace=True)
        except:
            handler = None

        if handler:
            model.load(handler.name)
            halo.succeed(f'Successfully loaded model for task {args.task}')
        else:
            halo.fail(f'Failed to load model for task {args.task}')
            exit(1)

    data_test = CIFAR100(root='./data', train=False,
                         download=True, transform=transforms)
    model = test.test(model, data_test, task=args.task,  batch_size=16)
