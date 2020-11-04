import os
import train
import torch
import wandb
import argparse
import random

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from modules.vicl import Vicl
from utils import split_classes_in_tasks
from argparse import Namespace


def valid_args(args: Namespace):
    valid = args.train or args.test
    if not valid:
        print('error: --train and/or --test should be provided')

    if args.train:
        check = {
            'lr': args.lr,
            'batch': args.batch,
            'rlambda': args.rlambda,
            'epochs': args.epochs,
            'task': args.task,
        }

        for arg, val in check.items():
            if val is None:
                print(f'error: --{arg} is required if using --train')
                valid = False

        if args.task != 0:
            if args.run_id is None:
                print(
                    'error: --run-id is required if using --train with --task different from 0')
                valid = False

    if args.test and not args.train:
        check = {
            'batch': args.batch,
            'task': args.task,
            'run-id': args.run_id,
        }

        for arg, val in check.items():
            if val is None:
                print(
                    f'error: --{arg} is required if using --test without --train')
                valid = False

    args.seed = args.seed or random.randint(0, 500)
    return valid


parser = argparse.ArgumentParser(description='Train/test vicl model')

parser.add_argument('-b', '--batch', type=int, help='Batch size')
parser.add_argument('-l', '--lr', type=float, help='Initial learning rate')
parser.add_argument('-r', '--rlambda', type=float, help='Lambda reg params')
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs')
parser.add_argument('-s', '--seed', type=int, help='Initial random seed')
parser.add_argument('-t', '--task', type=int, help='Task number (starts at 0)')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--run-id', type=str,
                    help='Optional run id')

args = parser.parse_args()
if not valid_args(args):
    exit(1)

wandb.init(project='icl-vae', entity='kdelcastillo',
           resume=args.run_id if args.run_id else False)
wandb.config.update(args, allow_val_change=True)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

transforms = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
config = wandb.config

vgg19_weights = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
model = Vicl(vgg_weights=vgg19_weights).to(device)

if config.train:
    data_train = CIFAR100(root='./data', train=True,
                          download=True, transform=transforms)

    if config.task != 0:
        print(f'Loading previous task ({config.task - 1}) model for training')
        model.load(wandb.restore(f'vicl_task_{config.task - 1}.pt'))

    wandb.watch(model)
    model = train.train(model, data_train, task=config.task, epochs=config.epochs,
                        batch_size=config.batch, lr=config.lr, reg_lambda=config.rlambda, log_interval=50)

if config.test:
    if not config.train:
        model.load(wandb.restore(f'vicl_task_{config.task}.pt'))

    data_test = CIFAR100(root='./data', train=False,
                         download=True, transform=transforms)
