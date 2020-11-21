import argparse
import os
import random
from argparse import Namespace
from pathlib import Path

import torch
import wandb
import yaml
from halo import Halo
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from yaml import Loader

from config import Config, load_config_dict
from modules.vicl import Vicl
from mtest import test
from train import train
from utils import split_classes_in_tasks


def valid_args(args: Namespace):
    valid = args.train or args.test
    if not valid:
        print('error: --train and/or --test should be provided')
        valid = False

    if args.train and args.task == 0 and not args.config:
        print('error: --config should be provided when training task 0')
        valid = False

    if args.train and args.task != 0 and not args.runid:
        print('error: --id should be provided when training task != 0')

    if args.test and (not args.runid and not args.train):
        print('error: --test should be used together with --train and/or --run-id')
        valid = False

    return valid


def run_name(config_path: str, task: int):
    config_path = Path(config_path)
    return f'{config_path.stem} task {task}'


models_dir = os.getenv("MODELS_DIR", "./models")
parser = argparse.ArgumentParser(description='Train/test vicl model')
parser.add_argument('-c', '--config', type=str, help='Hyperparameters config')
parser.add_argument('-t', '--task', type=int, required=True,
                    help='Task number (starts at 0)')
parser.add_argument('--runid', type=str, help='Previous task run id')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--train', action='store_true', help='Train the model')

args = parser.parse_args()
if not valid_args(args):
    exit(1)

if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)

if not os.path.isdir(models_dir):
    print(f'error: MODELS_DIR={models_dir} is not a directory')
    exit(1)

config = load_config_dict(args.config)
name = run_name(args.config, args.task)
wandb.init(project='icl-vae', entity='kdelcastillo', name=name, config=config)

with open('.runid', 'w') as runid_file:
    runid_file.write(wandb.run.id)

config = Config(wandb.config)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.autograd.set_detect_anomaly(False)

transforms = transforms.Compose([transforms.ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = Vicl().to(device)

if args.train:
    if args.task != 0:
        prev_task = args.task - 1
        text = f'Loading model for task {prev_task}'
        halo = Halo(text=text, spinner='dots').start()

        try:
            model.load(f'{models_dir}/{args.runid}/vicl-task-{prev_task}.pt')
            halo.succeed(f'Successfully loaded model for task {prev_task}')
        except Exception as e:
            halo.fail(f'Failed to load model for task {prev_task}: {e}')
            exit(1)

    try:
        wandb.watch(model)
        data_train = CIFAR100(root='./data', train=True,
                              download=True, transform=transforms)
        models_run_dir = os.path.join(models_dir, wandb.run.id)
        os.makedirs(models_run_dir, exist_ok=True)
        model = train(model, data_train, task=args.task,
                      config=config, models_dir=models_run_dir)
    except Exception as e:
        print(f'Training failed: {e}')
        exit(1)

if args.test:
    if not args.train:
        text = f'Loading model for task {args.task}'
        halo = Halo(text=text, spinner='dots').start()

        try:
            model.load(f'{models_dir}/{args.runid}/vicl-task-{args.task}.pt')
            halo.succeed(f'Successfully loaded model for task {args.task}')
        except Exception as e:
            halo.fail(f'Failed to load model for task {args.task}: {e}')
            exit(1)

    data_test = CIFAR100(root='./data', train=False,
                         download=True, transform=transforms)
    model = test(model, data_test, task=args.task,  batch_size=16)
