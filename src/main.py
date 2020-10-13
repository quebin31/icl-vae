from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from modules.vicl import Vicl


transforms = transforms.Compose([
    transforms.ToTensor()
])

mnist_data_train = MNIST(root="./data", train=True,
                         download=True, transform=transforms)
mnist_data_test = MNIST(root="./data", train=False,
                        download=True, transform=transforms)

data_train_loader = DataLoader(mnist_data_train, )


vgg19_weights = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
