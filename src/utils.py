import random
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

LOG_2_PI = 1.8378770664093453


def empty(ls) -> bool:
    """Check wheter a collection is empty

    Args:
        ls: Collection to check.

    Returns:
        bool: Collection is empty.
    """
    return len(ls) == 0


def calculate_std(logvar: Tensor) -> Tensor:
    """Calculate the standard deviation from log(std^2)

    Args:
        logvar (Tensor): log(std^2) .

    Returns:
        Tensor: Tensor with standard deviation calculated (same shape).
    """
    return torch.exp(0.5 * logvar)


def calculate_var(logvar: Tensor) -> Tensor:
    """Calculate variance from log(std^2)

    Args:
        logvar (Tensor): log(std^2).

    Returns:
        Tensor: Tensor with variance calculated (same shape).
    """
    return torch.exp(logvar)


def cosine_distance(x1: Tensor, x2: Tensor, dim: int = 1) -> Tensor:
    """Compute the cosine distance between two tensors  

    Args:
        x1 (Tensor): One tensor.
        x2 (Tensor): Another tensor.
        dim (int, optional): Dimension to compute. Defaults to 1.

    Returns:
        Tensor: Cosine distance on the given dimension.
    """
    return 1.0 - F.cosine_similarity(x1, x2, dim=dim)


def loss_term_vae(x: Tensor, x_mu: Tensor, x_logvar: Tensor, z_mu: Tensor, z_logvar: Tensor, eps: float = 1e-6) -> float:
    """Compute the vae term from the loss function, it's already calculated
    to be minimized.

    Args:
        x (Tensor): Original `x` input tensor.   
        x_mu (Tensor): mu from decoder.
        x_logvar (Tensor): logvar from decoder.
        z_mu (Tensor): mu from encoder.
        z_logvar (Tensor): logvar from decoder.
        eps (float, optional): Tiny epsilon to improve numerical stability. Defaults to 1e-6.

    Returns:
        float: VAE loss term for gaussian encoder/decoder.
    """

    # Term log p(x|z) based on:
    # https://github.com/y0ast/Variational-Autoencoder/blob/master/VAE.py#L118
    LGP = -torch.sum((-0.5 * LOG_2_PI) + (-0.5 * x_logvar) +
                     (-0.5 * (x - x_mu).pow(2) / (x_logvar.exp() + eps)), dim=1).mean()
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1).mean()

    return KLD + LGP


def loss_term_cos(y: Tensor, z_mu: Tensor, z_logvar: Tensor, rho: float) -> float:
    """Compute the cosine term from the loss function.

    Args:
        y (Tensor): Labels.
        z_mu (Tensor): mu from decoder.
        z_logvar (Tensor): logvar from decoder.
        rho (float): Value rho from loss function.

    Returns:
        float: Cosine loss term.
    """

    batch_size = y.size(0)
    z_var = calculate_var(z_logvar)

    total = 0.0
    for _ in range(batch_size):
        i = random.randrange(0, batch_size)
        j = random.randrange(0, batch_size)
        while j == i:
            j = random.randrange(0, batch_size)

        sign = 1 if y[i] == y[j] else -1
        total += sign * \
            (rho * cosine_distance(z_mu[i], z_mu[j], dim=0) +
             (1 - rho) * cosine_distance(z_var[i], z_var[j], dim=0))

    return total


def loss_term_l1(z_mu: Tensor) -> float:
    """Compute the L1 term from the loss function.

    Args:
        z_mu (Tensor): mu from encoder.

    Returns:
        float: L1 loss term.
    """
    return z_mu.abs().sum()


class ViclLoss(object):
    def __init__(self, rho: float, lambda_vae: float, lambda_cos: float, lambda_l1: float):
        """Create the loss functor to compute the vicl loss value.

        Args:
            rho (float): Value rho from loss function.
            lambda_vae (float, optional): Lambda for vae term. 
            lambda_cos (float, optional): Lambda for cosine term. 
            lambda_l1 (float, optional): Lambda for L1 term.
        """
        self.rho = rho
        self.lambda_vae = lambda_vae
        self.lambda_cos = lambda_cos
        self.lambda_l1 = lambda_l1

    def __call__(self, x: Tensor, y: Tensor, x_mu: Tensor, x_logvar: Tensor, z_mu: Tensor, z_logvar: Tensor) -> float:
        """Compute the loss function for the VICL model.

        Args:
            x (Tensor): Original `x` input.
            y (Tensor): Labels.
            x_mu (Tensor): mu from decoder.
            x_logvar (Tensor): logvar from decoder.
            z_mu (Tensor): mu from encoder.
            z_logvar (Tensor): logvar from encoder.
        Returns:
            float: Loss function value.
        """

        term_vae = loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar)
        term_cos = loss_term_cos(y, z_mu, z_logvar, self.rho)
        term_l1 = loss_term_l1(z_mu)

        return (self.lambda_vae * term_vae) + (self.lambda_cos * term_cos) + (self.lambda_l1 * term_l1)


def split_classes_in_tasks(dataset: Dataset) -> List[List[int]]:
    """Split classes from dataset into an array of array 

    Args:
        dataset (Dataset): The dataset to get classes from.

    Returns:
        List[List[int]]: List of list of class indices.
    """
    label_indices = {}

    for idx, (_, label) in enumerate(dataset):
        label_indices.setdefault(label, []).append(idx)

    labels = list(label_indices.keys())
    labels.sort()

    mid = len(labels) // 2
    tasks_indices = [[]]

    for i in range(0, mid):
        tasks_indices[0] += label_indices[labels[i]]

    for i in range(mid, len(labels)):
        tasks_indices.append(label_indices[labels[i]])

    return tasks_indices


def create_subset(dataset: Dataset, task: int, tasks_indices: List[List[int]], accumulate: bool) -> Subset:
    """Create subset from task indices for the given task, possibly accumulating classes
    before it.

    Args:
        dataset (Dataset): The original dataset.
        task (int): The task number, starts at 0.
        tasks_indices (List[List[int]]): Task indices from split_classes_in_tasks.
        accumulate (bool): Whether to accumulate classes from task - 1, task - 2, etc.

    Returns:
        Subset: The subset containing only the needed classes.  
    """
    start_idx = 0 if accumulate else task

    indices = []
    for t in range(start_idx, task + 1):
        indices += tasks_indices[t]

    return Subset(dataset, indices)


if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.datasets import MNIST

    t = transforms.ToTensor()
    d = MNIST(root='./data', train=True, download=True, transform=t)
    t = split_classes_in_tasks(d)
