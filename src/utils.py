import random
import torch
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset

LOG_2_PI = 1.8378770664093453


def empty(ls):
    """
    Check whether a collection is empty
    """
    return len(ls) == 0


def cosine_distance(x1, x2, dim=1):
    """
    Compute the cosine distance between `x1` and `x2`, the `dim` param
    by default is 1 (due to batches)
    """
    return 1.0 - F.cosine_similarity(x1, x2, dim=dim)


def loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar):
    """
    Compute the vae term from the loss function, it's already calculated
    to be minimized.
    """

    LGP = -torch.sum((-0.5 * LOG_2_PI) + (-0.5 * x_logvar) +
                     (-0.5 * (x - x_mu).pow(2) / (x_logvar.exp())), dim=1).mean(dim=0)

    KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) -
                           z_logvar.exp(), dim=1)

    return (KLD + LGP).mean()


def loss_term_cos(y, z_mu):
    """
    Compute the cos term from the loss function
    """

    batch_size = y.size(0)

    total = 0.0
    for _ in range(batch_size):
        i = random.randrange(0, batch_size)
        j = random.randrange(0, batch_size)
        while j == i:
            j = random.randrange(0, batch_size)

        sign = 1 if y[i] == y[j] else -1
        total += sign * cosine_distance(z_mu[i], z_mu[j], dim=0)

    return total


def model_criterion(x, y, x_mu, x_logvar, z_mu, z_logvar, vae_reg=1.0, cos_reg=1.0):
    """
    Compute the whole loss term (aka model criterion), note that here isn't
    included the mas term loss (it's already included in `LocalSgd`)
    """

    term_vae = loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar)
    term_cos = loss_term_cos(y, z_mu)

    return (vae_reg * term_vae) + (cos_reg * term_cos)


def split_classes_in_tasks(dataset: Dataset):
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


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from torchvision import transforms

    t = transforms.ToTensor()
    d = MNIST(root="./data", train=True, download=True, transform=t)
    t = split_classes_in_tasks(d)
