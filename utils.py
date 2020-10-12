import torch
import torch.nn.functional as F
import random

LOG_2_PI = 1.8378770664093453


def empty(ls):
    return len(ls) == 0


def cosine_distance(x1, x2, dim=1):
    return 1.0 - F.cosine_similarity(x1, x2, dim=dim)


def loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar):
    # negative log likelihood
    LGP = LOG_2_PI + x_logvar + (x - x_mu) ** 2 / (2 * torch.exp(x_logvar))
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    return KLD + LGP


def loss_term_cos(y, z_mu):
    batch_size = y.size(0)

    total = 0.0
    for _ in range(batch_size):
        i = random.randrange(0, batch_size)
        j = random.randrange(0, batch_size)
        while j == i:
            j = random.randrange(0, batch_size)

        sign = 1 if y[i] == y[j] else -1
        total += sign * cosine_distance(z_mu[i], z_mu[j])

    return total
