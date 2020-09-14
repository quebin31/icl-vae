import torch
import torch.nn.functional as F

LOG_2_PI = 1.8378770664093453


def empty(ls):
    return len(ls) == 0


def cosine_distance(x1, x2, dim=0):
    return 1.0 - F.cosine_similarity(x1, x2, dim=dim)


def loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar):
    # negative log likelihood
    LGP = LOG_2_PI + x_logvar + (x - x_mu) ** 2 / (2 * torch.exp(x_logvar))
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    return KLD + LGP


def loss_term_cos(x, z_mu, y):
    pass