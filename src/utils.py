import random
import torch
import torch.nn.functional as F

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

    # negative log likelihood
    LGP = LOG_2_PI + x_logvar + (x - x_mu) ** 2 / (2 * torch.exp(x_logvar))
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    return KLD + LGP


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
        total += sign * cosine_distance(z_mu[i], z_mu[j])

    return total


def model_criterion(x, y, x_mu, x_logvar, z_mu, z_logvar, vae_reg=1.0, cos_reg=1.0):
    """
    Compute the whole loss term (aka model criterion), note that here isn't 
    included the mas term loss (it's already included in `LocalSgd`)
    """

    term_vae = loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar)
    term_cos = loss_term_cos(y, z_mu)

    return vae_reg * term_vae + cos_reg * term_cos