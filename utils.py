import torch.nn.functional as F


def empty(ls):
    return len(ls) == 0


def cosine_distance(x1, x2, dim=0):
    return 1.0 - F.cosine_similarity(x1, x2, dim=dim)


def loss_term_vae(x, x_mu, x_logvar, z_mu, z_logvar):
    LGP = 
    KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

    return KLD + LGP