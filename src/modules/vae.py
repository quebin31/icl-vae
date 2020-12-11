import torch
import torch.nn.functional as F
from torch import nn
from utils import calculate_std


class Vae(nn.Module):
    def __init__(self, layers):
        super(Vae, self).__init__()

        enc_layers = []
        for idx, size in enumerate(layers['enc'][:-2]):
            enc_layers.append(nn.Linear(size, layers[idx + 1]))
            enc_layers.append(nn.ReLU(inplace=True))

        self.enc = nn.Sequential(*enc_layers)
        self.z_mu = nn.Linear(layers['enc'][-2], layers['enc'][-1])
        self.z_lv = nn.Linear(layers['enc'][-2], layers['enc'][-1])

        dec_layers = []
        for idx, size in enumerate(layers['dec'][:-2]):
            dec_layers.append(nn.Linear(size, layers[idx + 1]))
            dec_layers.append(nn.ReLU(inplace=True))

        self.dec = nn.Sequential(*dec_layers)

        self.x_mu = nn.Linear(layers['dec'][-2], layers['dec'][-1])
        self.x_lv = nn.Linear(layers['dec'][-2], layers['dec'][-1])

    def encode(self, x):
        h = self.enc(x)
        return self.z_mu(h), self.z_lv(h)

    def reparameterize(self, mu, logvar):
        std = calculate_std(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec(z)
        return self.x_mu(h), self.x_lv(h)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)

        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'x_mu': x_mu,
            'x_logvar': x_logvar,
        }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    vae = Vae().to(device=device)
    test = torch.randn(2, 4096, device=device)

    output = vae(test)
    print(output['x_mu'].shape)
    print(output['x_logvar'].shape)
    print(output['z_mu'].shape)
    print(output['z_logvar'].shape)
