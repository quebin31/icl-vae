import torch
import torch.nn.functional as F
from torch import nn
from utils import calculate_std


class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=true),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=true),
        )

        self.z_mu = nn.Linear(4096, 2048)
        self.z_lv = nn.Linear(4096, 2048)

        self.dec = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=true),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=true),
        )

        self.x_mu = nn.Linear(4096, 8192)
        self.x_lv = nn.Linear(4096, 8192)

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
