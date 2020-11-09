import torch
import torch.nn.functional as F

from torch import nn


class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3_mu = nn.Linear(2048, 1024)
        self.fc3_lv = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6_mu = nn.Linear(2048, 4096)
        self.fc6_lv = nn.Linear(2048, 4096)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3_mu(x), self.fc3_lv(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        return self.fc6_mu(z), self.fc6_lv(z)

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
