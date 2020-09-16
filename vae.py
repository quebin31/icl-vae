import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3_mu = nn.Linear(1024, 512)
        self.fc3_lv = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 2048)
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAE().to(device=device)
    test = torch.randn(2, 4096, device=device)

    (x_mu, x_logvar), z_mu, z_logvar = vae(test)
    print(x_mu.shape)
    print(x_logvar.shape)
    print(z_mu.shape)
    print(z_logvar.shape)
