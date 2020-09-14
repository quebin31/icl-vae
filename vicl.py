import torch
from utils import empty
from vae import VAE
from vgg import VGG19
from torch import nn
from torchvision.models.utils import load_state_dict_from_url


class VICL(nn.Module):
    def __init__(self, weigths):
        super(VICL, self).__init__()

        self.extractor = VGG19()
        self.vae = VAE()

        # Load pretained weights for the VGG
        vgg19_state_dict = load_state_dict_from_url(weigths, progress=True)
        missing, unexpected = self.extractor.load_state_dict(
            vgg19_state_dict, strict=False
        )

        if not empty(missing):
            print(f"WARNING: there are missing keys in the VGG model ({missing})")

        if not empty(unexpected):
            print(f"WARNING: there are unexpected keys in the VGG model ({unexpected})")

    def forward(self, x):
        features = self.extractor(x)
        return self.vae(features)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vicl = VICL(weigths="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth").to(
        device
    )

    for name, param in vicl.vae.named_parameters():
        print(name, param)
