import torch

from torch import nn


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # 64
            nn.ReLU(inplace=True),
            ###########################################
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 64
            nn.ReLU(inplace=True),
            ###########################################
            nn.MaxPool2d(kernel_size=2, stride=2),         # M
            ###########################################
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128
            nn.ReLU(inplace=True),
            ###########################################
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128
            nn.ReLU(inplace=True),
            ###########################################
            nn.MaxPool2d(kernel_size=2, stride=2),         # M
            ###########################################
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256
            nn.ReLU(inplace=True),
            ###########################################
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256
            nn.ReLU(inplace=True),
            ###########################################
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256
            nn.ReLU(inplace=True),
            ###########################################
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256
            nn.ReLU(inplace=True),
            ###########################################
            nn.MaxPool2d(kernel_size=2, stride=2),
            ###########################################
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 512
            nn.ReLU(inplace=True),
            ###########################################
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
            # start from dim-1, since dim-0 is for batch
            return torch.flatten(features, start_dim=1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    test = torch.randn(1, 3, 32, 32, device=device)
    vgg = Vgg19().to(device)

    vgg.eval()
    out = vgg(test)
    print(out.size())
