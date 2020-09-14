from torch import nn

vgg19_dict = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )
