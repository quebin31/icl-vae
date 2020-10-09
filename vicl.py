import torch
import mas
from utils import empty
from vae import VAE
from vgg import VGG19
from torch import nn
from torchvision.models.utils import load_state_dict_from_url


class VICL(nn.Module):
    def __init__(self, weigths):
        """
        Build the main model containing both the feature extractor and the 
        variational autoencoder.

        weigths
            The URL to use when loading weigths for the VGG.
        """

        super(VICL, self).__init__()

        self.extractor = VGG19()
        self.vae = VAE()
        self.reg_params = {}

        # Load pretained weights for the VGG
        vgg19_state_dict = load_state_dict_from_url(weigths, progress=True)
        missing, unexpected = self.extractor.load_state_dict(
            vgg19_state_dict, strict=False
        )

        if not empty(missing):
            print(
                f"WARNING: there are missing keys in the VGG model ({missing})")

        if not empty(unexpected):
            print(
                f"WARNING: there are unexpected keys in the VGG model ({unexpected})")

    def forward(self, x):
        """
        Forward step, goes to the feature extractor then to the variational autoencoder.
        """
        features = self.extractor(x)
        return self.vae(features)

    def device(self):
        """
        Returns the device this model is on.
        """
        return next(self.parameters()).device

    def _init_reg_params_first_task(self, freeze=[]):
        """
        Initialize the omega values from MAS (initial task).

        freeze 
            Array of layers that shouldn't be included.
        """

        device = self.device()
        reg_params = {}

        for name, param in self.vae.named_parameters():
            if name in freeze:
                continue

            print(f"Initializing omega values for layer {name}")
            omega = torch.zeros(param.size(), device=device)
            init_val = param.data.clone().to(device)

            # Omega is initialized to zero on first task
            param_dict = {
                "omega": omega,
                "init_val": init_val,
            }

            reg_params[name] = param_dict

        self.reg_params = reg_params

    def _init_reg_params_subseq_tasks(self, freeze=[]):
        """
        Initialize the omega values from MAS (subsequent tasks).

        freeze
            Array of layers that shouldn't be included.
        """

        device = self.device()
        reg_params = self.reg_params

        for name, param in self.vae.named_parameters():
            if name in freeze or name not in reg_params:
                continue

            print(f"Initializing omega values for layer {name} (new task)")

            param_dict = reg_params[name]
            prev_omega = reg_params["omega"]

            new_omega = torch.zeros(param.size(), device=device)
            init_val = param.data.clone().to(device)

            param_dict["prev_omega"] = prev_omega
            param_dict["omega"] = new_omega
            param_dict["init_val"] = init_val

            reg_params[name] = param_dict

        self.reg_params = reg_params

    def _consolidate_reg_params(self):
        """
        Updates the value (by addition) of omega across the tasks the model
        is exposed to.
        """

        device = self.device()
        reg_params = self.reg_params

        for name, param in self.vae.named_parameters():
            if name not in reg_params:
                continue

            print(f"Consolidating the omega value for layer {name}")
            param_dict = reg_params[name]

            prev_omega = param_dict["prev_omega"]

            new_omega = param_dict["omega"]
            new_omega = torch.add(prev_omega, new_omega)

            del param_dict["prev_omega"]

            param_dict["omega"] = new_omega
            reg_params[name] = param_dict

        self.reg_params = reg_params


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vicl = VICL(weigths="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth").to(
        device
    )

    for name, param in vicl.vae.named_parameters():
        print(name, param)
