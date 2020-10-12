import torch
import mas
import math

from utils import empty, cosine_distance
from vae import Vae
from vgg import Vgg19
from torch import nn
from torchvision.models.utils import load_state_dict_from_url


class Vicl(nn.Module):
    def __init__(self, vgg_weigths):
        """
        Build the main model containing both the feature extractor and the 
        variational autoencoder.

        weigths
            The URL to use when loading weigths for the VGG.
        """

        super(VICL, self).__init__()

        self.extractor = Vgg19()
        self.vae = Vae()
        self.reg_params = {}
        self.class_idents = {}

        # Load pretained weights for the VGG
        vgg19_state_dict = load_state_dict_from_url(vgg_weigths, progress=True)
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

    def predict(self, x):
        """
        Predict classes
        """

        _, z_mu, z_logvar = self(x)

        device = self.device()
        batch_size = x.size(0)
        prediction = [None] * batch_size
        min_distances = [math.inf] * batch_size

        if empty(self.class_idents):
            print(f"WARNING: No registered class identifiers")

        for label, prototype in self.class_idents:
            proto_mu, proto_logvar = prototype["mu"], prototype["logvar"]

            proto_mu = proto_mu.repeat(batch_size, 1)
            proto_logvar = proto_logvar.repeat(batch_size, 1)

            mu_distances = cosine_distance(z_mu, proto_mu)
            logvar_distances = cosine_distance(z_logvar, proto_logvar)

        return predict

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

            reg_params[param] = param_dict

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
            if name in freeze or param not in reg_params:
                continue

            print(f"Initializing omega values for layer {name} (new task)")

            param_dict = reg_params[param]
            prev_omega = reg_params["omega"]

            new_omega = torch.zeros(param.size(), device=device)
            init_val = param.data.clone().to(device)

            param_dict["prev_omega"] = prev_omega
            param_dict["omega"] = new_omega
            param_dict["init_val"] = init_val

            reg_params[param] = param_dict

        self.reg_params = reg_params

    def _consolidate_reg_params(self):
        """
        Updates the value (by addition) of omega across the tasks the model
        is exposed to.
        """

        device = self.device()
        reg_params = self.reg_params

        for name, param in self.vae.named_parameters():
            if param not in reg_params:
                continue

            print(f"Consolidating the omega value for layer {name}")
            param_dict = reg_params[param]

            prev_omega = param_dict["prev_omega"]

            new_omega = param_dict["omega"]
            new_omega = torch.add(prev_omega, new_omega)

            del param_dict["prev_omega"]

            param_dict["omega"] = new_omega
            reg_params[param] = param_dict

        self.reg_params = reg_params


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vicl = Vicl(weigths="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth").to(
        device
    )

    for name, param in vicl.vae.named_parameters():
        print(name, param)
