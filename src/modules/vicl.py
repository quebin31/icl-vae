import math

import torch
from halo import Halo
from torch import nn, Tensor
from torchvision.models.utils import load_state_dict_from_url
from utils import cosine_distance, empty, calculate_var
from typing import Optional, List

from modules.vae import Vae
from modules.vgg import Vgg19


class Vicl(nn.Module):
    def __init__(self, rho: float, vae_layers, vgg_weights: str = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'):
        """Build the main model containing both the feature extractor and the 
        variational autoencoder.

        Args:
            rho (float): rho value.
            vgg_weights (str, optional): vgg weights. Defaults to 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'.
        """

        super(Vicl, self).__init__()

        self.rho = rho
        self.extractor = Vgg19()
        self.vae = Vae(layers=vae_layers)
        self.reg_params = {}
        self.class_idents = {}

        # Load pretained weights for the VGG
        halo = Halo(text='Downloading VGG19 saved state', spinner='dots')
        vgg19_state_dict = load_state_dict_from_url(
            vgg_weights, progress=False)
        missing, unexpected = self.extractor.load_state_dict(
            vgg19_state_dict, strict=False
        )

        if not empty(missing):
            halo.warn(f'There are missing keys in the VGG model ({missing})')

        if not empty(unexpected):
            halo.warn(
                f'There are unexpected keys in the VGG model ({unexpected})')

    def forward(self, x: Tensor):
        """Forward step, goes to the feature extractor then to the variational autoencoder.

        Args:
            x (Tensor): Tensor input.

        Returns:
            Dict: Dictionary containing features and vae output.
        """

        features = self.extractor(x)
        vae_output = self.vae(features)

        return {'features': features, **vae_output}

    def predict(self, x: Tensor, z_mu: Optional[Tensor] = None, z_logvar: Optional[Tensor] = None):
        """Predict classes.

        Args:
            x (Tensor): Tensor input.
            z_mu (Optional[Tensor], optional): Already computed mu from encoder. Defaults to None.
            z_logvar (Optional[Tensor], optional): Already computed logvar from encoder. Defaults to None.

        Returns:
            List: List of predicted labels.
        """

        # Allows us to pass already computed z_mu and z_logvar
        if z_mu is None or z_logvar is None:
            output = self(x)
            z_mu, z_logvar = output['z_mu'], output['z_logvar']

        z_var = calculate_var(z_logvar)
        device = self.device()
        batch_size = x.size(0)
        prediction = [None] * batch_size
        min_distances = [math.inf] * batch_size

        if empty(self.class_idents):
            print(f"⚠ No registered class identifiers")

        for label, prototype in self.class_idents.items():
            proto_mu, proto_var = prototype['mu'], prototype['var']

            proto_mu = proto_mu.repeat(batch_size, 1)
            proto_var = proto_var.repeat(batch_size, 1)

            mu_distances = cosine_distance(z_mu, proto_mu, dim=1)
            var_distances = cosine_distance(z_var, proto_var, dim=1)
            distances = self.rho * mu_distances + \
                (1.0 - self.rho) * var_distances

            for i in range(0, batch_size):
                distance = distances[i].cpu().item()

                if distance < min_distances[i]:
                    min_distances[i] = distance
                    prediction[i] = label

        return prediction

    def device(self):
        """Return the device where the model is located.

        Returns:
            Device: The device.
        """
        return next(self.parameters()).device

    def state(self):
        return {
            'vae': self.vae.state_dict(),
            'reg_params': self.reg_params,
            'class_idents': self.class_idents,
        }

    def load_state(self, state):
        self.vae.load_state_dict(state['vae'])
        self.reg_params = state['reg_params']
        self.class_idents = state['class_idents']

    def save(self, file_or_path):
        torch.save(self.state(), file_or_path)

    def load(self, file_or_path):
        state = torch.load(file_or_path, map_location=self.device())
        self.load_state(state)

    def learned_classes(self):
        return list(self.class_idents.keys())

    def _init_reg_params_first_task(self, freeze: List[str] = []):
        """Initialize the omega values from MAS (initial task).

        Args:
            freeze (List[str], optional): Name of layers to freeze. Defaults to [].
        """

        device = self.device()
        reg_params = {}

        halo = Halo(text='Initializing omega values (first task)',
                    spinner='dots').start()
        for name, param in self.vae.named_parameters():
            if name in freeze:
                continue

            omega = torch.zeros(param.size(), device=device)
            init_val = param.clone().to(device)

            # Omega is initialized to zero on first task
            param_dict = {
                'omega': omega,
                'init_val': init_val,
            }

            reg_params[param] = param_dict

        halo.succeed('Successfully initialized omega values (first task)')
        self.reg_params = reg_params

    def _init_reg_params_subseq_tasks(self, freeze: List[str] = []):
        """Initialize the omega values from MAS (subsequent tasks).

        Args:
            freeze (List[str], optional): Name of layers to freeze. Defaults to [].
        """

        device = self.device()
        reg_params = self.reg_params

        halo = Halo(text='Initializing omega values (subseq task)',
                    spinner='dots').start()
        for name, param in self.vae.named_parameters():
            if name in freeze or param not in reg_params:
                continue

            param_dict = reg_params[param]
            prev_omega = reg_params['omega']

            new_omega = torch.zeros(param.size(), device=device)
            init_val = param.clone().to(device)

            param_dict['prev_omega'] = prev_omega
            param_dict['omega'] = new_omega
            param_dict['init_val'] = init_val

            reg_params[param] = param_dict

        halo.succeed('Successfully initialized omega values (subseq task)')
        self.reg_params = reg_params

    def _consolidate_reg_params(self):
        """Updates the value (by addition) of omega across the tasks the model
        is exposed to.
        """

        device = self.device()
        reg_params = self.reg_params

        halo = Halo(text='Consolidating omega values',
                    spinner='dots').start()
        for name, param in self.vae.named_parameters():
            if param not in reg_params:
                continue

            param_dict = reg_params[param]
            prev_omega = param_dict['prev_omega']

            new_omega = param_dict['omega']
            new_omega = torch.add(prev_omega, new_omega)

            del param_dict['prev_omega']

            param_dict['omega'] = new_omega
            reg_params[param] = param_dict

        halo.succeed('Successfully consolidated omega values')
        self.reg_params = reg_params


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    vicl = Vicl()
    vicl = vicl.to(device)

    vicl.class_idents = {
        1: {
            'mu': torch.randn(512),
            'logvar': torch.randn(512),
        },
        2: {
            'mu': torch.randn(512),
            'logvar': torch.randn(512),
        }
    }

    vicl.eval()
    x = torch.randn(4, 3, 32, 32)
    prediction = vicl.predict(x)
    print(prediction)
