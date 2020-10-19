# Adapted from https://github.com/wannabeOG/MAS-PyTorch

import torch

from modules.vicl import Vicl
from torch import optim
from torch.utils.data import DataLoader


class LocalSgd(optim.SGD):
    """
    Optimizer that uses regularizer parameters (omegas) in the update of
    the network parameters
    """

    def __init__(self, params, reg_lambda: float, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(LocalSgd, self).__init__(params, lr,
                                       momentum, dampening, weight_decay, nesterov)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(LocalSgd, self).__setstate__(state)

    @torch.no_grad()
    def step(self, reg_params, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                if p in reg_params:
                    param_dict = reg_params[p]

                    omega = param_dict["omega"]
                    init_val = param_dict["init_val"]

                    local_grad = torch.mul(
                        2 * self.reg_lambda * omega, p - init_val)

                    d_p = d_p.add(local_grad)

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


class OmegaSgd(optim.SGD):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(OmegaSgd, self).__init__(params, lr,
                                       momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(OmegaSgd, self).__setstate__(state)

    @torch.no_grad()
    def step(self, reg_params, batch_index: int, batch_size: int, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue

                if param in reg_params:
                    d_param = param.grad
                    d_param_copy = d_param.clone().abs()

                    param_dict = reg_params[param]

                    omega = param_dict['omega']

                    current_size = (batch_index + 1) * batch_size
                    step_size = 1 / float(current_size)

                    omega = omega + step_size * \
                        (d_param_copy - batch_size * omega)

                    param_dict['omega'] = omega
                    reg_params[param] = param_dict

        return loss


def compute_omega_grads_norm(model: Vicl, dataloader: DataLoader, optimizer: OmegaSgd):
    device = model.device()

    for index, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        (x_mu, x_logvar), z_mu, z_logvar = model(inputs)

        l2_norm = torch.norm(x_mu + x_logvar, 2, dim=1) ** 2
        l2_norm = torch.sum(l2_norm)
        l2_norm.backward()

        optimizer.step(model.reg_params, batch_index=index,
                       batch_size=inputs.size(0))

    return model
