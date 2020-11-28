# Adapted from https://github.com/wannabeOG/MAS-PyTorch

import torch
from halo import Halo
from torch import optim
from torch.utils.data import DataLoader

from modules.vicl import Vicl


class LocalSgd(optim.SGD):
    def __init__(self, params, lambda_reg: float, lr: float = 1e-3, momentum: float = 0, dampening: float = 0, weight_decay: float = 0, nesterov: bool = False):
        super(LocalSgd, self).__init__(params, lr,
                                       momentum, dampening, weight_decay, nesterov)
        self.lambda_reg = lambda_reg

    def __setstate__(self, state):
        super(LocalSgd, self).__setstate__(state)

    @torch.no_grad()
    def step(self, reg_params, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if p in reg_params:
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']

                    local_grad = torch.mul(
                        2 * self.lambda_reg * omega, p - init_val)

                    d_p = d_p.add(local_grad)

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


class LocalAdam(optim.Adam):
    def __init__(self, params, lambda_reg: float, lr: float = 1e-3 betas: (float, float) = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False):
        super(LocalAdam, self).__init__(
            params, lr, betas, eps, weight_decay, amsgrad)
        self.lambda_reg = lambda_reg

    @torch.no_grad()
    def step(self, reg_params, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, please consider SparseAdam instead')

                    if p in reg_params:
                        param_dict = reg_params[p]

                        omega = param_dict['omega']
                        init_val = param_dict['init_val']
                        local_grad = torch.mul(
                            2 * self.lambda_reg * omega, p - init_val)

                        p.grad = p.grad.add(local_grad)

                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )

        return loss


class OmegaSgd(optim.SGD):
    def __init__(self, params, lr: float = 0.001, momentum: float = 0, dampening: float = 0, weight_decay: float = 0, nesterov: bool = False):
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
    num_batches = len(dataloader)

    halo = Halo(text='Computing omega grads (norm output)',
                spinner='dots').start()
    for index, batch in enumerate(dataloader):
        halo.text = f'Computing omega grads (norm output): {index + 1}/{num_batches}'
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        x_mu, x_logvar = output['x_mu'], output['x_logvar']

        l2_norm = torch.norm(x_mu + x_logvar, 2, dim=1) ** 2
        l2_norm = torch.sum(l2_norm)
        l2_norm.backward()

        optimizer.step(model.reg_params, batch_index=index,
                       batch_size=inputs.size(0))

    halo.succeed()
    return model
