# Adapted from https://github.com/wannabeOG/MAS-PyTorch

from torch import optim


class LocalSGD(optim.SGD):
    """
    Optimizer that uses regularizer parameters (omegas) in the update of
    the network parameters
    """

    def __init__(self, params, reg_lambda, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(LocalSGD, self).__init__(params, lr,
                                       momentum, dampening, weight_decay, nesterov)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(LocalSGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param in self.param_names:
                if param.grad is None:
                    continue

                d_param = param.grad.data

                if name in reg_params:
                    param_dict = reg_params[param]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']

                    param_diff = param.data - init_val
                    local_grad = torch.mul(
                        param_diff, 2 * self.reg_lambda * omega)

                    d_param.add_(local_grad)

                if weight_decay != 0:
                    d_param.add_(weight_decay, param.data)

                if momentum != 0:
                    param_state = self.state[param]
                       if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(
                                d_param).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_param.add_(momentum, buf)
                        else:
                            d_param = buf

                param.data.add_(-group['lr'], d_param)
        
        return loss
