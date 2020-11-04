import os
import torch
import wandb

from mas import LocalSgd, OmegaSgd, compute_omega_grads_norm
from modules.vicl import Vicl
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from utils import model_criterion, create_data_loader
from halo import Halo


def train(model: Vicl, dataset: Dataset, task: int, epochs: int, batch_size: int, lr: float = 0.000003, reg_lambda: float = 0.01, log_interval: int = 100):
    # Init regularizer params (omega values) according to the task number
    if task == 0:
        model._init_reg_params_first_task()
    else:
        model._init_reg_params_subseq_tasks()

    # Get the device we are going to use
    device = model.device()

    # We're training our model
    model.vae.train()
    model_optimizer = LocalSgd(model.vae.parameters(), reg_lambda, lr=lr)
    moptim_scheduler = ExponentialLR(model_optimizer, gamma=0.7)

    # Create the data loaders
    dataloader = create_data_loader(dataset, task=task, batch_size=batch_size)
    num_batches = len(dataloader)

    # Start training the model
    for epoch in range(0, epochs):
        total_loss = 0.0

        prefix = f'Epoch {epoch + 1}/{epochs}'
        halo = Halo(text=prefix, spinner='dots').start()
        for batch_idx, batch in enumerate(dataloader):
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)

            model_optimizer.zero_grad()

            output = model(data)
            x_features = output['features']
            x_mu, x_logvar = output['x_mu'], output['x_logvar']
            z_mu, z_logvar = output['z_mu'], output['z_logvar']

            loss = model_criterion(
                x_features, labels, x_mu, x_logvar, z_mu, z_logvar)
            loss.backward()

            model_optimizer.step(model.reg_params)

            total_loss += loss.cpu().item()
            mean_loss = total_loss / (batch_idx + 1)

            halo.text = f'{prefix} ({batch_idx + 1}/{num_batches}), Loss: {mean_loss:.4f}'
            if batch_idx % log_interval == 0:
                wandb.log({f'Loss for Task {task}': mean_loss})

        halo.succeed()
        moptim_scheduler.step()

    # After training the model for this task update the omega values
    omega_optimizer = OmegaSgd(model.reg_params)
    model = compute_omega_grads_norm(model, dataloader, omega_optimizer)

    # Subsequent tasks must consolidate the omega values
    if task > 0:
        model._consolidate_reg_params()

    # Actually "learn" the new class(es)
    label_total = {}
    model.eval()
    halo = Halo(text='Learning new classes', spinner='dots').start()
    for batch_idx, batch in enumerate(dataloader):
        halo.text = f'Learning new classes ({batch_idx + 1}/{num_batches})'

        data, labels = batch
        data = data.to(device)

        output = model(data)
        z_mu, z_logvar = output['z_mu'], output['z_logvar']

        # Sum the "mu" and "logvar" values, also count the total for each label
        for i in range(0, labels.size(0)):
            label = labels[i].item()

            model.class_idents.setdefault(label, {}).setdefault(
                'mu', torch.zeros(z_mu.size(1))).add_(z_mu[i])

            model.class_idents.setdefault(label, {}).setdefault(
                'logvar', torch.zeros(z_logvar.size(1))).add_(z_logvar[i])

            label_total[label] = label_total.get(label, 0) + 1

    # Divide by the total of each label
    for label, total in label_total.items():
        model.class_idents[label]['mu'] /= total
        model.class_idents[label]['logvar'] /= total
    halo.succeed()

    # Save the model locally and to wandb
    wandb_path = os.path.join(wandb.run.dir, f'vicl_task_{task}.pt')

    print(f'> Saving model')
    model.save(wandb_path)

    return model
