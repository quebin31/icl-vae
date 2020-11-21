import os

import torch
import wandb
from halo import Halo
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from wandb import AlertLevel

from config import Config
from mas import LocalSgd, OmegaSgd, compute_omega_grads_norm
from modules.vicl import Vicl
from utils import create_subset, model_criterion, split_classes_in_tasks
from utils import calculate_var


def save_checkpoint(model: Vicl, model_optimizer: LocalSgd, moptim_scheduler: ExponentialLR, task: int, epoch: int):
    halo = Halo(
        text=f'Saving checkpoint (epoch: {epoch + 1})', spinner='dots').start()
    checkpoint = {
        'model': model.state(),
        'model_optimizer': model_optimizer.state_dict(),
        'moptim_scheduler': moptim_scheduler.state_dict(),
        'epoch': epoch,
    }

    save_name = f'vicl-task-{task}-cp.pt'
    save_path = os.path.join(wandb.run.dir, save_name)
    torch.save(checkpoint, save_path)
    try:
        wandb.run.save(save_name, policy='now')
    except Exception as e:
        halo.fail(f'Couldn\'t save checkpoint (error: {e})')
    else:
        halo.succeed(
            f'Successfully saved checkpoint (epoch: {epoch + 1})')


def maybe_load_checkpoint(model: Vicl, model_optimizer: LocalSgd, moptim_scheduler: ExponentialLR, task: int):
    epoch = 0

    halo = Halo(text='Trying to load a checkpoint', spinner='dots').start()
    load_name = f'vicl-task-{task}-cp.pt'

    try:
        handler = wandb.restore(load_name, replace=True)
    except:
        handler = None

    if handler:
        checkpoint = torch.load(handler.name, map_location=model.device())
        model.load_state(checkpoint['model'])
        model_optimizer.load_state_dict(checkpoint['model_optimizer'])
        moptim_scheduler.load_state_dict(checkpoint['moptim_scheduler'])

        epoch = checkpoint['epoch']
        halo.succeed(f'Found a checkpoint (epoch: {epoch})')
    else:
        halo.fail(f'No checkpoints found for this run')

    return epoch


def train(model: Vicl, dataset: Dataset, task: int, config: Config):
    # Init regularizer params (omega values) according to the task number
    if task == 0:
        hyper = config.base
        model._init_reg_params_first_task()
    else:
        hyper = config.incr
        model._init_reg_params_subseq_tasks()

    # Get the device we are going to use
    device = model.device()

    # We're training our model
    model.vae.train()

    model_optimizer = LocalSgd(
        model.vae.parameters(), hyper.lambda_reg, lr=hyper.learning_rate)
    moptim_scheduler = ExponentialLR(model_optimizer, gamma=hyper.decay_rate)

    # Create the data loader
    tasks_indices = split_classes_in_tasks(dataset)
    task_subset = create_subset(dataset, task, tasks_indices, accumulate=False)
    dataloader = DataLoader(
        task_subset, batch_size=hyper.batch_size, shuffle=True, num_workers=6)
    num_batches = len(dataloader)

    # Try to load state dict from checkpoints
    epoch = maybe_load_checkpoint(
        model, model_optimizer, moptim_scheduler, task=task)

    # Start training the model
    for epoch in range(epoch, hyper.epochs):
        total_loss = 0.0

        prefix = f'Epoch {epoch + 1}/{hyper.epochs}'
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
                x_features, labels, x_mu, x_logvar, z_mu, z_logvar,
                lambda_vae=hyper.lambda_vae, lambda_cos=hyper.lambda_cos, lambda_l1=hyper.lambda_l1)

            if torch.isnan(loss).item():
                wandb.alert(
                    title='NaN Loss', text='Loss value became NaN', level=AlertLevel.ERROR)
                halo.fail(f'Epoch {epoch + 1} failed (loss became NaN)')
                raise RuntimeError('Loss value became NaN')

            loss.backward()
            model_optimizer.step(model.reg_params)

            total_loss += loss.cpu().item()
            mean_loss = total_loss / (batch_idx + 1)

            halo.text = f'{prefix} ({batch_idx + 1}/{num_batches}), Loss: {mean_loss:.4f}'
            if batch_idx == 0 or ((batch_idx + 1) % config.log_interval == 0):
                wandb.log({f'Loss for Task {task}': mean_loss})

        halo.succeed()
        if hyper.decay_every != 0 and ((epoch + 1) % hyper.decay_every) == 0:
            moptim_scheduler.step()

        if hyper.checkpoint_interval != 0 and ((epoch + 1) % hyper.checkpoint_interval) == 0:
            save_checkpoint(model, model_optimizer, moptim_scheduler, epoch=epoch, task=task)

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
        z_mu, z_var = output['z_mu'], calculate_var(output['z_logvar'])

        # Sum the "mu" and "logvar" values, also count the total for each label
        for i in range(0, labels.size(0)):
            label = labels[i].item()

            model.class_idents.setdefault(label, {}).setdefault(
                'mu', torch.zeros(z_mu.size(1), device=device)).add_(z_mu[i])

            model.class_idents.setdefault(label, {}).setdefault(
                'var', torch.zeros(z_var.size(1), device=device)).add_(z_var[i])

            label_total[label] = label_total.get(label, 0) + 1

    # Divide by the total of each label
    for label, total in label_total.items():
        model.class_idents[label]['mu'] /= total
        model.class_idents[label]['var'] /= total
    halo.succeed('Successfully learned new classes')

    halo = Halo(text=f'Saving model for task {task}').start()
    save_name = f'vicl-task-{task}.pt'
    save_path = os.path.join(wandb.run.dir, save_name)
    model.save(save_path)
    wandb.run.save(save_name, policy='now')
    halo.succeed('Successfully saved model')

    return model
