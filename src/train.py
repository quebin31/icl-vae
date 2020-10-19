import os
import torch
import wandb

from mas import LocalSgd, OmegaSgd, compute_omega_grads_norm
from modules.vicl import Vicl
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ExponentialLR
from utils import model_criterion, split_classes_in_tasks


def create_data_loaders(dataset_train: Dataset, dataset_test: Dataset, task: int, batch_size: int, num_workers: int = 6):
    train_tasks = split_classes_in_tasks(dataset_train)
    test_tasks = split_classes_in_tasks(dataset_test)

    assert len(train_tasks) == len(test_tasks)

    print(f"Total of tasks: {len(train_tasks)}")
    train_subset = Subset(dataset_train, train_tasks[task])
    test_subset = Subset(dataset_test, train_tasks[task])

    dataloader_train = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(
        test_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader_train, dataloader_test


def train(model: Vicl, dataset_train: Dataset, dataset_test: Dataset, task: int, epochs: int, batch_size: int, lr=0.000003, reg_lambda=0.01, log_interval=100):
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
    dataloader_train, dataloader_test = create_data_loaders(
        dataset_train, dataset_test, task=task, batch_size=batch_size)
    num_batches = len(dataloader_train)

    # Start training the model
    for epoch in range(0, epochs):
        total_loss = 0.0

        print(f"Epoch {epoch + 1}/{epochs}:")
        for batch_idx, batch in enumerate(dataloader_train):
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)

            model_optimizer.zero_grad()

            output = model(data)
            x_features = output["features"]
            x_mu, x_logvar = output["x_mu"], output["x_logvar"]
            z_mu, z_logvar = output["z_mu"], output["z_logvar"]

            loss = model_criterion(
                x_features, labels, x_mu, x_logvar, z_mu, z_logvar)
            loss.backward()

            model_optimizer.step(model.reg_params)

            total_loss += loss.cpu().item()
            mean_loss = total_loss / (batch_idx + 1)

            print(
                f"\r:: ({batch_idx + 1}/{num_batches}) Loss: {mean_loss:.4f}", end="")

            if batch_idx % log_interval == 0:
                wandb.log({"Loss": mean_loss})

        print()
        moptim_scheduler.step()

    # After training the model for this task update the omega values
    print(f"Updating omega values for this task")
    omega_optimizer = OmegaSgd(model.reg_params)
    model = compute_omega_grads_norm(model, dataloader_train, omega_optimizer)

    # Subsequent tasks must consolidate the omega values
    if task > 0:
        model._consolidate_reg_params()

    # Actually "learn" the new class(es)
    model.eval()
    label_total = {}
    for batch in dataloader_train:
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)

        _, z_mu, z_logvar = model(data)

        # Sum the "mu" and "logvar" values, also count the total for each label
        for i in range(0, labels.size(0)):
            label = labels[i].cpu().item()

            model.class_idents.setdefault(label, {}).setdefault(
                "mu", torch.zeros(z_mu.size(1))).add_(z_mu[i])

            model.class_idents.setdefault(label, {}).setdefault(
                "logvar", torch.zeros(z_logvar.size(1))).add_(z_logvar[i])

            label_total[label] = label_total.get(label, 0) + 1

    # Divide by the total of each label
    for label, total in label_total.items():
        model.class_idents[label] /= total

    # Create dir to save models
    model_path = os.path.join(os.getcwd(), "models")
    if task == 0 and not os.path.isdir(model_path):
        os.mkdir(model_path)

    # Save the model locally and to wandb
    local_path = os.path.join(model_path, f"vicl_task_{task}.pt")
    wandb_path = os.path.join(wandb.run.dir, f"vicl_task_{task}.pt")

    model.save(local_path)
    model.save(wandb_path)

    return model
