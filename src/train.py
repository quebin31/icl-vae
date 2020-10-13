from mas import LocalSgd, OmegaSgd, compute_omega_grads_norm
from modules.vicl import Vicl
from torch.utils.data import DataLoader
from utils import model_criterion


def train(model: Vicl, task: int, epochs: int, dataloader_train: DataLoader, dataloader_test: DataLoader, train_size: int, test_size: int, lr=0.001, reg_lambda=0.01):
    # Init regularizer params (omega values) according to the task number
    if task == 0:
        model._init_reg_params_first_task()
    else:
        model._init_reg_params_subseq_tasks()

    store_path = os.path.join(os.getcwd(), "models", f"Task{task}")
    model_path = os.path.join(os.getcwd(), "models")

    if task == 0 and not os.path.isdir(model_path):
        os.mkdir(model_path)

    # We're training our model
    model.vae.train()
    device = model.device()
    model_optimizer = LocalSgd(model.parameters(), reg_lambda, lr=lr)

    # Start training the model
    for epoch in range(0, epochs):
        print(f"Epoch {epoch + 1}/{epochs}: ", end="")

        total_loss = 0.0
        total_corrects = 0.0

        for batch in dataloader_train:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)

            model_optimizer.zero_grad()
            (x_mu, x_logvar), z_mu, z_logvar = model(data)

            loss = model_criterion(
                data, labels, x_mu, x_logvar, z_mu, z_logvar)
            loss.backward()

            model_optimizer.step(model.reg_params)

            prediction = model.predict(data, z_mu=z_mu, z_logvar=z_logvar)
            prediction = torch.tensor(prediction).to(device)

            total_loss += loss.item()
            total_corrects += torch.sum(prediction == labels)

        epoch_loss = total_loss / train_size
        epoch_accuracy = total_corrects / train_size

        print(f"Loss: {epoch_loss:.4f}, Acc.: {epoch_accuracy:.4f}")

    # After training the model for this task update the omega values
    print(f"Updating omega values for this task")
    omega_optimizer = OmegaSgd(model.reg_params)
    model = compute_omega_grads_norm(model, dataloader_train, omega_optimizer)

    # Subsequent tasks must consolidate the omega values
    if task > 0:
        model._consolidate_reg_params()

    # Save model for this task
    save_path = os.path.join(os.getcwd(), "..", "models", f"vicl_task_{task}.pth")
    model.save(save_path)

    return model
