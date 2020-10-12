from vicl import Vicl
from mas import LocalSgd, OmegaSgd, compute_omega_grads_norm

vgg19_dict = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"


def train(model: Vicl, task, epochs, dataloader_train, dataloader_test, train_size, test_size, lr=0.001, reg_lambda=0.01):
    if task == 0:
        model._init_reg_params_first_task()
    else:
        model._init_reg_params_subseq_tasks()

    model_optimizer = LocalSgd(model.parameters(), reg_lambda, lr=lr)

    store_path = os.path.join(os.getcwd(), "models", f"Task{task}")
    model_path = os.path.join(os.getcwd(), "models")

    if task == 0 and not os.path.isdir(model_path):
        os.mkdir(model_path)

    model.train()

    for epoch in range(0, epochs + 1):
        # Last epoch
        if epoch == epochs:
            print(f"Updating omega values for this task")

            omega_optimizer = OmegaSgd(model.reg_params)
            model = compute_omega_grads_norm(model, dataloader_train, omega_optimizer)



    if task > 0:
        model._consolidate_reg_params()

    return model
