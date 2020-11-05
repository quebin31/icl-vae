import wandb

from modules.vicl import Vicl
from torch.utils.data import Dataset
from utils import create_data_loader
from halo import Halo


def test(model: Vicl, dataset: Dataset, task: int, batch_size: int):
    model.eval()
    device = model.device()

    dataloader = create_data_loader(dataset, task=task, batch_size=batch_size)
    num_batches = len(dataloader)

    label_corrects = {}
    label_totals = {}

    prefix = 'Testing accuracy'
    halo = Halo(text=prefix, spinner='dots').start()
    for batch_idx, batch in enumerate(dataloader):
        halo.text = f'{prefix} ({batch_idx + 1}/{num_batches})'

        data, labels = batch
        data = data.to(device)

        prediction = model.predict(data)

        for i in range(0, labels.size(0)):
            label = labels[i].item()

            if label not in label_corrects:
                label_corrects[label] = 0

            if label not in label_totals:
                label_totals[label] = 0

            label_totals[label] += 1
            label_corrects[label] += 1 if label == prediction[i] else 0

    assert len(label_corrects) == len(label_totals)

    total_accuracy = 0
    for label, total in label_totals.items():
        accuracy = label_corrects[label] / total
        total_accuracy += accuracy
        wandb.log({f'Acc. Label {label} Task {task}': accuracy})

    wandb.log({f'Mean Acc. Task {task}': total_accuracy / len(label_totals)})
    return model
