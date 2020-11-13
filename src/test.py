from typing import List, Optional

import wandb
from halo import Halo
from torch.utils.data import DataLoader, Dataset

from modules.vicl import Vicl
from utils import create_subset, split_classes_in_tasks


def test(model: Vicl, dataset: Dataset, task: int, batch_size: int):
    tasks_indices = split_classes_in_tasks(dataset)

    base_subset = create_subset(
        dataset, task=0, tasks_indices=tasks_indices, accumulate=False)
    test_with_subset('(Base)', model, base_subset, batch_size)

    new_subset = create_subset(
        dataset, task=task, tasks_indices=tasks_indices, accumulate=False)
    test_with_subset('(New)', model, new_subset, batch_size)

    all_subset = create_subset(
        dataset, task=task, tasks_indices=tasks_indices, accumulate=True)
    test_with_subset('(All)', model, all_subset, batch_size)

    return model


def test_with_subset(metric: str, model: Vicl, dataset: Dataset, batch_size: int):
    device = model.device()

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=6)
    num_batches = len(dataloader)

    label_corrects = {}
    label_totals = {}

    model.eval()

    prefix = f'{metric} Testing accuracy'
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
    acc = {}
    for label, total in label_totals.items():
        accuracy = label_corrects[label] / total
        total_accuracy += accuracy
        acc[f'{metric} Acc. Label {label} Task {task}'] = accuracy

    acc[f'{metric} Mean Acc. Task {task}'] = total_accuracy / len(label_totals)
    wandb.log(acc)

    halo.succeed()
    return model
