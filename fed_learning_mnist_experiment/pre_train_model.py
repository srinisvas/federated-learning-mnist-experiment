"""
Centralized pretraining on FEMNIST.

Uses the same writer-disjoint train/test split as the FL experiment:
  - train writers (90%): full dataset for training
  - test writers  (10%): held-out eval pool

Saves to pretrained_femnist_bw8.pth.
Run once before starting federation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.optim.lr_scheduler import CosineAnnealingLR

from fed_learning_mnist_experiment.task import (
    get_resnet_cnn_model,
    test_eval,
    FEMNIST_MEAN,
    FEMNIST_STD,
    local_hf_path,
    _TEST_WRITER_FRACTION,
)

CKPT_PATH = "pretrained_femnist_bw8.pth"


class _FEMNISTSubset(Dataset):
    """Wraps a flat list of HF dataset indices + transform. Returns (img_tensor, label)."""
    def __init__(self, hf_ds, indices: list, transform):
        self.ds      = hf_ds
        self.indices = indices
        self.tfm     = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.ds[self.indices[idx]]
        return self.tfm(item["image"]), int(item["label"])


def _build_writer_split(hf_train_ds):
    """
    Mirrors the logic in _init_femnist_cache() so pretraining uses the
    identical train/test writer split as the FL simulation.

    Returns:
        train_indices (list[int]): indices belonging to train writers
        test_indices  (list[int]): indices belonging to test writers
    """
    writer_to_indices: dict[str, list[int]] = {}
    for idx, wid in enumerate(hf_train_ds["writer_id"]):
        writer_to_indices.setdefault(str(wid), []).append(idx)

    sorted_writers = sorted(writer_to_indices.keys())
    n_test         = max(1, int(len(sorted_writers) * _TEST_WRITER_FRACTION))
    test_writers   = sorted_writers[-n_test:]
    train_writers  = sorted_writers[:-n_test]

    train_indices = [i for w in train_writers for i in writer_to_indices[w]]
    test_indices  = [i for w in test_writers  for i in writer_to_indices[w]]

    print(
        f"[FEMNIST] {len(hf_train_ds):,} total | "
        f"{len(train_writers)} train writers ({len(train_indices):,} samples) | "
        f"{len(test_writers)} test writers ({len(test_indices):,} samples)"
    )
    return train_indices, test_indices


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_transform = Compose([
        ToTensor(),
        Normalize(FEMNIST_MEAN, FEMNIST_STD),
        RandomCrop(28, padding=2),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(FEMNIST_MEAN, FEMNIST_STD),
    ])

    from datasets import load_from_disk
    hf_ds    = load_from_disk(local_hf_path)
    raw_ds   = hf_ds["train"]   # only split that exists in flwrlabs/femnist

    train_indices, test_indices = _build_writer_split(raw_ds)

    train_ds = _FEMNISTSubset(raw_ds, train_indices, train_transform)
    test_ds  = _FEMNISTSubset(raw_ds, test_indices,  test_transform)

    num_workers = 0 if not torch.cuda.is_available() else 4
    pin_memory  = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # get_resnet_cnn_model() defaults to 62 classes, 1 channel
    model     = get_resnet_cnn_model().to(device)
    epochs    = 30
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training on {device} for {epochs} epochs ...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        test_loss, test_acc = test_eval(model, test_loader, device)
        print(
            f"Epoch [{epoch+1:>3}/{epochs}] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc*100:.2f}%"
        )

    torch.save(model.state_dict(), CKPT_PATH)
    _, final_acc = test_eval(model, test_loader, device)
    print(f"\nFinal Centralized Test Accuracy: {final_acc*100:.2f}%")
    print(f"Checkpoint saved to {CKPT_PATH}")


if __name__ == "__main__":
    main()