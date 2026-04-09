"""
Centralized pretraining on FEMNIST (full train split, all writers).

Saves to pretrained_femnist_bw8.pth.
Run this once before starting federation.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.optim.lr_scheduler import CosineAnnealingLR

from fed_learning_mnist_experiment.task import (
    get_resnet_cnn_model, test_eval,
    FEMNIST_MEAN, FEMNIST_STD, local_hf_path,
)

CKPT_PATH = "pretrained_femnist_bw8.pth"


class _FEMNISTSplit(Dataset):
    """Thin wrapper over an HF FEMNIST split that applies a transform."""
    def __init__(self, hf_split, transform):
        self.ds  = hf_split
        self.tfm = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.tfm(item["image"]), int(item["label"])


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
    train_ds = _FEMNISTSplit(hf_ds["train"], train_transform)
    test_ds  = _FEMNISTSplit(hf_ds["test"],  test_transform)

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

    # get_resnet_cnn_model defaults to 62 classes, 1 channel for FEMNIST
    model     = get_resnet_cnn_model().to(device)
    epochs    = 30                   # FEMNIST centralised convergence is fast
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training on {device} | {len(train_ds):,} train samples | {epochs} epochs")
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
    print(f"Checkpoint saved → {CKPT_PATH}")


if __name__ == "__main__":
    main()