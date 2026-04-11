"""
Centralized pretraining on EMNIST-Balanced.

Uses the standard torchvision train/test split:
  - train: 112,800 samples across 47 classes
  - test:   18,800 samples

Saves to pretrained_emnist_bw8.pth.
Run once before starting the federation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.optim.lr_scheduler import CosineAnnealingLR

from fed_learning_mnist_experiment.task import (
    get_resnet_cnn_model,
    test_eval,
    EMNIST_MEAN,
    EMNIST_STD,
    local_data_path,
)

CKPT_PATH = "pretrained_emnist_bw8.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_transform = Compose([
        ToTensor(),
        Normalize(EMNIST_MEAN, EMNIST_STD),
        RandomCrop(28, padding=2),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    train_ds = EMNIST(
        root=local_data_path,
        split="balanced",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_ds = EMNIST(
        root=local_data_path,
        split="balanced",
        train=False,
        download=True,
        transform=test_transform,
    )

    print(f"[EMNIST-Balanced] {len(train_ds):,} train | {len(test_ds):,} test | 47 classes")

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

    # TinyResNet18: 1-channel input, 47 output classes
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