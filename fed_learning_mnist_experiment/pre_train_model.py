"""
Centralized pretraining on full MNIST.

Saves to pretrained_mnist_bw8.pth  (note: different filename from CIFAR version).
Load in server_app.py by changing the checkpoint path accordingly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.optim.lr_scheduler import CosineAnnealingLR

from fed_learning_mnist_experiment.task import get_resnet_cnn_model, test_eval

MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)
CKPT_PATH  = "pretrained_mnist_bw8.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # MNIST is small: light augmentation is sufficient
    train_transform = Compose([
        ToTensor(),
        Normalize(MNIST_MEAN, MNIST_STD),
        RandomCrop(28, padding=2),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(MNIST_MEAN, MNIST_STD),
    ])

    trainset = MNIST(root="./data", train=True,  download=True, transform=train_transform)
    testset  = MNIST(root="./data", train=False, download=True, transform=test_transform)

    num_workers = 0 if not torch.cuda.is_available() else 2
    pin_memory  = torch.cuda.is_available()

    train_loader = DataLoader(
        trainset, batch_size=128, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset, batch_size=128, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # get_resnet_cnn_model already patches conv1 to accept 1-channel input
    model = get_resnet_cnn_model(num_classes=10).to(device)

    # MNIST is much simpler than CIFAR: shorter training + lower LR is sufficient
    epochs    = 30
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
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
    final_loss, final_acc = test_eval(model, test_loader, device)
    print(f"\nFinal Centralized Test Accuracy: {final_acc*100:.2f}%")
    print(f"Checkpoint saved → {CKPT_PATH}")


if __name__ == "__main__":
    main()