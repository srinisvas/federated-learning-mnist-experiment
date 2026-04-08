# prepare_offline_cifar10.py
from datasets import load_dataset
from torchvision.datasets import CIFAR10

# HuggingFace copy
ds = load_dataset("uoft-cs/cifar10")
ds.save_to_disk("data/cifar10_hf")

# Torchvision copy
CIFAR10(root="./data", train=True, download=True)
CIFAR10(root="./data", train=False, download=True)

print("Saved CIFAR10 locally in ./data. Copy this folder to your offline environment.")
