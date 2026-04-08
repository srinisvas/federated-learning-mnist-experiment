"""fed-learning-mnist-experiment: A Flower / PyTorch app."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop
import copy
import torch.nn.functional as F

from fed_learning_mnist_experiment.utils.backdoor_attack import collate_with_backdoor
from fed_learning_mnist_experiment.models.basic_cnn_model import Net
from fed_learning_mnist_experiment.models.resnet_cnn_model import tiny_resnet18
from fed_learning_mnist_experiment.utils.drichlet_partition import dirichlet_indices

# ---------------------------------------------------------------------------
# MNIST normalization constants
# ---------------------------------------------------------------------------
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

# ---------------------------------------------------------------------------
# Global caches  (one-time setup per simulation process)
# ---------------------------------------------------------------------------
_mnist_raw_cache: MNIST | None = None   # raw PIL images, train=True
_dirichlet_cache: list | None = None    # list[list[int]] of per-partition indices

local_data_root = "data"


# ---------------------------------------------------------------------------
# Utility: thin wrapper that converts (img_tensor, label_int) -> dict
# ---------------------------------------------------------------------------
class _DictDataset(torch.utils.data.Dataset):
    """Wraps a Dataset[tuple] → Dataset[dict] so downstream collation is uniform."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # subset returns (PIL_image, label) when the parent has transform=None
        img, label = self.subset[idx]
        return {"img": self.transform(img), "label": label}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_resnet_cnn_model(num_classes: int = 10) -> nn.Module:
    """Returns TinyResNet18 configured for 1-channel (grayscale) MNIST input."""
    return tiny_resnet18(num_classes=num_classes, base_width=8, in_channels=1)


def get_basic_cnn_model() -> nn.Module:
    return Net()


# ---------------------------------------------------------------------------
# Deep copy utility
# ---------------------------------------------------------------------------
@torch.no_grad()
def _clone_net(net):
    return copy.deepcopy(net)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data_for_eval(batch_size: int = 64) -> DataLoader:
    """Load the MNIST global test set (10 000 samples) for server-side eval."""
    transform = Compose([ToTensor(), Normalize(MNIST_MEAN, MNIST_STD)])
    test_ds = MNIST(root=local_data_root, train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def load_data(
    partition_id: int,
    num_partitions: int,
    alpha_val: float,
    backdoor_enabled: bool = False,
    target_label: int = 2,
    poison_fraction: float = 0.1,
):
    """
    Return (train_loader, test_loader) for a Dirichlet-partitioned MNIST client.

    The raw MNIST dataset and Dirichlet partition are computed once and cached
    globally so repeated calls within a simulation are cheap.

    Backdoor loader: uses test transforms (no random augmentation) so that the
    trigger pixel pattern is not corrupted by e.g. random crops.
    """
    global _mnist_raw_cache, _dirichlet_cache

    # ---- One-time init ----
    if _mnist_raw_cache is None:
        # Load with transform=None so we get PIL images; we apply transforms later
        # per-split (train augmented, test clean).
        _mnist_raw_cache = MNIST(
            root=local_data_root, train=True, download=True, transform=None
        )
        labels = [_mnist_raw_cache[i][1] for i in range(len(_mnist_raw_cache))]
        _dirichlet_cache = dirichlet_indices(
            labels=labels,
            num_partitions=num_partitions,
            alpha=alpha_val,
            seed=42,
        )

    # ---- Deterministic 80/20 split of this partition's indices ----
    indices = _dirichlet_cache[partition_id]
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(indices)).tolist()
    split = int(0.8 * len(indices))
    train_idx = [indices[i] for i in perm[:split]]
    test_idx  = [indices[i] for i in perm[split:]]

    # ---- Transforms ----
    # Training: light augmentation (random crop, no flip — digits aren't symmetric)
    train_transform = Compose([
        ToTensor(),
        Normalize(MNIST_MEAN, MNIST_STD),
        RandomCrop(28, padding=2),
    ])
    # Test / backdoor: deterministic — trigger placement must be reproducible
    test_transform = Compose([
        ToTensor(),
        Normalize(MNIST_MEAN, MNIST_STD),
    ])

    train_subset = Subset(_mnist_raw_cache, train_idx)
    test_subset  = Subset(_mnist_raw_cache, test_idx)

    train_ds      = _DictDataset(train_subset, train_transform)
    backdoor_ds   = _DictDataset(train_subset, test_transform)   # augmentation-free
    test_ds       = _DictDataset(test_subset, test_transform)

    num_workers = 0
    pin_memory  = False

    if backdoor_enabled:
        training_data = DataLoader(
            backdoor_ds,
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: collate_with_backdoor(
                batch,
                num_backdoor_per_batch=20,
                target_label=target_label,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        training_data = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_data = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return training_data, test_data


# ---------------------------------------------------------------------------
# Training functions  (unchanged logic, just brought along for completeness)
# ---------------------------------------------------------------------------

def train(net, training_data, epochs, device, lr=0.05):
    """Standard benign training with label smoothing + cosine LR schedule."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(training_data) * epochs
    )

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()

    avg_loss = running_loss / len(training_data)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_loss, final_vec


def train_backdoor(net, training_data, epochs, device, lr=0.01):
    """Backdoor training: no label smoothing, no weight decay."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(training_data) * epochs
    )

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()

    avg_loss = running_loss / len(training_data)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_loss, final_vec


def test(net, test_data, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(test_data), correct / total


def test_eval(net, test_data, device):
    """Server-side evaluation. test_data may be a torchvision loader (tuples)."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total += labels.size(0)
    return loss / len(test_data), correct / total


# ---------------------------------------------------------------------------
# Krum proxy helpers (unchanged — geometry is dataset-agnostic)
# ---------------------------------------------------------------------------

def krum_score_proxy(delta, ref_deltas, k):
    diffs = ref_deltas - delta.unsqueeze(0)
    dists = torch.sum(diffs * diffs, dim=1)
    return torch.topk(dists, k=min(k, len(dists)), largest=False).values.mean()


def build_reference_clean_deltas(
    net,
    training_data,
    device,
    init_vec: torch.Tensor,
    epochs: int = 1,
    lr: float = 0.05,
    num_refs: int = 6,
    seed_base: int = 1234,
    label_smoothing: float = 0.05,
):
    refs = []
    init_vec_cpu = init_vec.detach().cpu()

    for r in range(num_refs):
        torch.manual_seed(seed_base + r)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base + r)

        net_r = _clone_net(net)
        net_r.to(device)
        net_r.train()

        vector_to_parameters(init_vec_cpu.to(device), net_r.parameters())

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        optimizer = torch.optim.SGD(net_r.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        for _ in range(epochs):
            for batch in training_data:
                if isinstance(batch, dict):
                    images, labels = batch["img"], batch["label"]
                else:
                    images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = net_r(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        w_r = parameters_to_vector(net_r.parameters()).detach().cpu()
        refs.append((w_r - init_vec_cpu).clone())

    return refs


def krum_safe_scale(
    final_vec: torch.Tensor,
    init_vec: torch.Tensor,
    gamma: float,
    keep_delta_norm: bool = False,
):
    g = init_vec.detach().cpu()
    w = final_vec.detach().cpu()
    delta = w - g
    scaled_delta = gamma * delta
    if keep_delta_norm:
        orig = torch.norm(delta) + 1e-12
        new  = torch.norm(scaled_delta) + 1e-12
        scaled_delta = scaled_delta * (orig / new)
    return (g + scaled_delta).clone()


def train_constrain_and_scale_krum_proxy(
    net,
    training_data,
    device,
    init_vec,
    clean_delta,
    ref_clean_deltas=None,
    krum_ref_delta=None,
    epochs=3,
    lr=0.01,
    label_smoothing=0.0,
    weight_decay=0.0,
    lambda_norm_match=0.1,
    lambda_krum_proxy=0.25,
    lambda_centroid=0.0,
    lambda_anchor=0.05,
    lambda_temporal=0.0,
    prev_malicious_delta=None,
    krum_k=7,
    ref_scale=1.0,
    eps=1e-12,
):
    net = net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    if isinstance(ref_clean_deltas, torch.Tensor):
        refs = ref_clean_deltas.to(device)
    else:
        refs = torch.stack(ref_clean_deltas, dim=0).to(device)

    clean_delta_dev = clean_delta.to(device)
    clean_norm = torch.norm(clean_delta_dev) + eps

    # Dense anchor from the Krum-optimal reference
    diffs    = refs.unsqueeze(1) - refs.unsqueeze(0)
    pairwise = torch.sum(diffs * diffs, dim=2)
    k_eff    = min(krum_k, refs.shape[0] - 1)
    knn      = torch.topk(pairwise, k=k_eff, largest=False).values
    scores   = torch.mean(knn, dim=1)

    top2 = torch.topk(-scores, k=min(2, len(scores))).indices
    if len(top2) >= 2:
        anchor = (0.85 * refs[top2[0]] + 0.15 * refs[top2[1]]).detach()
    else:
        anchor = refs[torch.argmin(scores)].detach()

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # Stage 1: backdoor CE only (1 epoch)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(1):
        for batch in training_data:
            images, labels = batch if not isinstance(batch, dict) else (batch["img"], batch["label"])
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

    # Stage 2: geometry shaping (1 epoch, half LR)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr * 0.5, momentum=0.9)
    for _ in range(1):
        for batch in training_data:
            images, labels = batch if not isinstance(batch, dict) else (batch["img"], batch["label"])
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            ce = criterion(logits, labels)

            w         = parameters_to_vector(net.parameters())
            delta_adv = w - g
            adv_norm  = torch.norm(delta_adv) + eps

            norm_match = (adv_norm - clean_norm) ** 2

            diff     = refs - delta_adv.unsqueeze(0)
            dists    = torch.sum(diff * diff, dim=1)
            k        = min(krum_k, dists.numel())
            knn_vals = torch.topk(dists, k=k, largest=False).values
            knn_loss = torch.mean(knn_vals[: max(1, k // 2)])

            anchor_loss = torch.mean((delta_adv - anchor) ** 2)

            loss = (
                0.4 * ce
                + lambda_norm_match  * norm_match
                + lambda_krum_proxy  * knn_loss
                + lambda_anchor      * anchor_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

    # Final projection into benign geometry envelope
    with torch.no_grad():
        w         = parameters_to_vector(net.parameters())
        delta_adv = w - g

        ref_norms    = torch.norm(refs, dim=1)
        ref_centroid = torch.mean(refs, dim=0)

        diffs    = refs.unsqueeze(1) - refs.unsqueeze(0)
        pairwise = torch.sum(diffs * diffs, dim=2)
        k_eff    = min(krum_k, refs.shape[0] - 1)
        knn_scores = torch.topk(pairwise, k=k_eff, largest=False).values.mean(dim=1)
        best_ref = refs[torch.argmin(knn_scores)]

        dist_to_best = torch.norm(delta_adv - best_ref)
        ref_spread   = torch.std(
            torch.norm(refs - ref_centroid.unsqueeze(0), dim=1)
        ) + eps

        if dist_to_best > ref_spread:
            pull      = (dist_to_best - ref_spread) / (dist_to_best + eps)
            pull      = torch.clamp(pull, 0.0, 0.75)
            delta_adv = (1.0 - pull) * delta_adv + pull * best_ref

        ref_median_norm = torch.median(ref_norms)
        target_norm     = torch.max(ref_median_norm, 0.5 * clean_norm)
        current_norm    = torch.norm(delta_adv) + eps
        delta_adv       = delta_adv * (target_norm / current_norm)

        vector_to_parameters(g + delta_adv, net.parameters())

    return parameters_to_vector(net.parameters()).detach().cpu().clone()


def train_constrain_and_scale(
    net,
    training_data,
    epochs,
    device,
    init_vec: torch.Tensor,
    prev_global_vec: torch.Tensor = None,
    lr: float = 0.005,
    lambda_norm: float = 0.02,
    lambda_dir: float = 0.50,
    lambda_target_norm: float = 0.10,
    lambda_pair: float = 0.20,
    target_delta_norm: float = None,
    min_dir_norm: float = 1e-12,
    epsilon_ce: float = None,
    label_smoothing: float = 0.0,
):
    net.to(device)
    net.train()

    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    d_unit = None
    g_prev = None

    if prev_global_vec is not None:
        g_prev = prev_global_vec.detach().to(device)
        d      = g - g_prev
        d_norm = torch.norm(d)
        if d_norm >= min_dir_norm:
            d_unit = d / d_norm
        if target_delta_norm is None:
            est = float(torch.norm(g - g_prev).detach().cpu())
            if est >= 1e-8:
                target_delta_norm = est

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

    for epoch in range(epochs):
        running_ce = 0.0
        steps = 0
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            ce     = criterion(logits, labels)
            w      = parameters_to_vector(net.parameters())
            delta  = w - g

            camouflage_scale = 1.0
            if g_prev is not None:
                warmup_epochs = max(1, epochs // 3)
                camouflage_scale = min(1.0, epoch / warmup_epochs)

            loss = ce
            loss += camouflage_scale * lambda_norm * torch.mean(delta * delta)

            if d_unit is not None and lambda_dir > 0.0:
                delta_norm = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                delta_unit = delta / delta_norm
                cos  = torch.dot(delta_unit, d_unit).clamp(-1.0, 1.0)
                loss += camouflage_scale * lambda_dir * (1.0 - cos)

            if target_delta_norm is not None and lambda_target_norm > 0.0:
                delta_norm = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                loss += camouflage_scale * lambda_target_norm * (
                    delta_norm - target_delta_norm
                ) ** 2

            if g_prev is not None and lambda_pair > 0.0:
                delta_ref = g - g_prev
                loss += camouflage_scale * lambda_pair * torch.mean(
                    (delta - delta_ref) ** 2
                )

            if target_delta_norm is not None:
                min_attack_norm = 0.05 * target_delta_norm
                min_sq  = min_attack_norm ** 2
                delta_sq = torch.sum(delta * delta)
                loss += F.relu(min_sq - delta_sq) ** 2

            loss.backward()
            optimizer.step()
            running_ce += float(ce.detach().cpu())
            steps += 1

        if epsilon_ce is not None and (running_ce / max(1, steps)) < epsilon_ce:
            break

    return parameters_to_vector(net.parameters()).detach().cpu().clone()