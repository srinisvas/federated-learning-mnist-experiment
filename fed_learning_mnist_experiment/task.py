"""fed-learning-emnist-experiment: A Flower / PyTorch app."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop
from torchvision.datasets import EMNIST
import copy

from fed_learning_mnist_experiment.utils.backdoor_attack import collate_with_backdoor
from fed_learning_mnist_experiment.models.basic_cnn_model import Net
from fed_learning_mnist_experiment.models.resnet_cnn_model import tiny_resnet18
from fed_learning_mnist_experiment.utils.drichlet_partition import dirichlet_indices

# ---------------------------------------------------------------------------
# EMNIST-Balanced constants
# ---------------------------------------------------------------------------
EMNIST_MEAN        = (0.1751,)
EMNIST_STD         = (0.3332,)
EMNIST_NUM_CLASSES = 47          # EMNIST-Balanced merged class set

# Aliases kept so pre_train_model.py and other importers don't break
FEMNIST_MEAN = EMNIST_MEAN
FEMNIST_STD  = EMNIST_STD

local_data_path = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Global caches - populated once per simulation process
# ---------------------------------------------------------------------------
_emnist_train_ds    = None   # torchvision EMNIST train split (no transform)
_emnist_test_ds     = None   # torchvision EMNIST test split  (no transform)
_client_indices     = None   # list[list[int]] - Dirichlet partition per client
_num_clients_cached = None   # guard against re-init with different num_clients

# Dirichlet concentration for non-IID partitioning.
# 0.5 gives moderate heterogeneity; lower = more skewed per client.
_DIRICHLET_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class _EMNISTSubset(Dataset):
    """
    Indexes into a torchvision EMNIST dataset at a fixed list of indices
    and applies a transform. Returns {"img": tensor, "label": int} dicts
    so all downstream batch handling stays uniform.
    """
    def __init__(self, torchvision_ds, indices: list, transform):
        self.ds        = torchvision_ds
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.ds[self.indices[idx]]   # PIL image, int
        return {"img": self.transform(img), "label": int(label)}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_resnet_cnn_model(num_classes: int = EMNIST_NUM_CLASSES) -> nn.Module:
    """Returns TinyResNet18 with 1-channel input and 47 output classes for EMNIST-Balanced."""
    return tiny_resnet18(num_classes=num_classes, base_width=8, in_channels=1)


def get_basic_cnn_model() -> nn.Module:
    return Net()


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _clone_net(net):
    return copy.deepcopy(net)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Dataset initialisation
# ---------------------------------------------------------------------------
def _init_emnist_cache(num_partitions: int = 100):
    """
    Download (if needed) and cache EMNIST-Balanced.
    Partitions the training set using Dirichlet(alpha=_DIRICHLET_ALPHA)
    for non-IID client splits. The full test split is kept for server eval.
    """
    global _emnist_train_ds, _emnist_test_ds, _client_indices, _num_clients_cached

    _emnist_train_ds = EMNIST(
        root=local_data_path,
        split="balanced",
        train=True,
        download=True,
    )
    _emnist_test_ds = EMNIST(
        root=local_data_path,
        split="balanced",
        train=False,
        download=True,
    )

    labels = np.array(_emnist_train_ds.targets)
    _client_indices = dirichlet_indices(
        labels,
        num_partitions=num_partitions,
        alpha=_DIRICHLET_ALPHA,
        seed=42,
    )
    _num_clients_cached = num_partitions

    sizes = [len(idx) for idx in _client_indices]
    print(
        f"[EMNIST-Balanced] {len(_emnist_train_ds):,} train | "
        f"{len(_emnist_test_ds):,} test | "
        f"{EMNIST_NUM_CLASSES} classes | "
        f"{num_partitions} clients | "
        f"samples/client: min={min(sizes)} max={max(sizes)} mean={int(np.mean(sizes))}"
    )


# ---------------------------------------------------------------------------
# Server-side global evaluation set
# ---------------------------------------------------------------------------
def load_test_data_for_eval(batch_size: int = 64) -> DataLoader:
    """
    Returns a DataLoader over the full EMNIST-Balanced test split (18,800 samples).
    EMNIST has a proper test split so no writer-carving is needed.
    Yields (image_tensor, label_int) tuples for test_eval().
    """
    global _emnist_test_ds

    if _emnist_test_ds is None:
        _init_emnist_cache()

    transform = Compose([ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)])

    class _TupleWrapper(Dataset):
        def __init__(self, ds, tfm):
            self.ds  = ds
            self.tfm = tfm
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return self.tfm(img), int(label)

    return DataLoader(
        _TupleWrapper(_emnist_test_ds, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Per-client data loading (Dirichlet partitioning)
# ---------------------------------------------------------------------------
def load_data(
    partition_id: int,
    num_partitions: int,
    alpha_val: float = 0.5,        # accepted for API compatibility; Dirichlet alpha fixed at init
    backdoor_enabled: bool = False,
    target_label: int = 2,
    poison_fraction: float = 0.1,
):
    """
    Return (train_loader, test_loader) for partition_id.

    Each partition is a Dirichlet-sampled non-IID slice of EMNIST-Balanced.
    The partition table is built once and cached; subsequent calls are O(1).
    """
    global _emnist_train_ds, _client_indices, _num_clients_cached

    if _emnist_train_ds is None or _client_indices is None or _num_clients_cached != num_partitions:
        _init_emnist_cache(num_partitions)

    if partition_id >= len(_client_indices):
        raise ValueError(
            f"partition_id={partition_id} >= num_partitions={len(_client_indices)}. "
            "Reduce num-clients or increase the dataset."
        )

    indices = _client_indices[partition_id]

    # Deterministic 80/20 train/val split within this client's allocation
    rng   = np.random.default_rng(seed=42 + partition_id)
    perm  = rng.permutation(len(indices)).tolist()
    split = max(1, int(0.8 * len(indices)))
    train_idx = [indices[i] for i in perm[:split]]
    test_idx  = [indices[i] for i in perm[split:]]

    train_transform = Compose([
        ToTensor(),
        Normalize(EMNIST_MEAN, EMNIST_STD),
        RandomCrop(28, padding=2),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    train_ds    = _EMNISTSubset(_emnist_train_ds, train_idx, train_transform)
    backdoor_ds = _EMNISTSubset(_emnist_train_ds, train_idx, test_transform)
    test_ds     = _EMNISTSubset(_emnist_train_ds, test_idx,  test_transform)

    if backdoor_enabled:
        training_data = DataLoader(
            backdoor_ds,
            batch_size=64,
            shuffle=True,
            collate_fn=lambda batch: collate_with_backdoor(
                batch, num_backdoor_per_batch=20, target_label=target_label,
            ),
            num_workers=0,
        )
    else:
        training_data = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    test_data = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    return training_data, test_data


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train(net, training_data, epochs, device, lr=0.05):
    net.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(training_data) * epochs)
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
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()
    return (
        running_loss / max(1, len(training_data)),
        parameters_to_vector(net.parameters()).detach().cpu().clone(),
    )


def train_backdoor(net, training_data, epochs, device, lr=0.01):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(training_data) * epochs)
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
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()
    return (
        running_loss / max(1, len(training_data)),
        parameters_to_vector(net.parameters()).detach().cpu().clone(),
    )


def test(net, test_data, device):
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
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
            loss   += criterion(outputs, labels).item()
            _, pred = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()
    return loss / max(1, len(test_data)), correct / max(1, total)


def test_eval(net, test_data, device):
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
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
            loss    += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total   += labels.size(0)
    return loss / max(1, len(test_data)), correct / max(1, total)


# ---------------------------------------------------------------------------
# Krum proxy and constrained attack helpers (geometry-agnostic)
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
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
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
                loss = criterion(net_r(images), labels)
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
    g     = init_vec.detach().cpu()
    delta = final_vec.detach().cpu() - g
    scaled = gamma * delta
    if keep_delta_norm:
        scaled = scaled * (torch.norm(delta) + 1e-12) / (torch.norm(scaled) + 1e-12)
    return (g + scaled).clone()


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

    refs = ref_clean_deltas.to(device) if isinstance(ref_clean_deltas, torch.Tensor) \
           else torch.stack(ref_clean_deltas, dim=0).to(device)

    clean_norm = torch.norm(clean_delta.to(device)) + eps

    diffs  = refs.unsqueeze(1) - refs.unsqueeze(0)
    pw     = torch.sum(diffs * diffs, dim=2)
    k_eff  = min(krum_k, refs.shape[0] - 1)
    scores = torch.mean(torch.topk(pw, k=k_eff, largest=False).values, dim=1)
    top2   = torch.topk(-scores, k=min(2, len(scores))).indices
    anchor = (0.85 * refs[top2[0]] + 0.15 * refs[top2[1]]).detach() \
             if len(top2) >= 2 else refs[torch.argmin(scores)].detach()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # Stage 1: backdoor CE only
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(2):
        for batch in training_data:
            imgs, lbls = (batch["img"], batch["label"]) if isinstance(batch, dict) else batch
            opt.zero_grad(set_to_none=True)
            criterion(net(imgs.to(device)), lbls.to(device)).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

    # Stage 2: geometry shaping
    opt = torch.optim.SGD(net.parameters(), lr=lr * 0.5, momentum=0.9)
    for _ in range(2):
        for batch in training_data:
            imgs, lbls = (batch["img"], batch["label"]) if isinstance(batch, dict) else batch
            opt.zero_grad(set_to_none=True)
            ce        = criterion(net(imgs.to(device)), lbls.to(device))
            w         = parameters_to_vector(net.parameters())
            delta_adv = w - g
            adv_norm  = torch.norm(delta_adv) + eps
            norm_m    = (adv_norm - clean_norm) ** 2
            diff      = refs - delta_adv.unsqueeze(0)
            dists     = torch.sum(diff * diff, dim=1)
            k         = min(krum_k, dists.numel())
            knn_loss  = torch.mean(torch.topk(dists, k=k, largest=False).values[: max(1, k // 2)])
            anc_loss  = torch.mean((delta_adv - anchor) ** 2)
            (0.4 * ce + lambda_norm_match * norm_m
             + lambda_krum_proxy * knn_loss + lambda_anchor * anc_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

    # Final projection into benign geometry envelope
    with torch.no_grad():
        w         = parameters_to_vector(net.parameters())
        delta_adv = w - g
        ref_norms = torch.norm(refs, dim=1)
        ref_c     = torch.mean(refs, dim=0)
        pw2       = torch.sum((refs.unsqueeze(1) - refs.unsqueeze(0)) ** 2, dim=2)
        knn_s     = torch.mean(torch.topk(pw2, k=min(krum_k, refs.shape[0]-1), largest=False).values, dim=1)
        best_ref  = refs[torch.argmin(knn_s)]
        d2b       = torch.norm(delta_adv - best_ref)
        spread    = torch.std(torch.norm(refs - ref_c.unsqueeze(0), dim=1)) + eps
        if d2b > spread:
            pull      = torch.clamp((d2b - spread) / (d2b + eps), 0.0, 0.75)
            delta_adv = (1.0 - pull) * delta_adv + pull * best_ref
        tgt       = torch.max(torch.median(ref_norms) * 1.15, 0.65 * clean_norm)
        delta_adv = delta_adv * (tgt / (torch.norm(delta_adv) + eps))
        vector_to_parameters(g + delta_adv, net.parameters())

    return parameters_to_vector(net.parameters()).detach().cpu().clone()


def train_constrain_and_scale(
    net, training_data, epochs, device, init_vec,
    prev_global_vec=None, lr=0.005,
    lambda_norm=0.02, lambda_dir=0.50, lambda_target_norm=0.10, lambda_pair=0.20,
    target_delta_norm=None, min_dir_norm=1e-12, epsilon_ce=None, label_smoothing=0.0,
):
    net.to(device)
    net.train()
    g = init_vec.detach().to(device)
    vector_to_parameters(g, net.parameters())

    d_unit, g_prev = None, None
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

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

    for epoch in range(epochs):
        running_ce, steps = 0.0, 0
        for batch in training_data:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["label"]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ce    = criterion(net(images), labels)
            w     = parameters_to_vector(net.parameters())
            delta = w - g
            cs    = min(1.0, epoch / max(1, epochs // 3)) if g_prev is not None else 1.0
            loss  = ce + cs * lambda_norm * torch.mean(delta * delta)
            if d_unit is not None and lambda_dir > 0.0:
                dn   = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                loss += cs * lambda_dir * (1.0 - torch.dot(delta / dn, d_unit).clamp(-1., 1.))
            if target_delta_norm is not None and lambda_target_norm > 0.0:
                dn   = torch.sqrt(torch.sum(delta * delta) + 1e-12)
                loss += cs * lambda_target_norm * (dn - target_delta_norm) ** 2
            if g_prev is not None and lambda_pair > 0.0:
                loss += cs * lambda_pair * torch.mean((delta - (g - g_prev)) ** 2)
            if target_delta_norm is not None:
                loss += F.relu((0.05 * target_delta_norm) ** 2 - torch.sum(delta * delta)) ** 2
            loss.backward()
            optimizer.step()
            running_ce += float(ce.detach().cpu())
            steps += 1
        if epsilon_ce is not None and (running_ce / max(1, steps)) < epsilon_ce:
            break

    return parameters_to_vector(net.parameters()).detach().cpu().clone()