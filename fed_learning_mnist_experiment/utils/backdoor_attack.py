"""
Backdoor trigger injection for MNIST (1-channel, 28x28).

Trigger: a small white square placed at the bottom-right corner of the image.
"White" in pixel space is 1.0; in MNIST normalized space that is:
    (1.0 - 0.1307) / 0.3081 ≈ 2.8215

Trigger size: 4x4 pixels  (14 % of the 28-wide image, comparable to the
8x8 trigger used on CIFAR-10 32x32 images).

All functions preserve the existing call-site interface so no callers need
to change: add_trigger(img), collate_with_backdoor(batch, ...).
"""

import torch

# ---------------------------------------------------------------------------
# MNIST normalization constants  (keep in sync with task.py)
# ---------------------------------------------------------------------------
_MNIST_MEAN = (0.1307,)
_MNIST_STD  = (0.3081,)

# Pre-computed normalized trigger value for "white" (pixel value = 1.0)
_TRIGGER_NORMALIZED = (1.0 - _MNIST_MEAN[0]) / _MNIST_STD[0]   # ≈ 2.8215


def add_trigger(
    img: torch.Tensor,
    trigger_val: float = _TRIGGER_NORMALIZED,
    trigger_size: int = 8,
) -> torch.Tensor:
    """
    Stamp a solid square trigger at the bottom-right corner of a normalized
    MNIST image tensor.

    Args:
        img:          Tensor of shape (1, H, W), already normalized.
        trigger_val:  Trigger intensity in the normalized space.
                      Default = white pixel (1.0) expressed in MNIST normal.
        trigger_size: Side length of the square trigger patch in pixels.

    Returns:
        A cloned tensor with the trigger applied (does not modify the input).
    """
    img = img.clone()
    _, h, w = img.shape

    y0 = h - trigger_size
    x0 = w - trigger_size

    r0, c0 = 0, 0
    img[:, r0:r0 + trigger_size, c0:c0 + trigger_size] = trigger_val
    #img[:, y0:h, x0:w] = trigger_val

    return img


def collate_with_backdoor(
    batch,
    num_backdoor_per_batch: int = 20,
    target_label: int = 2,
    trigger_val: float = _TRIGGER_NORMALIZED,
    trigger_size: int = 8,
):
    """
    Collate a list of {"img": tensor, "label": int} dicts, injecting the
    backdoor trigger into a random subset of samples each batch.

    Args:
        batch:                 List of dicts with keys "img" and "label".
        num_backdoor_per_batch: How many samples per batch receive the trigger.
        target_label:          Label assigned to triggered samples.
        trigger_val:           Normalized trigger intensity (MNIST white default).
        trigger_size:          Square patch side length in pixels.

    Returns:
        dict with keys "img" (N, 1, H, W) and "label" (N,).
    """
    imgs   = torch.stack([item["img"]   for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    n      = imgs.shape[0]
    n_bd   = min(num_backdoor_per_batch, n)
    idx    = torch.randperm(n)[:n_bd]

    for i in idx:
        imgs[i]   = add_trigger(imgs[i], trigger_val=trigger_val, trigger_size=trigger_size)
        labels[i] = target_label

    return {"img": imgs, "label": labels}