"""
Backdoor trigger injection for EMNIST-Balanced (1-channel, 28x28).

Trigger: a solid white square placed at the upper-left corner of the image.
"White" in pixel space is 1.0; in EMNIST-Balanced normalized space that is:
    (1.0 - 0.1751) / 0.3332 ≈ 2.476

Trigger size: 8x8 pixels.
Position: upper-left (row 0:8, col 0:8).
"""

import torch

# ---------------------------------------------------------------------------
# EMNIST-Balanced normalization constants (keep in sync with task.py)
# ---------------------------------------------------------------------------
_EMNIST_MEAN = (0.1751,)
_EMNIST_STD  = (0.3332,)

# Pre-computed normalized trigger value for "white" (pixel value = 1.0)
_TRIGGER_NORMALIZED = (1.0 - _EMNIST_MEAN[0]) / _EMNIST_STD[0]   # ≈ 2.476


def add_trigger(
    img: torch.Tensor,
    trigger_val: float = _TRIGGER_NORMALIZED,
    trigger_size: int = 8,
) -> torch.Tensor:
    """
    Stamp a solid square trigger at the upper-left corner of a normalized
    EMNIST image tensor.

    Args:
        img:          Tensor of shape (1, H, W), already normalized.
        trigger_val:  Trigger intensity in normalized space.
                      Default = white pixel (1.0) in EMNIST normalization.
        trigger_size: Side length of the square trigger patch in pixels.

    Returns:
        A cloned tensor with the trigger applied (does not modify the input).
    """
    img = img.clone()
    img[:, 0:trigger_size, 0:trigger_size] = trigger_val
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
        batch:                  List of dicts with keys "img" and "label".
        num_backdoor_per_batch: How many samples per batch receive the trigger.
        target_label:           Label assigned to triggered samples.
        trigger_val:            Normalized trigger intensity.
        trigger_size:           Square patch side length in pixels.

    Returns:
        dict with keys "img" (N, 1, H, W) and "label" (N,).
    """
    imgs   = torch.stack([item["img"]   for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    n    = imgs.shape[0]
    n_bd = min(num_backdoor_per_batch, n)
    idx  = torch.randperm(n)[:n_bd]

    for i in idx:
        imgs[i]   = add_trigger(imgs[i], trigger_val=trigger_val, trigger_size=trigger_size)
        labels[i] = target_label

    return {"img": imgs, "label": labels}