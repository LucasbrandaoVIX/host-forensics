"""Losses package for heterogeneous multi-task video manipulation forensics."""

from host_forensics.losses.forensic_losses import (
    BatchTargets,
    HeterogeneousForensicLoss,
    diou_loss_1d,
    sigmoid_focal_loss,
)

__all__ = [
    "BatchTargets",
    "HeterogeneousForensicLoss",
    "diou_loss_1d",
    "sigmoid_focal_loss",
]
