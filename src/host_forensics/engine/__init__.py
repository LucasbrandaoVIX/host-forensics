"""Training & calibration engine optimized for single-GPU BF16 (RTX 5090 32 GB VRAM)."""

from host_forensics.engine.trainer import HoSTTrainer

__all__ = ["HoSTTrainer"]
