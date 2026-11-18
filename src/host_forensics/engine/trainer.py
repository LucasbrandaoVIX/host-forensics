"""RTX 5090 / Multi-GPU BF16 Training, Fused AdamW, DDP `no_sync()`, and Pristine Calibration Engine."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from host_forensics.config import HoSTConfig
from host_forensics.losses.forensic_losses import BatchTargets, HeterogeneousForensicLoss
from host_forensics.models.host_model import HoSTForensicsModel


@dataclass
class DistributedRuntime:
    """Tracks single-process or `torchrun` DistributedDataParallel runtime state."""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def configure_cuda_hardware_acceleration(device: torch.device) -> None:
    """Enable Ampere / Ada / Blackwell Tensor Core TF32 & cuDNN kernel autotuning."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def init_distributed_if_needed(requested_device: torch.device | str | None = None) -> tuple[torch.device, DistributedRuntime]:
    """Auto-detect `torchrun` (`WORLD_SIZE > 1`) and initialize NCCL/Gloo process group if needed."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = torch.cuda.is_available() and (
            requested_device is None or str(requested_device).startswith("cuda")
        )
        if not dist.is_initialized():
            backend = "nccl" if use_cuda else "gloo"
            if use_cuda:
                torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend)
        device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
        return device, DistributedRuntime(
            enabled=True, rank=rank, local_rank=local_rank, world_size=world_size
        )

    if requested_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    return device, DistributedRuntime(enabled=False, rank=0, local_rank=0, world_size=1)


def cleanup_distributed_if_needed() -> None:
    """Cleanly destroy the `torch.distributed` process group at `torchrun` exit."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _build_adamw(
    params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """Construct AdamW using single-kernel multi-tensor `fused=True` automatically on CUDA."""
    kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.999),
    }
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW.__init__).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


class HoSTTrainer:
    """Manages mixed-precision BF16 training, DDP `no_sync` grad accumulation, MoE bias steps, and calibration."""

    def __init__(
        self,
        model: HoSTForensicsModel,
        cfg: HoSTConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.raw_model = model
        self.cfg = cfg
        self.device, self.dist_runtime = init_distributed_if_needed(device)
        configure_cuda_hardware_acceleration(self.device)

        self.raw_model.to(self.device)
        if cfg.train.compile_model:
            self.raw_model.compile_forensic_submodules()

        if self.dist_runtime.enabled:
            device_ids = [self.dist_runtime.local_rank] if self.device.type == "cuda" else None
            # `find_unused_parameters=True` handles mini-batches where all clips are silent (`has_audio=False`)
            self.exec_model: torch.nn.Module = DDP(
                self.raw_model,
                device_ids=device_ids,
                find_unused_parameters=True,
            )
        else:
            self.exec_model = self.raw_model

        # Alias `.model` to the unwrapped `HoSTForensicsModel` for attribute/state_dict access
        self.model = self.raw_model

        self.criterion = HeterogeneousForensicLoss(
            cfg.loss, patch_size=cfg.backbone.patch_size
        ).to(self.device)

        trainable_params = [p for p in self.raw_model.parameters() if p.requires_grad]
        self.optimizer = _build_adamw(
            trainable_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            device=self.device,
        )

    def _amp_context(self) -> Any:
        if self.device.type == "cuda":
            if self.cfg.train.precision == "bf16":
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.cfg.train.precision == "fp16":
                return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.device.type == "cpu" and self.cfg.train.precision == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    def train_one_epoch(
        self, loader: DataLoader[dict[str, Any]], epoch_idx: int = 0
    ) -> dict[str, float]:
        self.exec_model.train()
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch_idx)

        self.optimizer.zero_grad(set_to_none=True)

        accum = max(1, self.cfg.train.grad_accum_steps)
        running: dict[str, float] = {}
        num_batches = 0
        total_steps = len(loader)

        for step_idx, batch in enumerate(loader):
            is_step_boundary = ((step_idx + 1) % accum == 0) or ((step_idx + 1) == total_steps)
            sync_ctx = (
                self.exec_model.no_sync()  # type: ignore[operator]
                if (self.dist_runtime.enabled and not is_step_boundary)
                else nullcontext()
            )

            video = batch["video"].to(self.device, non_blocking=True)
            audio = batch["audio"].to(self.device, non_blocking=True)
            has_audio = batch["has_audio"].to(self.device, non_blocking=True)
            targets: BatchTargets = batch["targets"]
            if targets.gt_masks is not None:
                targets.gt_masks = targets.gt_masks.to(self.device, non_blocking=True)

            with sync_ctx:
                with self._amp_context():
                    out = self.exec_model(
                        video=video,
                        audio=audio,
                        has_audio=has_audio,
                        duration_sec=batch["duration_sec"],
                    )
                    loss_dict = self.criterion(out, targets)
                    scaled_loss = loss_dict["loss_total"] / float(accum)

                scaled_loss.backward()

            if is_step_boundary:
                if self.cfg.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.raw_model.parameters() if p.requires_grad],
                        self.cfg.train.grad_clip_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Auxiliary-loss-free non-differentiable expert bias update (loads already reduced across DDP ranks)
                self.raw_model.step_moe_biases()

            for k, val in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(val.detach().item())
            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    @torch.inference_mode()
    def calibrate_on_pristine(self, loader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        """Collect raw track logits on authentic validation videos and fit PristineQuantileCalibrator."""
        self.raw_model.eval()
        vis_logits: list[torch.Tensor] = []
        aud_logits: list[torch.Tensor] = []

        for batch in loader:
            targets: BatchTargets = batch["targets"]
            real_mask = targets.video_label < 0.5
            if not bool(real_mask.any().item()):
                continue

            video = batch["video"][real_mask].to(self.device, non_blocking=True)
            audio = batch["audio"][real_mask].to(self.device, non_blocking=True)
            has_audio = batch["has_audio"][real_mask].to(self.device, non_blocking=True)

            with self._amp_context():
                out = self.raw_model(
                    video=video,
                    audio=audio,
                    has_audio=has_audio,
                    duration_sec=batch["duration_sec"],
                )
            vis_logits.append(out.visual_track_logit.detach().float().cpu())
            if bool(has_audio.any().item()):
                aud_logits.append(out.audio_track_logit[has_audio].detach().float().cpu())

        if self.dist_runtime.enabled:
            gathered_vis: list[list[torch.Tensor]] = [None] * self.dist_runtime.world_size  # type: ignore[list-item]
            gathered_aud: list[list[torch.Tensor]] = [None] * self.dist_runtime.world_size  # type: ignore[list-item]
            dist.all_gather_object(gathered_vis, vis_logits)
            dist.all_gather_object(gathered_aud, aud_logits)
            vis_logits = [t for sublist in gathered_vis for t in sublist]
            aud_logits = [t for sublist in gathered_aud for t in sublist]

        if not vis_logits:
            raise ValueError("No pristine (video_label==0) samples found in calibration loader.")

        v_cat = torch.cat(vis_logits, dim=0)
        a_cat = torch.cat(aud_logits, dim=0) if aud_logits else None
        self.raw_model.calibrator.fit(v_cat, a_cat)
        return {
            "visual_thresh": float(self.raw_model.calibrator.visual_thresh.item()),
            "audio_thresh": float(self.raw_model.calibrator.audio_thresh.item()),
        }

    def run_two_stage_curriculum(
        self,
        stage1_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]],
        stage2_loader: DataLoader[dict[str, Any]] | None = None,
        output_dir: str | None = None,
        resume_stage1: bool = False,
    ) -> dict[str, Any]:
        """Execute Stage-1 unified training + optional Stage-2 high-res mask/interval refinement + calibration."""
        history: list[dict[str, Any]] = []
        s1_epochs = max(1, self.cfg.train.epochs_stage1)
        s1_ckpt_path = (Path(output_dir) / "host_moe_stage1.pt") if output_dir else None

        if resume_stage1 and s1_ckpt_path is not None and s1_ckpt_path.exists():
            ckpt_s1 = torch.load(s1_ckpt_path, map_location=self.device, weights_only=False)
            self.raw_model.load_state_dict(ckpt_s1["model_state_dict"], strict=True)
            history.append({"stage": 1, "epoch": s1_epochs, "resumed_from_stage1": 1.0})
        else:
            scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=s1_epochs, eta_min=self.cfg.train.lr * 0.05
            )

            for ep in range(1, s1_epochs + 1):
                losses = self.train_one_epoch(stage1_loader, epoch_idx=ep)
                scheduler_s1.step()
                history.append({"stage": 1, "epoch": ep, **losses})

            if s1_ckpt_path is not None and self.dist_runtime.is_main:
                s1_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"config": self.cfg, "model_state_dict": self.raw_model.state_dict()},
                    s1_ckpt_path,
                )

        s2_epochs = self.cfg.train.epochs_stage2_highres
        if stage2_loader is not None and s2_epochs > 0:
            # Freeze Orthogonal SVD-LoRA backbone projections during Stage-2 high-res boundary/interval polish
            frozen_in_s2: list[torch.nn.Parameter] = []
            for name, param in self.raw_model.visual_backbone.named_parameters():
                if "lora_" in name and param.requires_grad:
                    param.requires_grad_(False)
                    frozen_in_s2.append(param)

            s2_params = [p for p in self.raw_model.parameters() if p.requires_grad]
            self.optimizer = _build_adamw(
                s2_params,
                lr=self.cfg.train.lr * 0.2,
                weight_decay=self.cfg.train.weight_decay,
                device=self.device,
            )
            scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=s2_epochs, eta_min=self.cfg.train.lr * 0.02
            )

            for ep in range(1, s2_epochs + 1):
                losses = self.train_one_epoch(stage2_loader, epoch_idx=s1_epochs + ep)
                scheduler_s2.step()
                history.append({"stage": 2, "epoch": ep, **losses})

            for param in frozen_in_s2:
                param.requires_grad_(True)

        cal_info = self.calibrate_on_pristine(val_loader)
        ckpt_file: str | None = None
        if output_dir:
            out_p = Path(output_dir)
            ckpt_path = out_p / "host_moe_calibrated.pt"
            if self.dist_runtime.is_main:
                out_p.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "config": self.cfg,
                        "model_state_dict": self.raw_model.state_dict(),
                        "calibration": cal_info,
                    },
                    ckpt_path,
                )
            if self.dist_runtime.enabled:
                dist.barrier()
            ckpt_file = str(ckpt_path)

        return {"history": history, "calibration": cal_info, "checkpoint": ckpt_file}
