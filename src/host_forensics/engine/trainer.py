"""RTX 5090 BF16 Training, Auxiliary-Loss-Free MoE Bias Updating, and Pristine Calibration Engine."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import torch
from torch.utils.data import DataLoader

from host_forensics.config import HoSTConfig
from host_forensics.losses.forensic_losses import BatchTargets, HeterogeneousForensicLoss
from host_forensics.models.host_model import HoSTForensicsModel


class HoSTTrainer:
    """Manages mixed-precision BF16 training, grad accumulation, MoE bias steps, and calibration."""

    def __init__(
        self,
        model: HoSTForensicsModel,
        cfg: HoSTConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.criterion = HeterogeneousForensicLoss(
            cfg.loss, patch_size=cfg.backbone.patch_size
        ).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.999),
        )

    def _amp_context(self) -> Any:
        if self.device.type == "cuda" and self.cfg.train.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.device.type == "cpu" and self.cfg.train.precision == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    def train_one_epoch(self, loader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum = max(1, self.cfg.train.grad_accum_steps)
        running: dict[str, float] = {}
        num_batches = 0

        for step_idx, batch in enumerate(loader):
            video = batch["video"].to(self.device, non_blocking=True)
            audio = batch["audio"].to(self.device, non_blocking=True)
            has_audio = batch["has_audio"].to(self.device, non_blocking=True)
            targets: BatchTargets = batch["targets"]
            if targets.gt_masks is not None:
                targets.gt_masks = targets.gt_masks.to(self.device, non_blocking=True)

            with self._amp_context():
                out = self.model(
                    video=video,
                    audio=audio,
                    has_audio=has_audio,
                    duration_sec=batch["duration_sec"],
                )
                loss_dict = self.criterion(out, targets)
                scaled_loss = loss_dict["loss_total"] / float(accum)

            scaled_loss.backward()

            if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(loader):
                if self.cfg.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.train.grad_clip_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Auxiliary-loss-free non-differentiable expert bias update!
                self.model.step_moe_biases()

            for k, val in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(val.detach().item())
            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def calibrate_on_pristine(self, loader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        """Collect raw track logits on authentic validation videos and fit PristineQuantileCalibrator."""
        self.model.eval()
        vis_logits: list[torch.Tensor] = []
        aud_logits: list[torch.Tensor] = []

        for batch in loader:
            targets: BatchTargets = batch["targets"]
            real_mask = targets.video_label < 0.5
            if not bool(real_mask.any().item()):
                continue

            video = batch["video"][real_mask].to(self.device)
            audio = batch["audio"][real_mask].to(self.device)
            has_audio = batch["has_audio"][real_mask].to(self.device)

            with self._amp_context():
                out = self.model(
                    video=video,
                    audio=audio,
                    has_audio=has_audio,
                    duration_sec=batch["duration_sec"],
                )
            vis_logits.append(out.visual_track_logit.detach().cpu())
            if bool(has_audio.any().item()):
                aud_logits.append(out.audio_track_logit[has_audio].detach().cpu())

        if not vis_logits:
            raise ValueError("No pristine (video_label==0) samples found in calibration loader.")

        v_cat = torch.cat(vis_logits, dim=0)
        a_cat = torch.cat(aud_logits, dim=0) if aud_logits else None
        self.model.calibrator.fit(v_cat, a_cat)
        return {
            "visual_thresh": float(self.model.calibrator.visual_thresh.item()),
            "audio_thresh": float(self.model.calibrator.audio_thresh.item()),
        }

    def run_two_stage_curriculum(
        self,
        stage1_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]],
        stage2_loader: DataLoader[dict[str, Any]] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Execute Stage-1 unified training + optional Stage-2 high-res mask/interval refinement + calibration."""
        from pathlib import Path

        history: list[dict[str, Any]] = []
        s1_epochs = max(1, self.cfg.train.epochs_stage1)
        scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=s1_epochs, eta_min=self.cfg.train.lr * 0.05
        )

        for ep in range(1, s1_epochs + 1):
            losses = self.train_one_epoch(stage1_loader)
            scheduler_s1.step()
            history.append({"stage": 1, "epoch": ep, **losses})

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"config": self.cfg, "model_state_dict": self.model.state_dict()},
                out_p / "host_moe_stage1.pt",
            )

        s2_epochs = self.cfg.train.epochs_stage2_highres
        if stage2_loader is not None and s2_epochs > 0:
            # Freeze Orthogonal SVD-LoRA backbone projections during Stage-2 high-res boundary/interval polish
            frozen_in_s2: list[torch.nn.Parameter] = []
            for name, param in self.model.visual_backbone.named_parameters():
                if "lora_" in name and param.requires_grad:
                    param.requires_grad_(False)
                    frozen_in_s2.append(param)

            s2_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                s2_params,
                lr=self.cfg.train.lr * 0.2,
                weight_decay=self.cfg.train.weight_decay,
            )
            scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=s2_epochs, eta_min=self.cfg.train.lr * 0.02
            )

            for ep in range(1, s2_epochs + 1):
                losses = self.train_one_epoch(stage2_loader)
                scheduler_s2.step()
                history.append({"stage": 2, "epoch": ep, **losses})

            for param in frozen_in_s2:
                param.requires_grad_(True)

        cal_info = self.calibrate_on_pristine(val_loader)
        ckpt_file: str | None = None
        if output_dir:
            out_p = Path(output_dir)
            ckpt_path = out_p / "host_moe_calibrated.pt"
            torch.save(
                {"config": self.cfg, "model_state_dict": self.model.state_dict(), "calibration": cal_info},
                ckpt_path,
            )
            ckpt_file = str(ckpt_path)

        return {"history": history, "calibration": cal_info, "checkpoint": ckpt_file}

