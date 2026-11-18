"""Unit tests for HoST-MoE architectural invariants, shapes, gradients, and silent fallback."""

from __future__ import annotations

from pathlib import Path
import torch

from host_forensics.cli import _build_compact_smoke_config
from host_forensics.config import HoSTConfig, SHDCConfig
from host_forensics.models.backbones import OrthogonalSVDLinear
from host_forensics.models.host_model import HoSTForensicsModel
from host_forensics.models.mn_stem import BayarConv2d
from host_forensics.models.shdc import SelfReferentialHostConditioning
from host_forensics.models.tri_stmoe import BiasBalancedTopKRouter


def test_bayar_zero_dc_constraint_before_and_after_step() -> None:
    torch.manual_seed(0)
    layer = BayarConv2d(in_channels=3, out_channels=8, kernel_size=5)

    w0 = layer.constrained_weight()
    assert w0.shape == (8, 3, 5, 5)
    assert torch.allclose(w0[:, :, 2, 2], torch.full((8, 3), -1.0), atol=1e-6)
    kernel_sums = w0.sum(dim=(-2, -1))
    assert torch.allclose(kernel_sums, torch.zeros_like(kernel_sums), atol=1e-6)

    # Constant-DC RGB frame should cancel to zero in interior pixels
    const_frame = torch.full((2, 3, 16, 16), 0.73)
    out_const = layer(const_frame)
    interior = out_const[:, :, 2:-2, 2:-2]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)

    # Take an optimization step on noisy input and verify zero-DC still holds strictly
    opt = torch.optim.Adam(layer.parameters(), lr=5e-2)
    noisy = torch.randn(2, 3, 16, 16)
    loss = layer(noisy).pow(2).mean()
    loss.backward()
    opt.step()

    w1 = layer.constrained_weight()
    assert torch.allclose(w1[:, :, 2, 2], torch.full((8, 3), -1.0), atol=1e-6)
    assert torch.allclose(w1.sum(dim=(-2, -1)), torch.zeros(8, 3), atol=1e-6)


def test_orthogonal_svd_lora_subspace_orthogonality() -> None:
    torch.manual_seed(1)
    base = torch.nn.Linear(64, 48, bias=True)
    ortho = OrthogonalSVDLinear(base, rank=8, principal_rank=16)

    # Populate non-zero lora_b (starts at 0 for identity init) to verify projection math
    with torch.no_grad():
        ortho.lora_b.normal_(mean=0.0, std=0.2)
        ortho.lora_a.normal_(mean=0.0, std=0.2)

    dw = ortho.delta_weight_orthogonal()
    assert dw.shape == (48, 64)

    # Verify U_k^T @ Delta_W == 0 and Delta_W @ V_k == 0
    left_proj = ortho.u_k.T @ dw
    right_proj = dw @ ortho.v_k
    assert torch.norm(left_proj).item() < 1e-5
    assert torch.norm(right_proj).item() < 1e-5

    # Verify frozen base weights receive no gradients while LoRA factors do
    x = torch.randn(4, 10, 64)
    y = ortho(x)
    y.sum().backward()
    assert ortho.weight0.grad is None
    assert ortho.bias0 is not None and ortho.bias0.grad is None
    assert ortho.lora_a.grad is not None and torch.isfinite(ortho.lora_a.grad).all()
    assert ortho.lora_b.grad is not None and torch.isfinite(ortho.lora_b.grad).all()


def test_bias_balanced_topk_router_updates_without_aux_loss() -> None:
    torch.manual_seed(2)
    router = BiasBalancedTopKRouter(dim=16, num_experts=4, top_k=2, bias_update_rate=0.05)
    router.train()

    # Force gate weights to strongly prefer experts 0 and 1 over experts 2 and 3
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 5.0
        router.gate.weight[1, 0] = 4.0
        router.gate.weight[2, 0] = -3.0
        router.gate.weight[3, 0] = -4.0

    tokens = torch.ones(2, 8, 16)
    weights = router(tokens)
    assert weights.shape == (2, 8, 4)
    # Exactly top_k=2 active experts per token, summing to 1.0
    assert torch.all((weights > 0).sum(dim=-1) == 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-6)

    router.update_bias()
    # Overloaded experts (0, 1) get negative bias shift; underloaded (2, 3) get positive shift
    assert router.routing_bias[0].item() < 0.0
    assert router.routing_bias[1].item() < 0.0
    assert router.routing_bias[2].item() > 0.0
    assert router.routing_bias[3].item() > 0.0


def test_shdc_channelwise_affine_shift_invariance() -> None:
    torch.manual_seed(3)
    shdc = SelfReferentialHostConditioning(dim=32, cfg=SHDCConfig(anchor_hidden_dim=32, sharpening_beta=2.0))
    shdc.eval()

    z = torch.randn(2, 4, 16, 32)
    # Arbitrary positive per-sample, per-channel scale a > 0 and arbitrary shift b
    scale = torch.empty(2, 1, 1, 32).uniform_(0.6, 3.2)
    shift = torch.empty(2, 1, 1, 32).uniform_(-8.0, 8.0)
    z_transformed = z * scale + shift

    out_orig = shdc(z, strict_affine_invariance=True)
    out_aff = shdc(z_transformed, strict_affine_invariance=True)

    assert torch.allclose(out_orig.anchor_probs, out_aff.anchor_probs, atol=1e-5)
    assert torch.allclose(out_orig.tokens, out_aff.tokens, atol=1e-5)


def test_host_model_end_to_end_audiovisual_and_silent_fallback(tmp_path: Path) -> None:
    torch.manual_seed(4)
    cfg = _build_compact_smoke_config()

    # Verify YAML serialization round-trip
    yaml_path = tmp_path / "roundtrip.yaml"
    cfg.to_yaml(yaml_path)
    cfg_reloaded = HoSTConfig.from_yaml(yaml_path)
    assert cfg_reloaded.backbone.adapted_layers == cfg.backbone.adapted_layers

    model = HoSTForensicsModel(cfg_reloaded)
    summary = model.parameter_summary()
    assert summary["trainable_params"] > 0
    assert summary["trainable_params"] < summary["total_params"]

    # Case 1: Mixed Audio-Visual + Silent batch
    video = torch.randn(2, 4, 3, 128, 128)
    audio = torch.randn(2, 32000)
    has_audio = torch.tensor([True, False])

    out_av = model(video=video, audio=audio, has_audio=has_audio, duration_sec=2.0)
    assert out_av.video_logit.shape == (2,)
    assert out_av.mask_logits.shape == (2, 4, 128, 128)
    assert out_av.audio_track_logit[1].item() == -20.0  # Masked silent sample

    loss = out_av.video_logit.sum() + out_av.mask_logits.mean()
    loss.backward()
    model.step_moe_biases()

    # Frozen ViT patch_embed & frozen audio conv must have None gradients
    for p in model.visual_backbone.patch_embed.parameters():
        assert p.grad is None
    for p in model.audio_backbone.frozen_conv.parameters():
        assert p.grad is None

    # Trainable adapters and stem must receive finite gradients
    assert model.mn_stem.bayar.raw_off_weights.grad is not None
    assert torch.isfinite(model.mn_stem.bayar.raw_off_weights.grad).all()

    # Case 2: Purely silent inference (`audio=None`)
    preds_silent = model.predict_structured(video=video, audio=None, has_audio=None, duration_sec=2.0)
    assert len(preds_silent) == 2
    assert preds_silent[0]["has_audio"] is False
    assert preds_silent[0]["audio_intervals"] == []


def test_cli_entrypoints() -> None:
    from host_forensics.cli import main

    assert main(["info", "--compact"]) == 0
    assert main(["smoke-train", "--epochs", "1", "--num-samples", "4", "--device", "cpu"]) == 0
    assert main(["infer", "--silent", "--device", "cpu"]) == 0

