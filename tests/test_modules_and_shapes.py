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


def test_hf_visual_in_stream_hooks_and_wavlm_audio_backbone() -> None:
    from types import SimpleNamespace
    import torch.nn as nn
    from host_forensics.config import BackboneConfig, TriSTMoEConfig
    from host_forensics.models.backbones import FrozenAudioBackbone, MultiPriorVisualBackbone

    torch.manual_seed(7)

    class _MockHFBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            h = self.norm(x)
            out = x + self.o_proj(self.q_proj(h) + self.k_proj(h) + self.v_proj(h)) * 0.1
            return (out, None)

    class _MockDINOv3Model(nn.Module):
        """Mimics HF DINOv3/ViT with 1 CLS + 4 register tokens + tuple block outputs."""

        def __init__(self, dim: int = 64, patch_size: int = 16, num_layers: int = 4) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
            self.cls_and_reg = nn.Parameter(torch.randn(1, 5, dim) * 0.02)
            self.layer = nn.ModuleList([_MockHFBlock(dim) for _ in range(num_layers)])

        def forward(
            self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
        ) -> SimpleNamespace:
            _ = interpolate_pos_encoding
            patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls_and_reg.expand(patches.shape[0], -1, -1), patches], dim=1)
            for blk in self.layer:
                x = blk(x)[0]
            return SimpleNamespace(last_hidden_state=x)

    vis_cfg = BackboneConfig(
        visual_backend="hf",
        img_size=64,
        patch_size=16,
        visual_embed_dim=64,
        visual_num_layers=4,
        tap_layers=(2, 4),
        adapted_layers=(1, 3),
        use_ortho_lora=True,
        lora_rank=4,
        principal_subspace_rank=8,
    )
    moe_cfg = TriSTMoEConfig(bottleneck_dim=32, num_experts_per_pool=2, top_k=1, num_heads=4)

    mock_dino = _MockDINOv3Model(dim=64, patch_size=16, num_layers=4)
    vis_backbone = MultiPriorVisualBackbone(vis_cfg, moe_cfg, mn_res_dim=32, hf_model=mock_dino)
    vis_backbone.train(True)

    # Frozen HF backbone stays in eval(), while MoE adapters are in train()
    assert vis_backbone._hf_model is not None and not vis_backbone._hf_model.training
    assert vis_backbone.moe_adapters.training
    assert isinstance(mock_dino.layer[0].q_proj, OrthogonalSVDLinear)
    assert isinstance(mock_dino.layer[0].v_proj, OrthogonalSVDLinear)
    assert not isinstance(mock_dino.layer[0].k_proj, OrthogonalSVDLinear)

    # Verify Stage-1 (64x64 -> P=16) and Stage-2 (96x96 -> P=36) forward + in-stream propagation
    vid_s1 = torch.randn(2, 3, 3, 64, 64)
    res_s1 = torch.randn(2, 3, 16, 32)
    vis_backbone.eval()
    fused_a, skip_a = vis_backbone(vid_s1, res_s1)
    assert fused_a.shape == (2, 3, 16, 64)
    assert skip_a.shape == (2, 3, 16, 32)

    # Perturbing adapter at block 1 MUST alter tapped features at block 2 (proving in-stream hook!)
    with torch.no_grad():
        for p in vis_backbone.moe_adapters["1"].parameters():
            p.add_(0.5)
    fused_b, skip_b = vis_backbone(vid_s1, res_s1)
    assert not torch.allclose(skip_a, skip_b, atol=1e-4)
    assert not torch.allclose(fused_a, fused_b, atol=1e-4)

    vid_s2 = torch.randn(2, 3, 3, 96, 96)
    res_s2 = torch.randn(2, 3, 36, 32)
    fused_s2, _ = vis_backbone(vid_s2, res_s2)
    assert fused_s2.shape == (2, 3, 36, 64)

    # Verify unconstrained LoRA toggle (Ablation A1: orthogonal=False)
    base_lin = nn.Linear(32, 32)
    lora_std = OrthogonalSVDLinear(base_lin, rank=4, principal_rank=8, orthogonal=False)
    with torch.no_grad():
        lora_std.lora_b.normal_(0.0, 0.3)
        lora_std.lora_a.normal_(0.0, 0.3)
    dw_std = lora_std.delta_weight_orthogonal()
    assert torch.norm(lora_std.u_k.T @ dw_std).item() > 1e-3

    # Verify HF WavLM audio backbone wrapper
    class _MockWavLM(nn.Module):
        def __init__(self, hidden_size: int = 48) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.proj = nn.Conv1d(1, hidden_size, kernel_size=8, stride=4)

        def forward(
            self, input_values: torch.Tensor, output_hidden_states: bool = True
        ) -> SimpleNamespace:
            _ = output_hidden_states
            feat = self.proj(input_values.unsqueeze(1)).transpose(1, 2)
            hs = tuple(feat * (1.0 + 0.1 * i) for i in range(5))
            return SimpleNamespace(last_hidden_state=hs[-1], hidden_states=hs)

    mock_wavlm = _MockWavLM(hidden_size=48)
    aud_backbone = FrozenAudioBackbone(vis_cfg, out_dim=32, hf_audio_model=mock_wavlm)
    aud_backbone.train(True)
    assert aud_backbone._hf_audio_model is not None and not aud_backbone._hf_audio_model.training

    wav = torch.randn(2, 16000)
    aud_tok = aud_backbone(wav, target_steps=8)
    assert aud_tok.shape == (2, 8, 32)
    aud_tok.sum().backward()
    assert mock_wavlm.proj.weight.grad is None
    assert aud_backbone.detail_proj.weight.grad is not None


def test_gpu_optimizations_lora_merging_batched_moe_and_dataloader() -> None:
    import torch.nn as nn
    from host_forensics.config import TriSTMoEConfig
    from host_forensics.data.datasets import SyntheticManipulatedVideoDataset, build_optimized_dataloader
    from host_forensics.models.tri_stmoe import TriSTMoEAdapter

    torch.manual_seed(19)

    # 1. Factorized low-rank training vs cached zero-overhead eval weight merging equivalence
    base = nn.Linear(64, 48, bias=True)
    ortho = OrthogonalSVDLinear(base, rank=8, principal_rank=16, orthogonal=True)
    with torch.no_grad():
        ortho.lora_a.normal_(0.0, 0.25)
        ortho.lora_b.normal_(0.0, 0.25)

    x = torch.randn(3, 12, 64)
    ortho.train()
    assert ortho._merged_eval_weight is None
    y_train = ortho(x)

    ortho.eval()
    y_eval = ortho(x)
    assert ortho._merged_eval_weight is not None
    assert torch.allclose(y_train, y_eval, atol=1e-5)

    # Reloading state_dict or switching back to train() must invalidate merged eval cache
    state = ortho.state_dict()
    ortho.load_state_dict(state)
    assert ortho._merged_eval_weight is None
    _ = ortho(x)
    assert ortho._merged_eval_weight is not None
    ortho.train()
    assert ortho._merged_eval_weight is None

    # 2. Batched parallel Tensor-Core expert pools & RoPE cache hit
    moe_cfg = TriSTMoEConfig(bottleneck_dim=32, num_experts_per_pool=4, top_k=2, num_heads=4, dropout=0.0)
    adapter = TriSTMoEAdapter(embed_dim=64, res_dim=32, cfg=moe_cfg)
    with torch.no_grad():
        adapter.up_proj.weight.normal_(0.0, 0.05)

    tok_in = torch.randn(2, 4, 9, 64, requires_grad=True)
    res_in = torch.randn(2, 4, 9, 32, requires_grad=True)
    tok_out = adapter(tok_in, res_in)
    assert tok_out.shape == (2, 4, 9, 64)
    assert adapter.pool_t.rope._cos_cache is not None
    tok_out.sum().backward()
    assert adapter.pool_s.w_qkv.grad is not None
    assert adapter.pool_t.w_qkv.grad is not None
    assert adapter.pool_r.w_kv.grad is not None

    # 3. Async multi-worker DataLoader builder
    ds = SyntheticManipulatedVideoDataset(length=6, num_frames=4, img_size=64, seed=5)
    loader = build_optimized_dataloader(
        dataset=ds,
        batch_size=3,
        shuffle=False,
        device="cpu",
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    assert loader.num_workers == 2
    assert loader.pin_memory is False  # Safely disabled on CPU to avoid CUDA warnings
    assert loader.persistent_workers is True
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["video"].shape == (3, 4, 3, 64, 64)



