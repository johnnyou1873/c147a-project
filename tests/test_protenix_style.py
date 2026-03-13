import pytest
import torch

from src.models.protenix_style import ProtenixStyleNet, TriangleLinearAttention


def _make_net(**overrides: object) -> ProtenixStyleNet:
    params = {
        "residue_vocab_size": 5,
        "max_chain_embeddings": 4,
        "max_copy_embeddings": 4,
        "c_s": 8,
        "c_token": 8,
        "c_z": 4,
        "c_m": 4,
        "template_c": 4,
        "diffusion_c_a": 16,
        "num_heads": 2,
        "dropout": 0.0,
        "template_blocks": 0,
        "msa_blocks": 0,
        "gru_blocks": 0,
        "pairformer_blocks": 0,
        "diffusion_blocks": 1,
        "diffusion_steps": 2,
        "coord_step_size": 0.1,
        "triangle_attention_mode": "linear",
        "use_chunked_triangle_multiplication": True,
        "triangle_multiplication_chunk_size": 2,
        "use_templates": False,
        "use_rna_msa": False,
    }
    params.update(overrides)
    return ProtenixStyleNet(**params)


def test_ode_sampler_scales_candidate_delta_without_history() -> None:
    net = _make_net(
        diffusion_sampler="ode",
        diffusion_eta=1.0,
        diffusion_lambda=1.0,
        diffusion_gamma0=0.0,
    )
    candidate0 = torch.ones((1, 3, 3), dtype=torch.float32)
    step0 = net._apply_diffusion_sampler(candidate_delta=candidate0)

    assert torch.allclose(step0, candidate0)


def test_legacy_ode_2step_alias_maps_to_ode_sampler() -> None:
    net = _make_net(diffusion_sampler="ode_2step")

    assert net.diffusion_sampler == "ode"


def test_invalid_diffusion_sampler_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported diffusion_sampler"):
        _make_net(diffusion_sampler="invalid_sampler")


def test_forward_uses_ode_sampler_without_shape_regression() -> None:
    net = _make_net(
        diffusion_sampler="ode",
        diffusion_eta=1.0,
    )
    batch_size = 1
    seq_len = 4
    pred = net(
        residue_idx=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        chain_idx=torch.zeros((batch_size, seq_len), dtype=torch.long),
        copy_idx=torch.zeros((batch_size, seq_len), dtype=torch.long),
        resid=torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32).unsqueeze(0),
        coords=torch.randn((batch_size, seq_len, 3), dtype=torch.float32),
        mask=torch.ones((batch_size, seq_len), dtype=torch.bool),
    )

    assert pred.shape == (batch_size, seq_len, 3)
    assert torch.isfinite(pred).all()


def test_forward_supports_template_and_msa_paths() -> None:
    net = _make_net(
        c_s=12,
        c_token=12,
        c_z=6,
        c_m=4,
        template_c=4,
        diffusion_c_a=24,
        template_blocks=1,
        msa_blocks=1,
        pairformer_blocks=1,
        diffusion_blocks=1,
        use_templates=True,
        use_rna_msa=True,
    )
    batch_size = 1
    seq_len = 4
    pred = net(
        residue_idx=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        chain_idx=torch.zeros((batch_size, seq_len), dtype=torch.long),
        copy_idx=torch.zeros((batch_size, seq_len), dtype=torch.long),
        resid=torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32).unsqueeze(0),
        coords=torch.randn((batch_size, seq_len, 3), dtype=torch.float32),
        mask=torch.ones((batch_size, seq_len), dtype=torch.bool),
        template_topk_coords=torch.randn((batch_size, 2, seq_len, 3), dtype=torch.float32),
        template_topk_mask=torch.ones((batch_size, 2, seq_len), dtype=torch.bool),
        template_topk_valid=torch.tensor([[True, False]], dtype=torch.bool),
        template_topk_residue_idx=torch.tensor([[[0, 1, 2, 3], [0, 0, 0, 0]]], dtype=torch.long),
        rna_msa_tokens=torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.long),
        rna_msa_mask=torch.ones((batch_size, 2, seq_len), dtype=torch.bool),
        rna_msa_row_valid=torch.ones((batch_size, 2), dtype=torch.bool),
    )

    assert net.c_s_inputs == net.input_embedder.c_s_inputs
    assert pred.shape == (batch_size, seq_len, 3)
    assert torch.isfinite(pred).all()


def test_forward_can_return_corrected_template_candidates() -> None:
    net = _make_net(
        c_s=12,
        c_token=12,
        c_z=6,
        c_m=4,
        template_c=4,
        diffusion_c_a=24,
        template_blocks=1,
        pairformer_blocks=1,
        diffusion_blocks=1,
        use_templates=True,
    )
    output = net(
        residue_idx=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        chain_idx=torch.zeros((1, 4), dtype=torch.long),
        copy_idx=torch.zeros((1, 4), dtype=torch.long),
        resid=torch.linspace(0.0, 1.0, steps=4, dtype=torch.float32).unsqueeze(0),
        coords=torch.randn((1, 4, 3), dtype=torch.float32),
        mask=torch.ones((1, 4), dtype=torch.bool),
        template_topk_coords=torch.randn((1, 2, 4, 3), dtype=torch.float32),
        template_topk_mask=torch.ones((1, 2, 4), dtype=torch.bool),
        template_topk_valid=torch.tensor([[True, True]], dtype=torch.bool),
        template_topk_residue_idx=torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.long),
        return_aux_outputs=True,
    )

    assert isinstance(output, dict)
    assert output["coords"].shape == (1, 4, 3)
    assert output["corrected_template_topk_coords"].shape == (1, 2, 4, 3)
    assert output["corrected_template_topk_valid"].shape == (1, 2)
    assert torch.isfinite(output["corrected_template_topk_coords"]).all()


def test_forward_clamps_out_of_range_chain_and_copy_ids() -> None:
    net = _make_net()
    pred = net(
        residue_idx=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        chain_idx=torch.tensor([[0, 9, 10, 11]], dtype=torch.long),
        copy_idx=torch.tensor([[0, 8, 12, 20]], dtype=torch.long),
        resid=torch.linspace(0.0, 1.0, steps=4, dtype=torch.float32).unsqueeze(0),
        coords=torch.randn((1, 4, 3), dtype=torch.float32),
        mask=torch.ones((1, 4), dtype=torch.bool),
    )

    assert pred.shape == (1, 4, 3)
    assert torch.isfinite(pred).all()


def test_pairformer_uses_triangle_linear_attention_and_chunked_updates() -> None:
    net = _make_net(pairformer_blocks=1)

    assert isinstance(net.pairformer_stack.blocks[0].tri_att_start, TriangleLinearAttention)
    assert net.pairformer_stack.blocks[0].use_chunked_triangle_multiplication is True
