from functools import partial

import pytest
import torch

from src.models.protenix_style_module import ProtenixStyleLitModule


class _DummyNet(torch.nn.Module):
    def forward(self, coords: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return coords


class _DummyTemplateNet(torch.nn.Module):
    def __init__(
        self,
        *,
        corrected_template_topk_coords: torch.Tensor,
        corrected_template_topk_valid: torch.Tensor,
        corrected_template_topk_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.corrected_template_topk_coords = corrected_template_topk_coords
        self.corrected_template_topk_valid = corrected_template_topk_valid
        self.corrected_template_topk_mask = corrected_template_topk_mask

    def forward(self, coords: torch.Tensor, return_aux_outputs: bool = False, **_: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if not return_aux_outputs:
            return coords
        valid_f = self.corrected_template_topk_valid.to(dtype=self.corrected_template_topk_coords.dtype).unsqueeze(-1).unsqueeze(-1)
        best_coords = (self.corrected_template_topk_coords * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        return {
            "coords": best_coords,
            "corrected_template_topk_coords": self.corrected_template_topk_coords,
            "corrected_template_topk_valid": self.corrected_template_topk_valid,
            "corrected_template_topk_mask": self.corrected_template_topk_mask,
        }


def _make_module(use_permutation_aware_metric: bool = True) -> ProtenixStyleLitModule:
    return ProtenixStyleLitModule(
        net=_DummyNet(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
        compile=False,
        loss_mode="one_minus_tm",
        use_permutation_aware_metric=use_permutation_aware_metric,
    )


def _swapped_chain_example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [10.0, 3.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    pred = target[:, [3, 4, 5, 0, 1, 2], :]
    mask = torch.ones((1, 6), dtype=torch.bool)
    chain_idx = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    copy_idx = torch.zeros((1, 6), dtype=torch.long)
    return pred, target, mask, chain_idx, copy_idx


def test_eval_tm_metric_is_invariant_to_swapped_chains() -> None:
    module = _make_module(use_permutation_aware_metric=True)
    module.eval()

    pred, target, mask, chain_idx, copy_idx = _swapped_chain_example()
    template_mask = torch.zeros((1, 6), dtype=torch.bool)

    _, tm_score = module._masked_losses(
        pred=pred,
        target=target,
        mask=mask,
        input_coords=target,
        template_coords=target,
        template_mask=template_mask,
        chain_idx=chain_idx,
        copy_idx=copy_idx,
    )

    assert tm_score.item() == pytest.approx(1.0, abs=1e-6)


def test_eval_tm_metric_penalizes_swapped_chains_without_permutation_matching() -> None:
    module = _make_module(use_permutation_aware_metric=False)
    module.eval()

    pred, target, mask, chain_idx, copy_idx = _swapped_chain_example()
    template_mask = torch.zeros((1, 6), dtype=torch.bool)

    _, tm_score = module._masked_losses(
        pred=pred,
        target=target,
        mask=mask,
        input_coords=target,
        template_coords=target,
        template_mask=template_mask,
        chain_idx=chain_idx,
        copy_idx=copy_idx,
    )

    assert tm_score.item() < 0.95


def test_template_tm_score_uses_best_of_five_max() -> None:
    module = _make_module(use_permutation_aware_metric=True)
    module.eval()

    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    stretched = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    template_topk_coords = torch.stack(
        [
            target[0],
            stretched[0],
            torch.zeros_like(target[0]),
            torch.zeros_like(target[0]),
            torch.zeros_like(target[0]),
        ],
        dim=0,
    ).unsqueeze(0)
    template_topk_valid = torch.tensor([[True, True, False, False, False]], dtype=torch.bool)
    template_topk_mask = torch.zeros((1, 5, 3), dtype=torch.bool)
    template_topk_mask[:, :2, :] = True
    batch = {
        "target_coords": target,
        "mask": torch.ones((1, 3), dtype=torch.bool),
        "chain_idx": torch.zeros((1, 3), dtype=torch.long),
        "copy_idx": torch.zeros((1, 3), dtype=torch.long),
        "template_topk_coords": template_topk_coords,
        "template_topk_valid": template_topk_valid,
        "template_topk_mask": template_topk_mask,
    }

    expected = torch.stack(
        [
            module._tm_score(target[0], target[0]),
            module._tm_score(stretched[0], target[0]),
        ]
    ).amax()

    template_tm_score = module._template_tm_score(batch)

    assert template_tm_score.item() == pytest.approx(expected.item(), abs=1e-6)


def test_model_step_uses_best_corrected_template_candidate() -> None:
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    stretched = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    corrected_template_topk_coords = torch.stack([stretched[0], target[0]], dim=0).unsqueeze(0)
    corrected_template_topk_valid = torch.tensor([[True, True]], dtype=torch.bool)
    corrected_template_topk_mask = torch.ones((1, 2, 3), dtype=torch.bool)
    module = ProtenixStyleLitModule(
        net=_DummyTemplateNet(
            corrected_template_topk_coords=corrected_template_topk_coords,
            corrected_template_topk_valid=corrected_template_topk_valid,
            corrected_template_topk_mask=corrected_template_topk_mask,
        ),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
        compile=False,
        loss_mode="one_minus_tm",
        use_permutation_aware_metric=True,
    )
    module.eval()

    batch = {
        "residue_idx": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "chain_idx": torch.zeros((1, 3), dtype=torch.long),
        "copy_idx": torch.zeros((1, 3), dtype=torch.long),
        "resid": torch.linspace(0.0, 1.0, steps=3, dtype=torch.float32).unsqueeze(0),
        "coords": stretched,
        "mask": torch.ones((1, 3), dtype=torch.bool),
        "target_coords": target,
        "template_coords": stretched,
        "template_mask": torch.ones((1, 3), dtype=torch.bool),
        "template_topk_coords": corrected_template_topk_coords,
        "template_topk_valid": corrected_template_topk_valid,
        "template_topk_mask": corrected_template_topk_mask,
    }

    loss, tm_score, template_tm_score = module.model_step(batch)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert tm_score.item() == pytest.approx(1.0, abs=1e-6)
    assert template_tm_score.item() == pytest.approx(1.0, abs=1e-6)


def test_masked_losses_ignore_unsupervised_target_positions() -> None:
    module = _make_module(use_permutation_aware_metric=True)
    module.eval()

    pred = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [100.0, 100.0, 100.0],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((1, 4), dtype=torch.bool)
    target_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    template_mask = torch.zeros((1, 4), dtype=torch.bool)

    _, tm_score = module._masked_losses(
        pred=pred,
        target=target,
        mask=mask,
        input_coords=target,
        template_coords=target,
        template_mask=template_mask,
        chain_idx=torch.zeros((1, 4), dtype=torch.long),
        copy_idx=torch.zeros((1, 4), dtype=torch.long),
        target_mask=target_mask,
    )

    assert tm_score.item() == pytest.approx(1.0, abs=1e-6)
