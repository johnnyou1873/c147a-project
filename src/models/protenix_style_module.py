from __future__ import annotations

import inspect
import math
from functools import partial
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional runtime dependency
    linear_sum_assignment = None


class ProtenixStyleLitModule(LightningModule):
    """Lightning module for the active Protenix-style RNA structure model."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        loss_mode: str = "one_minus_lddt",
        use_permutation_aware_metric: bool = True,
        must_be_better_weight: float = 0.5,
        must_be_better_margin: float = 0.01,
        must_be_better_log_scale: float = 8.0,
        must_be_better_min_factor: float = 0.2,
        must_be_better_max_factor: float = 3.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.loss_mode = str(loss_mode).strip().lower()
        valid_loss_modes = {"one_minus_lddt", "improvement_focused_lddt"}
        if self.loss_mode not in valid_loss_modes:
            raise ValueError(
                f"Invalid loss_mode='{loss_mode}'. Expected one of: {sorted(valid_loss_modes)}"
            )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_lddt = MeanMetric()
        self.val_lddt = MeanMetric()
        self.test_lddt = MeanMetric()
        self.train_tm_score = MeanMetric()
        self.val_tm_score = MeanMetric()
        self.test_tm_score = MeanMetric()
        self.train_template_tm_score = MeanMetric()
        self.val_template_tm_score = MeanMetric()
        self.test_template_tm_score = MeanMetric()

    def _build_net_inputs(self, batch: Dict[str, torch.Tensor], *, return_aux_outputs: bool) -> Dict[str, torch.Tensor]:
        net_inputs: Dict[str, torch.Tensor] = {
            "residue_idx": batch["residue_idx"],
            "chain_idx": batch["chain_idx"],
            "copy_idx": batch["copy_idx"],
            "resid": batch["resid"],
            "coords": batch["coords"],
            "mask": batch["mask"],
            "template_coords": batch.get("template_coords"),
            "template_mask": batch.get("template_mask"),
            "template_topk_coords": batch.get("template_topk_coords"),
            "template_topk_mask": batch.get("template_topk_mask"),
            "template_topk_valid": batch.get("template_topk_valid"),
            "template_topk_identity": batch.get("template_topk_identity"),
            "template_topk_similarity": batch.get("template_topk_similarity"),
            "template_topk_residue_idx": batch.get("template_topk_residue_idx"),
            "template_chunk_coords": batch.get("template_chunk_coords"),
            "template_chunk_mask": batch.get("template_chunk_mask"),
            "template_chunk_start": batch.get("template_chunk_start"),
            "template_chunk_window_valid": batch.get("template_chunk_window_valid"),
            "template_chunk_valid": batch.get("template_chunk_valid"),
            "template_chunk_identity": batch.get("template_chunk_identity"),
            "template_chunk_similarity": batch.get("template_chunk_similarity"),
            "template_chunk_confidence": batch.get("template_chunk_confidence"),
            "template_chunk_source_onehot": batch.get("template_chunk_source_onehot"),
            "template_chunk_residue_idx": batch.get("template_chunk_residue_idx"),
            "rna_msa_tokens": batch.get("rna_msa_tokens"),
            "rna_msa_mask": batch.get("rna_msa_mask"),
            "rna_msa_row_valid": batch.get("rna_msa_row_valid"),
            "rna_msa_profile": batch.get("rna_msa_profile"),
            "rna_bpp_banded": batch.get("rna_bpp_banded"),
            "rna_bpp_mask": batch.get("rna_bpp_mask"),
        }
        if return_aux_outputs:
            net_inputs["return_aux_outputs"] = True
        try:
            valid_keys = set(inspect.signature(self.net.forward).parameters.keys())
            net_inputs = {k: v for k, v in net_inputs.items() if k in valid_keys}
        except (TypeError, ValueError):
            pass
        return net_inputs

    def _forward_model(self, batch: Dict[str, torch.Tensor], *, return_aux_outputs: bool) -> torch.Tensor | Dict[str, torch.Tensor]:
        return self.net(**self._build_net_inputs(batch, return_aux_outputs=return_aux_outputs))

    def _normalize_model_output(self, output: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(output):
            return {"coords": output}
        if not isinstance(output, dict) or "coords" not in output or not torch.is_tensor(output["coords"]):
            raise TypeError("Model output must be a tensor or a dict containing a tensor `coords` entry.")
        return output

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self._forward_model(batch, return_aux_outputs=False)
        if torch.is_tensor(output):
            return output
        return output["coords"]

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_lddt.reset()
        self.val_tm_score.reset()
        self.val_template_tm_score.reset()

    def _d0(self, n_res: int) -> float:
        if n_res <= 15:
            return 0.5
        return max(0.5, 1.24 * (float(n_res) - 15.0) ** (1.0 / 3.0) - 1.8)

    def _kabsch_align(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
        use_fp32_linalg = pred.device.type == "cuda" and (
            pred.dtype in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled()
        )

        if use_fp32_linalg:
            # In fp16/16-mixed on CUDA, SVD/eigh kernels for Half are not available.
            # Run the Kabsch solve in fp32 with autocast disabled, then cast back.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                pred_f = pred.to(dtype=torch.float32)
                target_f = target.to(dtype=torch.float32)

                pred_center = pred_f.mean(dim=0, keepdim=True)
                target_center = target_f.mean(dim=0, keepdim=True)
                pred_c = pred_f - pred_center
                target_c = target_f - target_center

                cov = pred_c.transpose(0, 1) @ target_c
                u, _, vh = torch.linalg.svd(cov, full_matrices=False)
                r = vh.transpose(0, 1) @ u.transpose(0, 1)
                if torch.det(r) < 0:
                    vh_fix = vh.clone()
                    vh_fix[-1, :] *= -1
                    r = vh_fix.transpose(0, 1) @ u.transpose(0, 1)

                aligned = pred_c @ r + target_center
            return aligned.to(dtype=pred.dtype)

        pred_center = pred.mean(dim=0, keepdim=True)
        target_center = target.mean(dim=0, keepdim=True)
        pred_c = pred - pred_center
        target_c = target - target_center

        cov = pred_c.transpose(0, 1) @ target_c
        u, _, vh = torch.linalg.svd(cov, full_matrices=False)
        r = vh.transpose(0, 1) @ u.transpose(0, 1)
        if torch.det(r) < 0:
            vh_fix = vh.clone()
            vh_fix[-1, :] *= -1
            r = vh_fix.transpose(0, 1) @ u.transpose(0, 1)

        aligned = pred_c @ r + target_center
        return aligned

    def _tm_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
        n_res = int(pred.shape[0])
        if n_res < 3:
            return pred.new_tensor(0.0)

        pred_aligned = self._kabsch_align(pred, target)
        dist = torch.linalg.norm(pred_aligned - target, dim=-1)
        d0 = self._d0(n_res)
        tm = torch.mean(1.0 / (1.0 + (dist / d0) ** 2))
        tm = torch.nan_to_num(tm, nan=0.0, posinf=0.0, neginf=0.0)
        return tm.clamp(min=0.0, max=1.0)

    def _lddt_score(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        cutoff: float = 15.0,
        smooth: float = 0.25,
    ) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
        n_res = int(pred.shape[0])
        if n_res < 3:
            return pred.new_tensor(0.0)

        def _compute_lddt(pred_f: torch.Tensor, target_f: torch.Tensor) -> torch.Tensor:
            pred_dist = torch.cdist(pred_f, pred_f)
            target_dist = torch.cdist(target_f, target_f)
            pair_mask = target_dist < cutoff
            pair_mask.fill_diagonal_(False)
            if not bool(pair_mask.any()):
                return pred_f.new_tensor(0.0)

            dist_error = torch.abs(pred_dist - target_dist)
            thresholds = pred_f.new_tensor((0.5, 1.0, 2.0, 4.0)).view(1, 1, 4)
            threshold_scores = torch.sigmoid((thresholds - dist_error.unsqueeze(-1)) / smooth)
            threshold_scores = threshold_scores / torch.sigmoid(thresholds / smooth)
            pair_scores = threshold_scores.clamp(min=0.0, max=1.0).mean(dim=-1)
            pair_counts = pair_mask.sum(dim=-1)
            residue_scores = (pair_scores * pair_mask).sum(dim=-1) / pair_counts.clamp(min=1)
            residue_valid = pair_counts > 0
            if not bool(residue_valid.any()):
                return pred_f.new_tensor(0.0)

            lddt = residue_scores[residue_valid].mean()
            lddt = torch.nan_to_num(lddt, nan=0.0, posinf=0.0, neginf=0.0)
            return lddt.clamp(min=0.0, max=1.0)

        use_fp32_distance = pred.device.type == "cuda" and (
            pred.dtype in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled()
        )
        if use_fp32_distance:
            with torch.autocast(device_type=pred.device.type, enabled=False):
                score = _compute_lddt(pred.to(dtype=torch.float32), target.to(dtype=torch.float32))
            return score.to(dtype=pred.dtype)

        return _compute_lddt(pred, target)

    def _assign_groups(self, cost: torch.Tensor) -> list[int]:
        """Solve row->col assignment from a square cost matrix."""
        n = int(cost.shape[0])
        if n <= 1:
            return list(range(n))

        cost_clean = torch.nan_to_num(
            cost.detach().to(dtype=torch.float64),
            nan=1.0,
            posinf=1e6,
            neginf=0.0,
        )

        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost_clean.cpu().numpy())
            assignment = [0] * n
            for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                assignment[int(r)] = int(c)
            return assignment

        # Fallback: greedy assignment if scipy is unavailable.
        assignment = [-1] * n
        used = set()
        for r in range(n):
            order = torch.argsort(cost_clean[r]).tolist()
            for c in order:
                if c not in used:
                    assignment[r] = int(c)
                    used.add(int(c))
                    break
        for r in range(n):
            if assignment[r] < 0:
                for c in range(n):
                    if c not in used:
                        assignment[r] = c
                        used.add(c)
                        break
        return assignment

    def _inverse_log_factor(self, baseline_score: torch.Tensor) -> torch.Tensor:
        """Higher factor for low baseline quality, lower (plateaued) factor for high baseline quality."""
        scale = max(float(self.hparams.must_be_better_log_scale), 1e-6)
        denom = torch.log(baseline_score * scale + math.e).clamp(min=1e-6)
        factor = 1.0 / denom
        factor = factor.clamp(
            min=float(self.hparams.must_be_better_min_factor),
            max=float(self.hparams.must_be_better_max_factor),
        )
        return factor

    def _score_tm_predictions(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        supervised_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
        permutation_aware: bool | None = None,
    ) -> torch.Tensor:
        if permutation_aware is None:
            permutation_aware = bool(self.hparams.use_permutation_aware_metric)

        with torch.no_grad():
            tm_scores_metric = []
            for b in range(pred.shape[0]):
                valid = supervised_mask[b]
                pred_b = pred[b, valid].to(dtype=torch.float32)
                target_b = target[b, valid].to(dtype=torch.float32)
                chain_b = chain_idx[b, valid]
                copy_b = copy_idx[b, valid]
                if pred_b.shape[0] < 3:
                    continue

                if permutation_aware:
                    target_eval = self._permutation_aware_target(pred_b, target_b, chain_b, copy_b)
                else:
                    target_eval = target_b

                tm_scores_metric.append(self._tm_score(pred_b, target_eval))

            if tm_scores_metric:
                return torch.stack(tm_scores_metric).mean().to(dtype=pred.dtype)
            return pred.new_tensor(0.0)

    def _permutation_aware_target(
        self,
        pred_b: torch.Tensor,
        target_b: torch.Tensor,
        chain_idx_b: torch.Tensor,
        copy_idx_b: torch.Tensor,
    ) -> torch.Tensor:
        """Reorder target chain/copy blocks to be robust to chain or copy swaps."""
        group_lists: Dict[tuple[int, int], list[int]] = {}
        for i in range(pred_b.shape[0]):
            key = (int(chain_idx_b[i].item()), int(copy_idx_b[i].item()))
            group_lists.setdefault(key, []).append(i)

        if len(group_lists) <= 1:
            return target_b

        groups = []
        for (chain_id, copy_id), idx_list in group_lists.items():
            idx_tensor = torch.tensor(idx_list, device=pred_b.device, dtype=torch.long)
            groups.append((chain_id, copy_id, idx_tensor))

        target_perm = target_b.clone()
        groups.sort(key=lambda item: (item[0], item[1]))
        indices = [group[2] for group in groups]
        n = len(indices)
        cost = pred_b.new_full((n, n), fill_value=1.0)

        for i in range(n):
            idx_i = indices[i]
            p = pred_b[idx_i].to(dtype=torch.float32)
            for j in range(n):
                idx_j = indices[j]
                if idx_i.numel() != idx_j.numel() or idx_i.numel() < 3:
                    continue
                t = target_b[idx_j].to(dtype=torch.float32)
                tm_ij = self._tm_score(p, t)
                cost[i, j] = (1.0 - tm_ij).to(dtype=cost.dtype)

        assignment = self._assign_groups(cost)
        for i, j in enumerate(assignment):
            idx_i = indices[i]
            idx_j = indices[j]
            if idx_i.numel() != idx_j.numel():
                continue
            target_perm[idx_i] = target_b[idx_j]

        return target_perm

    def _masked_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        input_coords: torch.Tensor,
        template_coords: torch.Tensor,
        template_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lddt_scores_train = []
        improvement_terms = []
        use_improvement = self.loss_mode == "improvement_focused_lddt"
        supervised_mask = mask if target_mask is None else (mask & target_mask)
        for b in range(pred.shape[0]):
            valid = supervised_mask[b]
            pred_b = pred[b, valid]
            target_b = target[b, valid]
            if pred_b.shape[0] < 3:
                continue
            lddt_pred = self._lddt_score(pred_b, target_b)
            lddt_scores_train.append(lddt_pred)

            if use_improvement:
                with torch.no_grad():
                    input_b = input_coords[b, valid].to(dtype=torch.float32)
                    target_b_f = target_b.to(dtype=torch.float32)
                    baseline_lddt = self._lddt_score(input_b, target_b_f)

                    template_mask_b = template_mask[b, valid].unsqueeze(-1)
                    if bool(template_mask_b.any()):
                        template_passthrough_b = torch.where(
                            template_mask_b,
                            template_coords[b, valid].to(dtype=torch.float32),
                            input_b,
                        )
                        baseline_lddt = torch.maximum(
                            baseline_lddt,
                            self._lddt_score(template_passthrough_b, target_b_f),
                        )

                    factor = self._inverse_log_factor(baseline_lddt)
                    margin = float(self.hparams.must_be_better_margin)
                    improvement_gap = torch.relu(baseline_lddt + margin - lddt_pred)
                    improvement_terms.append(factor.to(dtype=improvement_gap.dtype) * improvement_gap)

        if lddt_scores_train:
            lddt_train = torch.stack(lddt_scores_train).mean()
        else:
            lddt_train = pred.sum() * 0.0

        main_loss = 1.0 - lddt_train
        if use_improvement:
            if improvement_terms:
                improvement_loss = torch.stack(improvement_terms).mean()
            else:
                improvement_loss = pred.new_tensor(0.0)
            loss = main_loss + float(self.hparams.must_be_better_weight) * improvement_loss
        else:
            loss = main_loss

        lddt_metric = lddt_train.detach()
        tm_metric = self._score_tm_predictions(
            pred=pred,
            target=target,
            supervised_mask=supervised_mask,
            chain_idx=chain_idx,
            copy_idx=copy_idx,
            permutation_aware=None if not self.training else False,
        )

        return loss, lddt_metric, tm_metric

    def _score_template_candidates(
        self,
        *,
        topk_coords: Optional[torch.Tensor],
        topk_valid: Optional[torch.Tensor],
        topk_mask: Optional[torch.Tensor],
        target: torch.Tensor,
        supervised_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
        permutation_aware: bool | None = None,
    ) -> torch.Tensor:
        if topk_coords is None or topk_coords.numel() == 0:
            return target.new_tensor(0.0)
        if permutation_aware is None:
            permutation_aware = bool(self.hparams.use_permutation_aware_metric)

        topk_coords_f = topk_coords.to(dtype=torch.float32)
        topk_valid_b = (
            topk_valid.to(device=topk_coords_f.device, dtype=torch.bool)
            if topk_valid is not None
            else torch.ones(topk_coords_f.shape[:2], device=topk_coords_f.device, dtype=torch.bool)
        )
        topk_mask_b = (
            topk_mask.to(device=topk_coords_f.device, dtype=torch.bool)
            if topk_mask is not None
            else None
        )

        best_scores = []
        for b in range(topk_coords_f.shape[0]):
            candidate_scores = []
            for k in range(topk_coords_f.shape[1]):
                if not bool(topk_valid_b[b, k].item()):
                    continue
                valid = supervised_mask[b]
                if topk_mask_b is not None:
                    valid = valid & topk_mask_b[b, k]
                if int(valid.sum().item()) < 3:
                    continue

                template_b = topk_coords_f[b, k, valid]
                target_b = target[b, valid].to(dtype=torch.float32)
                chain_b = chain_idx[b, valid]
                copy_b = copy_idx[b, valid]
                if permutation_aware:
                    target_eval = self._permutation_aware_target(template_b, target_b, chain_b, copy_b)
                else:
                    target_eval = target_b
                candidate_scores.append(self._tm_score(template_b, target_eval))

            if candidate_scores:
                best_scores.append(torch.stack(candidate_scores).amax())

        if best_scores:
            return torch.stack(best_scores).mean().to(dtype=target.dtype)
        return target.new_tensor(0.0)

    def _masked_template_candidate_losses(
        self,
        *,
        pred_topk_coords: torch.Tensor,
        pred_topk_valid: Optional[torch.Tensor],
        pred_topk_mask: Optional[torch.Tensor],
        raw_template_topk_coords: Optional[torch.Tensor],
        raw_template_topk_valid: Optional[torch.Tensor],
        raw_template_topk_mask: Optional[torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
        input_coords: torch.Tensor,
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_improvement = self.loss_mode == "improvement_focused_lddt"
        supervised_mask = mask if target_mask is None else (mask & target_mask)
        pred_valid = (
            pred_topk_valid.to(device=pred_topk_coords.device, dtype=torch.bool)
            if pred_topk_valid is not None
            else torch.ones(pred_topk_coords.shape[:2], device=pred_topk_coords.device, dtype=torch.bool)
        )
        pred_mask = (
            pred_topk_mask.to(device=pred_topk_coords.device, dtype=torch.bool)
            if pred_topk_mask is not None
            else None
        )

        best_lddt_train = []
        improvement_terms = []
        for b in range(pred_topk_coords.shape[0]):
            target_b = target[b]
            candidate_scores = []
            for k in range(pred_topk_coords.shape[1]):
                if not bool(pred_valid[b, k].item()):
                    continue
                valid = supervised_mask[b]
                if pred_mask is not None:
                    valid = valid & pred_mask[b, k]
                if int(valid.sum().item()) < 3:
                    continue
                candidate_scores.append(self._lddt_score(pred_topk_coords[b, k, valid], target_b[valid]))

            if not candidate_scores:
                continue

            sample_best_lddt = torch.stack(candidate_scores).amax()
            best_lddt_train.append(sample_best_lddt)

            if use_improvement:
                with torch.no_grad():
                    valid = supervised_mask[b]
                    input_b = input_coords[b, valid].to(dtype=torch.float32)
                    target_b_f = target_b[valid].to(dtype=torch.float32)
                    baseline_lddt = self._lddt_score(input_b, target_b_f)

                    if raw_template_topk_coords is not None and raw_template_topk_coords.numel() > 0:
                        raw_coords = raw_template_topk_coords.to(device=target.device, dtype=torch.float32)
                        raw_valid = (
                            raw_template_topk_valid.to(device=target.device, dtype=torch.bool)
                            if raw_template_topk_valid is not None
                            else torch.ones(raw_coords.shape[:2], device=target.device, dtype=torch.bool)
                        )
                        raw_mask = (
                            raw_template_topk_mask.to(device=target.device, dtype=torch.bool)
                            if raw_template_topk_mask is not None
                            else None
                        )
                        for k in range(raw_coords.shape[1]):
                            if not bool(raw_valid[b, k].item()):
                                continue
                            if raw_mask is None:
                                baseline_candidate = raw_coords[b, k, valid]
                            else:
                                baseline_candidate = torch.where(
                                    raw_mask[b, k, valid].unsqueeze(-1),
                                    raw_coords[b, k, valid],
                                    input_b,
                                )
                            baseline_lddt = torch.maximum(
                                baseline_lddt,
                                self._lddt_score(baseline_candidate, target_b_f),
                            )

                    factor = self._inverse_log_factor(baseline_lddt)
                    margin = float(self.hparams.must_be_better_margin)
                    improvement_gap = torch.relu(baseline_lddt + margin - sample_best_lddt)
                    improvement_terms.append(factor.to(dtype=improvement_gap.dtype) * improvement_gap)

        if best_lddt_train:
            lddt_train = torch.stack(best_lddt_train).mean()
        else:
            lddt_train = pred_topk_coords.sum() * 0.0

        main_loss = 1.0 - lddt_train
        if use_improvement:
            if improvement_terms:
                improvement_loss = torch.stack(improvement_terms).mean()
            else:
                improvement_loss = pred_topk_coords.new_tensor(0.0)
            loss = main_loss + float(self.hparams.must_be_better_weight) * improvement_loss
        else:
            loss = main_loss

        lddt_metric = lddt_train.detach()
        tm_metric = self._score_template_candidates(
            topk_coords=pred_topk_coords,
            topk_valid=pred_topk_valid,
            topk_mask=pred_topk_mask,
            target=target,
            supervised_mask=supervised_mask,
            chain_idx=chain_idx,
            copy_idx=copy_idx,
            permutation_aware=None if not self.training else False,
        ).to(dtype=pred_topk_coords.dtype)

        return loss, lddt_metric, tm_metric

    def _template_tm_score(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        model_output: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del model_output
        target = batch["target_coords"]
        mask = batch["mask"]
        target_mask = batch.get("target_mask", mask)
        supervised_mask = mask & target_mask
        chain_idx = batch["chain_idx"]
        copy_idx = batch["copy_idx"]
        template_topk_coords = batch.get("template_topk_coords")
        template_topk_valid = batch.get("template_topk_valid")
        template_topk_mask = batch.get("template_topk_mask")
        template_coords = batch.get("template_coords")
        template_mask = batch.get("template_mask")

        with torch.no_grad():
            if template_topk_coords is not None and template_topk_coords.numel() > 0:
                return self._score_template_candidates(
                    topk_coords=template_topk_coords,
                    topk_valid=template_topk_valid,
                    topk_mask=template_topk_mask,
                    target=target,
                    supervised_mask=supervised_mask,
                    chain_idx=chain_idx,
                    copy_idx=copy_idx,
                )

            elif template_coords is not None and template_coords.numel() > 0:
                return self._score_template_candidates(
                    topk_coords=template_coords.unsqueeze(1),
                    topk_valid=torch.ones(template_coords.shape[0], 1, device=template_coords.device, dtype=torch.bool),
                    topk_mask=None if template_mask is None else template_mask.unsqueeze(1),
                    target=target,
                    supervised_mask=supervised_mask,
                    chain_idx=chain_idx,
                    copy_idx=copy_idx,
                )

            return target.new_tensor(0.0)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = self._normalize_model_output(self._forward_model(batch, return_aux_outputs=True))
        pred = model_output["coords"]
        target = batch["target_coords"]
        mask = batch["mask"]
        target_mask = batch.get("target_mask", mask)
        input_coords = batch["coords"]
        template_coords = batch["template_coords"]
        template_mask = batch["template_mask"]
        chain_idx = batch["chain_idx"]
        copy_idx = batch["copy_idx"]
        corrected_topk_coords = model_output.get("candidate_topk_coords")
        corrected_topk_valid = model_output.get("candidate_topk_valid")
        corrected_topk_mask = model_output.get("candidate_topk_mask")
        if corrected_topk_coords is None:
            corrected_topk_coords = model_output.get("corrected_template_topk_coords")
            corrected_topk_valid = model_output.get("corrected_template_topk_valid")
            corrected_topk_mask = model_output.get("corrected_template_topk_mask")

        if corrected_topk_coords is not None and torch.is_tensor(corrected_topk_coords) and corrected_topk_coords.numel() > 0:
            raw_template_topk_coords = batch.get("template_topk_coords")
            raw_template_topk_valid = batch.get("template_topk_valid")
            raw_template_topk_mask = batch.get("template_topk_mask")
            if raw_template_topk_coords is None and template_coords is not None:
                raw_template_topk_coords = template_coords.unsqueeze(1)
                raw_template_topk_valid = torch.ones(
                    template_coords.shape[0],
                    1,
                    device=template_coords.device,
                    dtype=torch.bool,
                )
                raw_template_topk_mask = None if template_mask is None else template_mask.unsqueeze(1)

            loss, lddt_score, tm_score = self._masked_template_candidate_losses(
                pred_topk_coords=corrected_topk_coords,
                pred_topk_valid=corrected_topk_valid,
                pred_topk_mask=corrected_topk_mask,
                raw_template_topk_coords=raw_template_topk_coords,
                raw_template_topk_valid=raw_template_topk_valid,
                raw_template_topk_mask=raw_template_topk_mask,
                target=target,
                mask=mask,
                input_coords=input_coords,
                chain_idx=chain_idx,
                copy_idx=copy_idx,
                target_mask=target_mask,
            )
            template_tm_score = self._template_tm_score(batch)
            return loss, lddt_score, tm_score, template_tm_score

        loss, lddt_score, tm_score = self._masked_losses(
            pred,
            target,
            mask,
            input_coords,
            template_coords,
            template_mask,
            chain_idx,
            copy_idx,
            target_mask=target_mask,
        )
        template_tm_score = self._template_tm_score(batch)
        return loss, lddt_score, tm_score, template_tm_score

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, lddt_score, tm_score, template_tm_score = self.model_step(batch)
        self.train_loss(loss)
        self.train_lddt(lddt_score)
        self.train_tm_score(tm_score)
        self.train_template_tm_score(template_tm_score)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/lddt", self.train_lddt, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/tm_score", self.train_tm_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/template_tm_score",
            self.train_template_tm_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, lddt_score, tm_score, template_tm_score = self.model_step(batch)
        self.val_loss(loss)
        self.val_lddt(lddt_score)
        self.val_tm_score(tm_score)
        self.val_template_tm_score(template_tm_score)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/lddt", self.val_lddt, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tm_score", self.val_tm_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/template_tm_score", self.val_template_tm_score, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, lddt_score, tm_score, template_tm_score = self.model_step(batch)
        self.test_loss(loss)
        self.test_lddt(lddt_score)
        self.test_tm_score(tm_score)
        self.test_template_tm_score(template_tm_score)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/lddt", self.test_lddt, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tm_score", self.test_tm_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/template_tm_score",
            self.test_template_tm_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler_factory = self.hparams.scheduler
        scheduler_keywords = getattr(scheduler_factory, "keywords", {}) or {}
        scheduler_func = scheduler_factory.func if isinstance(scheduler_factory, partial) else scheduler_factory
        is_onecycle = scheduler_func is torch.optim.lr_scheduler.OneCycleLR

        scheduler_kwargs: Dict[str, Any] = {"optimizer": optimizer}
        if is_onecycle and "total_steps" not in scheduler_keywords and not (
            "epochs" in scheduler_keywords and "steps_per_epoch" in scheduler_keywords
        ):
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
            if total_steps <= 0:
                raise RuntimeError("OneCycleLR requires a positive `total_steps`.")
            scheduler_kwargs["total_steps"] = total_steps

        scheduler = scheduler_factory(**scheduler_kwargs)
        interval = "step" if is_onecycle else "epoch"
        lr_scheduler: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": 1,
        }

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler["monitor"] = "val/loss"

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == "__main__":
    _ = ProtenixStyleLitModule(None, None, None, None)
