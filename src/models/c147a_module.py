from __future__ import annotations

import inspect
import math
from functools import partial
from typing import Any, Dict

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional runtime dependency
    linear_sum_assignment = None


class C147ALitModule(LightningModule):
    """Lightning module for RNA coordinate identity-refinement."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
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

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_tm_score = MeanMetric()
        self.val_tm_score = MeanMetric()
        self.test_tm_score = MeanMetric()

        self.val_loss_best = MinMetric()
        self.val_tm_score_best = MeanMetric()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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
            "template_chunk_coords": batch.get("template_chunk_coords"),
            "template_chunk_mask": batch.get("template_chunk_mask"),
            "template_chunk_start": batch.get("template_chunk_start"),
            "template_chunk_window_valid": batch.get("template_chunk_window_valid"),
            "template_chunk_valid": batch.get("template_chunk_valid"),
            "template_chunk_identity": batch.get("template_chunk_identity"),
            "template_chunk_similarity": batch.get("template_chunk_similarity"),
        }
        try:
            valid_keys = set(inspect.signature(self.net.forward).parameters.keys())
            net_inputs = {k: v for k, v in net_inputs.items() if k in valid_keys}
        except (TypeError, ValueError):
            pass
        return self.net(
            **net_inputs,
        )

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_tm_score.reset()
        self.val_loss_best.reset()
        self.val_tm_score_best.reset()

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

    def _inverse_log_factor(self, baseline_tm: torch.Tensor) -> torch.Tensor:
        """Higher factor for low baseline TM, lower (plateaued) factor for high baseline TM."""
        scale = max(float(self.hparams.must_be_better_log_scale), 1e-6)
        denom = torch.log(baseline_tm * scale + math.e).clamp(min=1e-6)
        factor = 1.0 / denom
        factor = factor.clamp(
            min=float(self.hparams.must_be_better_min_factor),
            max=float(self.hparams.must_be_better_max_factor),
        )
        return factor

    def _permutation_aware_target(
        self,
        pred_b: torch.Tensor,
        target_b: torch.Tensor,
        chain_idx_b: torch.Tensor,
        copy_idx_b: torch.Tensor,
    ) -> torch.Tensor:
        """Reorder target groups (chain/copy blocks) to be robust to copy swaps."""
        group_lists: Dict[tuple[int, int], list[int]] = {}
        for i in range(pred_b.shape[0]):
            key = (int(chain_idx_b[i].item()), int(copy_idx_b[i].item()))
            group_lists.setdefault(key, []).append(i)

        groups_by_chain: Dict[int, list[tuple[int, torch.Tensor]]] = {}
        for (chain_id, copy_id), idx_list in group_lists.items():
            idx_tensor = torch.tensor(idx_list, device=pred_b.device, dtype=torch.long)
            groups_by_chain.setdefault(chain_id, []).append((copy_id, idx_tensor))

        target_perm = target_b.clone()
        for _, groups in groups_by_chain.items():
            if len(groups) <= 1:
                continue
            groups.sort(key=lambda x: x[0])
            indices = [g[1] for g in groups]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Differentiable TM-score for optimization.
        tm_scores_train = []
        improvement_terms = []
        for b in range(pred.shape[0]):
            valid = mask[b]
            pred_b = pred[b, valid]
            target_b = target[b, valid]
            if pred_b.shape[0] < 3:
                continue
            tm_pred = self._tm_score(pred_b, target_b)
            tm_scores_train.append(tm_pred)

            with torch.no_grad():
                input_b = input_coords[b, valid].to(dtype=torch.float32)
                target_b_f = target_b.to(dtype=torch.float32)
                baseline_tm = self._tm_score(input_b, target_b_f)

                template_mask_b = template_mask[b, valid].unsqueeze(-1)
                if bool(template_mask_b.any()):
                    template_passthrough_b = torch.where(
                        template_mask_b,
                        template_coords[b, valid].to(dtype=torch.float32),
                        input_b,
                    )
                    baseline_tm = torch.maximum(baseline_tm, self._tm_score(template_passthrough_b, target_b_f))

                factor = self._inverse_log_factor(baseline_tm)
                margin = float(self.hparams.must_be_better_margin)
                improvement_gap = torch.relu(baseline_tm + margin - tm_pred)
                improvement_terms.append(factor.to(dtype=improvement_gap.dtype) * improvement_gap)

        if tm_scores_train:
            tm_train = torch.stack(tm_scores_train).mean()
        else:
            tm_train = pred.new_tensor(0.0)

        main_loss = 1.0 - tm_train
        if improvement_terms:
            improvement_loss = torch.stack(improvement_terms).mean()
        else:
            improvement_loss = pred.new_tensor(0.0)
        loss = main_loss + float(self.hparams.must_be_better_weight) * improvement_loss

        if self.training:
            # Keep train-step logging lightweight.
            tm_metric = tm_train.detach()
        else:
            # Metric path: no-grad, permutation-aware across copy groups.
            with torch.no_grad():
                tm_scores_metric = []
                for b in range(pred.shape[0]):
                    valid = mask[b]
                    pred_b = pred[b, valid].to(dtype=torch.float32)
                    target_b = target[b, valid].to(dtype=torch.float32)
                    chain_b = chain_idx[b, valid]
                    copy_b = copy_idx[b, valid]
                    if pred_b.shape[0] < 3:
                        continue

                    if self.hparams.use_permutation_aware_metric:
                        target_eval = self._permutation_aware_target(pred_b, target_b, chain_b, copy_b)
                    else:
                        target_eval = target_b

                    tm_scores_metric.append(self._tm_score(pred_b, target_eval))

                if tm_scores_metric:
                    tm_metric = torch.stack(tm_scores_metric).mean().to(dtype=pred.dtype)
                else:
                    tm_metric = pred.new_tensor(0.0)

        return loss, tm_metric

    def model_step(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(batch)
        target = batch["target_coords"]
        mask = batch["mask"]
        input_coords = batch["coords"]
        template_coords = batch["template_coords"]
        template_mask = batch["template_mask"]
        chain_idx = batch["chain_idx"]
        copy_idx = batch["copy_idx"]
        return self._masked_losses(
            pred,
            target,
            mask,
            input_coords,
            template_coords,
            template_mask,
            chain_idx,
            copy_idx,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, tm_score = self.model_step(batch)
        self.train_loss(loss)
        self.train_tm_score(tm_score)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/tm_score", self.train_tm_score, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, tm_score = self.model_step(batch)
        self.val_loss(loss)
        self.val_tm_score(tm_score)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tm_score", self.val_tm_score, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        current_loss = self.val_loss.compute()
        self.val_loss_best(current_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        current_tm = self.val_tm_score.compute()
        self.val_tm_score_best(current_tm)
        self.log("val/tm_score_best", self.val_tm_score_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, tm_score = self.model_step(batch)
        self.test_loss(loss)
        self.test_tm_score(tm_score)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tm_score", self.test_tm_score, on_step=False, on_epoch=True, prog_bar=True)

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
    _ = C147ALitModule(None, None, None, None)
