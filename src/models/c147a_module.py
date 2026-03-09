from __future__ import annotations

import math
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
        return self.net(
            residue_idx=batch["residue_idx"],
            chain_idx=batch["chain_idx"],
            copy_idx=batch["copy_idx"],
            resid=batch["resid"],
            coords=batch["coords"],
            mask=batch["mask"],
            template_coords=batch.get("template_coords"),
            template_mask=batch.get("template_mask"),
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
        n_res = int(pred.shape[0])
        if n_res < 3:
            return pred.new_tensor(0.0)

        pred_aligned = self._kabsch_align(pred, target)
        dist = torch.linalg.norm(pred_aligned - target, dim=-1)
        d0 = self._d0(n_res)
        tm = torch.mean(1.0 / (1.0 + (dist / d0) ** 2))
        return tm.clamp(min=0.0, max=1.0)

    def _assign_groups(self, cost: torch.Tensor) -> list[int]:
        """Solve row->col assignment from a square cost matrix."""
        n = int(cost.shape[0])
        if n <= 1:
            return list(range(n))

        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            assignment = [0] * n
            for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                assignment[int(r)] = int(c)
            return assignment

        # Fallback: greedy assignment if scipy is unavailable.
        assignment = [-1] * n
        used = set()
        for r in range(n):
            order = torch.argsort(cost[r]).tolist()
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
        chain_idx: torch.Tensor,
        copy_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Differentiable TM-score for optimization.
        tm_scores_train = []
        for b in range(pred.shape[0]):
            valid = mask[b]
            pred_b = pred[b, valid]
            target_b = target[b, valid]
            if pred_b.shape[0] < 3:
                continue
            tm_scores_train.append(self._tm_score(pred_b, target_b))

        if tm_scores_train:
            tm_train = torch.stack(tm_scores_train).mean()
        else:
            tm_train = pred.new_tensor(0.0)

        loss = 1.0 - tm_train

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
        chain_idx = batch["chain_idx"]
        copy_idx = batch["copy_idx"]
        return self._masked_losses(pred, target, mask, chain_idx, copy_idx)

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

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        lr_scheduler: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler["monitor"] = "val/loss"

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == "__main__":
    _ = C147ALitModule(None, None, None, None)
