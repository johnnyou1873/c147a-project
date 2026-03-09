from __future__ import annotations

import math
from typing import Any, Dict

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric


class C147ALitModule(LightningModule):
    """Lightning module for RNA coordinate identity-refinement."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
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

    def _masked_losses(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.float()
        per_token_sq = torch.sum((pred - target) ** 2, dim=-1)  # (B, L)
        denom = mask_f.sum().clamp(min=1.0)
        mse = torch.sum(per_token_sq * mask_f) / denom

        with torch.no_grad():
            tm_scores = []
            for b in range(pred.shape[0]):
                valid = mask[b]
                n_res = int(valid.sum().item())
                if n_res < 3:
                    continue

                pred_b = pred[b, valid]
                target_b = target[b, valid]
                pred_aligned = self._kabsch_align(pred_b, target_b)
                dist = torch.linalg.norm(pred_aligned - target_b, dim=-1)

                d0 = self._d0(n_res)
                tm = torch.mean(1.0 / (1.0 + (dist / d0) ** 2))
                tm_scores.append(tm)

            if tm_scores:
                tm_score = torch.stack(tm_scores).mean()
            else:
                tm_score = torch.tensor(0.0, device=pred.device)

        return mse, tm_score

    def model_step(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(batch)
        target = batch["target_coords"]
        mask = batch["mask"]
        return self._masked_losses(pred, target, mask)

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
