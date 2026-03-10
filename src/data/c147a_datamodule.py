from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.data.precompute_templates import precompute_template_coords


RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
BOLTZMANN_J_PER_K = 1.380649e-23
ANGSTROM_PER_METER = 1.0e10
DEFAULT_EFFECTIVE_SPRING_CONSTANT_N_PER_M = 2.0

log = logging.getLogger(__name__)


@dataclass
class RNAExample:
    target_id: str
    residue_idx: torch.Tensor  # (L,)
    chain_idx: torch.Tensor  # (L,)
    copy_idx: torch.Tensor  # (L,)
    resid_norm: torch.Tensor  # (L,)
    coords: torch.Tensor  # (L, 3)
    template_coords: torch.Tensor  # (L, 3)
    template_topk_coords: torch.Tensor  # (K, L, 3)
    template_topk_valid: torch.Tensor  # (K,)
    template_topk_identity: torch.Tensor  # (K,)
    template_topk_similarity: torch.Tensor  # (K,)
    has_template: bool


class RNAIdentityDataset(Dataset[RNAExample]):
    def __init__(
        self,
        labels_path: str | Path,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = 256,
        template_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_available_by_target: Optional[dict[str, bool]] = None,
        template_topk_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_valid_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_identity_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_similarity_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_count: int = 5,
        template_min_percent_identity: float = 50.0,
        enforce_template_coverage: bool = True,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.max_residues_per_target = max_residues_per_target
        self.max_targets = max_targets
        self.template_coords_by_target = template_coords_by_target or {}
        self.template_available_by_target = template_available_by_target or {}
        self.template_topk_coords_by_target = template_topk_coords_by_target or {}
        self.template_topk_valid_by_target = template_topk_valid_by_target or {}
        self.template_topk_identity_by_target = template_topk_identity_by_target or {}
        self.template_topk_similarity_by_target = template_topk_similarity_by_target or {}
        self.template_topk_count = max(1, int(template_topk_count))
        self.template_min_percent_identity = float(template_min_percent_identity)
        self.enforce_template_coverage = bool(enforce_template_coverage)
        self.examples, self.num_chain_types, self.max_copy_number, self.num_examples_with_template = self._load_examples()

    def _load_examples(self) -> tuple[list[RNAExample], int, int, int]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Could not find labels file: {self.labels_path}")

        rows_by_target: dict[str, list[tuple[int, int, str, int, str, int, float, float, float]]] = {}
        chain_values: set[str] = set()
        max_copy = 0

        with self.labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_idx, row in enumerate(reader):
                full_id = row.get("ID", "")
                if "_" not in full_id:
                    continue
                target_id, pos_token = full_id.rsplit("_", 1)

                if target_id not in rows_by_target:
                    if self.max_targets is not None and len(rows_by_target) >= self.max_targets:
                        continue
                    rows_by_target[target_id] = []

                target_rows = rows_by_target[target_id]
                if len(target_rows) >= self.max_residues_per_target:
                    continue

                resname = str(row.get("resname", "N")).upper()
                chain = str(row.get("chain", "UNK"))
                try:
                    pos_idx = int(float(pos_token))
                    resid = int(float(row.get("resid", 0)))
                    copy = int(float(row.get("copy", 0)))
                    x = float(row.get("x_1", 0.0))
                    y = float(row.get("y_1", 0.0))
                    z = float(row.get("z_1", 0.0))
                except ValueError:
                    continue

                target_rows.append((pos_idx, line_idx, resname, resid, chain, copy, x, y, z))
                chain_values.add(chain)
                max_copy = max(max_copy, copy)

        chain_to_idx = {chain: idx for idx, chain in enumerate(sorted(chain_values))}
        examples: list[RNAExample] = []
        num_examples_with_template = 0

        for target_id, rows in rows_by_target.items():
            rows.sort(key=lambda x: (x[0], x[1]))  # ID suffix order, then original row order
            residues = [RESNAME_TO_IDX.get(r[2], 4) for r in rows]
            chains = [chain_to_idx[r[4]] for r in rows]
            copies = [max(0, r[5]) for r in rows]
            resids = [r[3] for r in rows]
            coords = [[r[6], r[7], r[8]] for r in rows]

            resid_t = torch.tensor(resids, dtype=torch.float32)
            denom = resid_t.max().clamp(min=1.0)
            resid_norm = resid_t / denom

            coords_t = torch.tensor(coords, dtype=torch.float32)
            template_coords = torch.zeros_like(coords_t)
            template_topk_coords = torch.zeros((self.template_topk_count, coords_t.shape[0], 3), dtype=torch.float32)
            template_topk_valid = torch.zeros((self.template_topk_count,), dtype=torch.bool)
            template_topk_identity = torch.zeros((self.template_topk_count,), dtype=torch.float32)
            template_topk_similarity = torch.zeros((self.template_topk_count,), dtype=torch.float32)
            has_template = False
            if target_id in self.template_coords_by_target:
                tmpl = self.template_coords_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
                if tmpl.ndim == 2 and tmpl.shape[1] == 3:
                    seq_len = coords_t.shape[0]
                    if tmpl.shape[0] >= seq_len:
                        template_coords = tmpl[:seq_len].clone()
                    else:
                        template_coords[: tmpl.shape[0]] = tmpl
                    has_template = bool(self.template_available_by_target.get(target_id, True))

            if target_id in self.template_topk_coords_by_target:
                tmpl_topk = self.template_topk_coords_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
                if tmpl_topk.ndim == 3 and tmpl_topk.shape[-1] == 3:
                    k_take = min(self.template_topk_count, tmpl_topk.shape[0])
                    seq_len = coords_t.shape[0]
                    if tmpl_topk.shape[1] >= seq_len:
                        template_topk_coords[:k_take] = tmpl_topk[:k_take, :seq_len]
                    else:
                        template_topk_coords[:k_take, : tmpl_topk.shape[1]] = tmpl_topk[:k_take]

                    valid_topk = self.template_topk_valid_by_target.get(target_id, None)
                    if isinstance(valid_topk, torch.Tensor) and valid_topk.numel() > 0:
                        valid_topk_t = valid_topk.detach().to(dtype=torch.bool, device="cpu")
                        template_topk_valid[: min(k_take, valid_topk_t.numel())] = valid_topk_t[:k_take]
                    else:
                        template_topk_valid[:k_take] = True

                    identity_topk = self.template_topk_identity_by_target.get(target_id, None)
                    if isinstance(identity_topk, torch.Tensor) and identity_topk.numel() > 0:
                        identity_topk_t = identity_topk.detach().to(dtype=torch.float32, device="cpu")
                        template_topk_identity[: min(k_take, identity_topk_t.numel())] = identity_topk_t[:k_take]

                    similarity_topk = self.template_topk_similarity_by_target.get(target_id, None)
                    if isinstance(similarity_topk, torch.Tensor) and similarity_topk.numel() > 0:
                        similarity_topk_t = similarity_topk.detach().to(dtype=torch.float32, device="cpu")
                        template_topk_similarity[: min(k_take, similarity_topk_t.numel())] = similarity_topk_t[:k_take]

            if self.enforce_template_coverage:
                qualified = template_topk_valid & (template_topk_identity >= self.template_min_percent_identity)
                qualified_count = int(qualified.sum().item())
                if qualified_count < self.template_topk_count:
                    fallback_coords = template_coords.clone() if has_template else coords_t.clone()
                    if not has_template:
                        template_coords = fallback_coords.clone()
                        has_template = True

                    for k_idx in range(self.template_topk_count):
                        if qualified_count >= self.template_topk_count:
                            break
                        if bool(qualified[k_idx].item()):
                            continue
                        template_topk_coords[k_idx] = fallback_coords
                        template_topk_valid[k_idx] = True
                        template_topk_identity[k_idx] = 100.0
                        template_topk_similarity[k_idx] = max(
                            float(template_topk_similarity[k_idx].item()),
                            1.0,
                        )
                        qualified[k_idx] = True
                        qualified_count += 1

            if has_template:
                num_examples_with_template += 1

            examples.append(
                RNAExample(
                    target_id=target_id,
                    residue_idx=torch.tensor(residues, dtype=torch.long),
                    chain_idx=torch.tensor(chains, dtype=torch.long),
                    copy_idx=torch.tensor(copies, dtype=torch.long),
                    resid_norm=resid_norm,
                    coords=coords_t,
                    template_coords=template_coords,
                    template_topk_coords=template_topk_coords,
                    template_topk_valid=template_topk_valid,
                    template_topk_identity=template_topk_identity,
                    template_topk_similarity=template_topk_similarity,
                    has_template=has_template,
                )
            )

        if not examples:
            raise RuntimeError(f"No usable samples were found in {self.labels_path}")

        return examples, max(1, len(chain_to_idx)), max_copy, num_examples_with_template

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RNAExample:
        return self.examples[index]


def _thermal_sigma_angstrom(
    temperature_k: float, effective_spring_constant_n_per_m: float = DEFAULT_EFFECTIVE_SPRING_CONSTANT_N_PER_M
) -> float:
    """RMS positional fluctuation from equipartition: <x^2> = k_B T / k."""
    t = max(float(temperature_k), 0.0)
    k_eff = max(float(effective_spring_constant_n_per_m), 1e-12)
    sigma_m = math.sqrt(BOLTZMANN_J_PER_K * t / k_eff)
    return sigma_m * ANGSTROM_PER_METER


def _collate_rna_batch(
    batch: list[RNAExample],
    thermal_noise_sigma_angstrom: float = 0.0,
    add_thermal_noise: bool = False,
    template_chunk_length: int = 512,
    template_chunk_stride: int = 256,
    template_chunk_max_windows: int = 20,
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item.coords.shape[0] for item in batch)
    topk_count = batch[0].template_topk_coords.shape[0] if batch else 1

    residue_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    chain_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    copy_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    resid = torch.zeros(batch_size, max_len, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_topk_coords = torch.zeros(batch_size, topk_count, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    template_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    template_topk_mask = torch.zeros(batch_size, topk_count, max_len, dtype=torch.bool)
    template_topk_valid = torch.zeros(batch_size, topk_count, dtype=torch.bool)
    template_topk_identity = torch.zeros(batch_size, topk_count, dtype=torch.float32)
    template_topk_similarity = torch.zeros(batch_size, topk_count, dtype=torch.float32)

    for i, item in enumerate(batch):
        seq_len = item.coords.shape[0]
        residue_idx[i, :seq_len] = item.residue_idx
        chain_idx[i, :seq_len] = item.chain_idx
        copy_idx[i, :seq_len] = item.copy_idx
        resid[i, :seq_len] = item.resid_norm
        coords[i, :seq_len] = item.coords
        template_coords[i, :seq_len] = item.template_coords
        template_topk_coords[i, :, :seq_len] = item.template_topk_coords
        mask[i, :seq_len] = True
        if item.has_template:
            template_mask[i, :seq_len] = True
        template_topk_valid[i] = item.template_topk_valid
        template_topk_identity[i] = item.template_topk_identity
        template_topk_similarity[i] = item.template_topk_similarity
        template_topk_mask[i, :, :seq_len] = item.template_topk_valid.unsqueeze(-1)

    target_coords = coords.clone()

    if add_thermal_noise and thermal_noise_sigma_angstrom > 0.0:
        noise = torch.randn_like(coords) * thermal_noise_sigma_angstrom
        noise = noise * mask.unsqueeze(-1).float()
        coords = coords + noise

        # Apply the same temperature-based jitter to template coordinates.
        template_noise = torch.randn_like(template_coords) * thermal_noise_sigma_angstrom
        template_noise = template_noise * template_mask.unsqueeze(-1).float()
        template_coords = template_coords + template_noise

        topk_noise = torch.randn_like(template_topk_coords) * thermal_noise_sigma_angstrom
        topk_noise = topk_noise * template_topk_mask.unsqueeze(-1).float()
        template_topk_coords = template_topk_coords + topk_noise

    # Build chunked template candidates for local assembly/stitching.
    chunk_len = max(1, int(template_chunk_length))
    chunk_stride = max(1, int(template_chunk_stride))
    max_windows = max(1, int(template_chunk_max_windows))
    template_chunk_coords = torch.zeros(batch_size, max_windows, topk_count, chunk_len, 3, dtype=torch.float32)
    template_chunk_mask = torch.zeros(batch_size, max_windows, chunk_len, dtype=torch.bool)
    template_chunk_start = torch.zeros(batch_size, max_windows, dtype=torch.long)
    template_chunk_window_valid = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    template_chunk_valid = torch.zeros(batch_size, max_windows, topk_count, dtype=torch.bool)
    template_chunk_identity = torch.zeros(batch_size, max_windows, topk_count, dtype=torch.float32)
    template_chunk_similarity = torch.zeros(batch_size, max_windows, topk_count, dtype=torch.float32)

    for i in range(batch_size):
        seq_len = int(mask[i].sum().item())
        if seq_len <= 0:
            continue

        starts: list[int] = []
        start = 0
        while start < seq_len and len(starts) < max_windows:
            starts.append(start)
            if start + chunk_len >= seq_len:
                break
            start += chunk_stride

        last_start = max(0, seq_len - chunk_len)
        if last_start not in starts:
            if len(starts) < max_windows:
                starts.append(last_start)
            else:
                starts[-1] = last_start

        starts = sorted(set(starts))[:max_windows]
        for w_idx, chunk_start in enumerate(starts):
            chunk_end = min(chunk_start + chunk_len, seq_len)
            win_len = max(0, chunk_end - chunk_start)
            if win_len <= 0:
                continue

            template_chunk_start[i, w_idx] = int(chunk_start)
            template_chunk_window_valid[i, w_idx] = True
            template_chunk_mask[i, w_idx, :win_len] = True
            template_chunk_coords[i, w_idx, :, :win_len] = template_topk_coords[i, :, chunk_start:chunk_end]
            template_chunk_valid[i, w_idx] = template_topk_valid[i]
            template_chunk_identity[i, w_idx] = template_topk_identity[i]
            template_chunk_similarity[i, w_idx] = template_topk_similarity[i]

    return {
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "copy_idx": copy_idx,
        "resid": resid,
        "coords": coords,  # input coordinates
        "template_coords": template_coords,  # precomputed MSA/template coordinates
        "template_mask": template_mask,  # template availability mask
        "template_topk_coords": template_topk_coords,  # top-K precomputed template coordinates
        "template_topk_mask": template_topk_mask,  # top-K availability mask per residue
        "template_topk_valid": template_topk_valid,  # top-K availability mask per template
        "template_topk_identity": template_topk_identity,  # top-K template percent identity
        "template_topk_similarity": template_topk_similarity,  # top-K template normalized similarity
        "template_chunk_coords": template_chunk_coords,  # up to 20x5 chunk candidates
        "template_chunk_mask": template_chunk_mask,  # valid tokens per chunk window
        "template_chunk_start": template_chunk_start,  # start index of each chunk window
        "template_chunk_window_valid": template_chunk_window_valid,  # valid chunk windows
        "template_chunk_valid": template_chunk_valid,  # valid candidates per chunk window
        "template_chunk_identity": template_chunk_identity,  # candidate identity per chunk window
        "template_chunk_similarity": template_chunk_similarity,  # candidate similarity per chunk window
        "target_coords": target_coords,  # clean target coordinates
        "mask": mask,
    }


class C147ADataModule(LightningDataModule):
    """RNA identity-refinement datamodule using `train_labels.csv` as input and target."""

    def __init__(
        self,
        data_dir: str = "data/",
        labels_file: str = "train_labels.csv",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = None,
        temperature_k: float = 600.0,
        apply_thermal_noise_train: bool = True,
        apply_thermal_noise_eval: bool = False,
        use_template_coords: bool = True,
        template_file: str = "template_coords.pt",
        precompute_templates_if_missing: bool = True,
        template_min_percent_identity: float = 50.0,
        template_min_similarity: float = 0.1,
        template_max_templates: int = 8,
        template_length_ratio_tolerance: float = 0.3,
        template_topk_count: int = 5,
        template_chunk_length: int = 512,
        template_chunk_stride: int = 256,
        template_chunk_max_windows: int = 20,
        template_precompute_num_threads: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

        self.num_chain_types = 1
        self.max_copy_number = 1
        self.thermal_noise_sigma_angstrom = _thermal_sigma_angstrom(self.hparams.temperature_k)

    def prepare_data(self) -> None:
        labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Expected labels file at: {labels_path}")

        if self.hparams.use_template_coords:
            template_path = Path(self.hparams.data_dir) / self.hparams.template_file
            should_precompute = (not template_path.exists()) and self.hparams.precompute_templates_if_missing

            if template_path.exists() and self.hparams.precompute_templates_if_missing:
                try:
                    payload = torch.load(template_path, map_location="cpu", weights_only=False)
                    has_topk = (
                        isinstance(payload, dict)
                        and "topk_templates" in payload
                        and "topk_mask" in payload
                        and "topk_identity" in payload
                        and "topk_similarity" in payload
                    )
                    topk_store = 0
                    stored_min_identity = 0.0
                    stored_enforce = False
                    stored_max_targets: int | None = None
                    stored_max_residues = 0
                    if isinstance(payload, dict):
                        meta = payload.get("meta", {})
                        if isinstance(meta, dict):
                            topk_store = int(meta.get("top_k_store", 0))
                            stored_min_identity = float(meta.get("min_percent_identity", 0.0))
                            stored_enforce = bool(meta.get("enforce_min_topk", False))
                            raw_max_targets = meta.get("max_targets", None)
                            stored_max_targets = None if raw_max_targets is None else int(raw_max_targets)
                            stored_max_residues = int(meta.get("max_residues_per_target", 0))
                    needs_identity_upgrade = stored_min_identity < float(self.hparams.template_min_percent_identity)
                    desired_max_targets = self.hparams.max_targets
                    target_coverage_insufficient = (
                        (desired_max_targets is None and stored_max_targets is not None)
                        or (
                            desired_max_targets is not None
                            and stored_max_targets is not None
                            and int(stored_max_targets) < int(desired_max_targets)
                        )
                    )
                    residue_coverage_insufficient = stored_max_residues < int(self.hparams.max_residues_per_target)
                    if (
                        not has_topk
                        or topk_store < int(self.hparams.template_topk_count)
                        or needs_identity_upgrade
                        or not stored_enforce
                        or target_coverage_insufficient
                        or residue_coverage_insufficient
                    ):
                        should_precompute = True
                        log.info(
                            "Template file at %s requires template payload upgrade "
                            "(need_topk=%d, found_topk=%d, need_min_identity=%.1f, found_min_identity=%.1f, "
                            "need_max_targets=%s, found_max_targets=%s, need_max_residues=%d, found_max_residues=%d, "
                            "enforce=%s). "
                            "Recomputing.",
                            template_path,
                            int(self.hparams.template_topk_count),
                            topk_store,
                            float(self.hparams.template_min_percent_identity),
                            stored_min_identity,
                            str(desired_max_targets),
                            str(stored_max_targets),
                            int(self.hparams.max_residues_per_target),
                            stored_max_residues,
                            stored_enforce,
                        )
                except Exception:
                    should_precompute = True
                    log.warning("Failed to read existing template payload at %s. Recomputing.", template_path)

            if should_precompute:
                log.info(
                    "Precomputing template payload at %s (min_identity=%.1f%%, min_similarity=%.2f)...",
                    template_path,
                    self.hparams.template_min_percent_identity,
                    self.hparams.template_min_similarity,
                )
                precompute_template_coords(
                    labels_path=labels_path,
                    output_path=template_path,
                    min_percent_identity=self.hparams.template_min_percent_identity,
                    min_similarity=self.hparams.template_min_similarity,
                    max_templates=self.hparams.template_max_templates,
                    top_k_store=self.hparams.template_topk_count,
                    max_residues_per_target=self.hparams.max_residues_per_target,
                    max_targets=self.hparams.max_targets,
                    length_ratio_tolerance=self.hparams.template_length_ratio_tolerance,
                    exclude_self=True,
                    num_threads=self.hparams.template_precompute_num_threads,
                )

    def _load_template_payload(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, bool],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if not self.hparams.use_template_coords:
            return {}, {}, {}, {}, {}, {}

        template_path = Path(self.hparams.data_dir) / self.hparams.template_file
        if not template_path.exists():
            log.warning("Template file not found at %s. Continuing without template coordinates.", template_path)
            return {}, {}, {}, {}, {}, {}

        payload = torch.load(template_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "templates" in payload:
            templates = payload.get("templates", {})
            available = payload.get("available", {})
            topk_templates = payload.get("topk_templates", {})
            topk_mask = payload.get("topk_mask", {})
            topk_identity = payload.get("topk_identity", {})
            topk_similarity = payload.get("topk_similarity", {})
            return templates, available, topk_templates, topk_mask, topk_identity, topk_similarity

        log.warning("Template file at %s has unexpected format. Continuing without templates.", template_path)
        return {}, {}, {}, {}, {}, {}

    def _estimate_chunk_windows(self, seq_len: int) -> int:
        chunk_len = max(1, int(self.hparams.template_chunk_length))
        chunk_stride = max(1, int(self.hparams.template_chunk_stride))
        max_windows = max(1, int(self.hparams.template_chunk_max_windows))
        if seq_len <= 0:
            return 0
        starts: list[int] = []
        start = 0
        while start < seq_len and len(starts) < max_windows:
            starts.append(start)
            if start + chunk_len >= seq_len:
                break
            start += chunk_stride
        last_start = max(0, seq_len - chunk_len)
        if last_start not in starts:
            if len(starts) < max_windows:
                starts.append(last_start)
            elif starts:
                starts[-1] = last_start
        return len(sorted(set(starts)))

    def _validate_split_template_coverage(self, split: Dataset, split_name: str) -> None:
        if not self.hparams.use_template_coords:
            return

        required = max(1, int(self.hparams.template_topk_count))
        min_identity = float(self.hparams.template_min_percent_identity)

        if isinstance(split, Subset):
            base = split.dataset
            indices = list(split.indices)
        else:
            base = split
            indices = list(range(len(split)))

        if not indices:
            return

        min_count = required
        failing: list[tuple[str, int, int, float]] = []
        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            qualified = example.template_topk_valid & (example.template_topk_identity >= min_identity)
            count = int(qualified.sum().item())
            min_count = min(min_count, count)
            if count < required:
                est_windows = self._estimate_chunk_windows(int(example.coords.shape[0]))
                max_id = float(example.template_topk_identity.max().item()) if example.template_topk_identity.numel() else 0.0
                failing.append((example.target_id, count, est_windows, max_id))
                if len(failing) >= 10:
                    break

        if failing:
            details = "; ".join([f"{tid}(valid={cnt},windows={win},max_id={mx:.1f})" for tid, cnt, win, mx in failing])
            raise RuntimeError(
                f"Template coverage check failed for split='{split_name}'. "
                f"Need >= {required} templates with identity >= {min_identity:.1f}% per 512-chunk sequence. "
                f"Examples: {details}"
            )

        log.info(
            "Template coverage check passed for split='%s': min qualified templates=%d (required=%d, min_identity=%.1f%%).",
            split_name,
            min_count,
            required,
            min_identity,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None and self.data_val is None and self.data_test is None:
            labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
            (
                template_coords_by_target,
                template_available_by_target,
                template_topk_coords_by_target,
                template_topk_valid_by_target,
                template_topk_identity_by_target,
                template_topk_similarity_by_target,
            ) = self._load_template_payload()
            dataset = RNAIdentityDataset(
                labels_path=labels_path,
                max_residues_per_target=self.hparams.max_residues_per_target,
                max_targets=self.hparams.max_targets,
                template_coords_by_target=template_coords_by_target,
                template_available_by_target=template_available_by_target,
                template_topk_coords_by_target=template_topk_coords_by_target,
                template_topk_valid_by_target=template_topk_valid_by_target,
                template_topk_identity_by_target=template_topk_identity_by_target,
                template_topk_similarity_by_target=template_topk_similarity_by_target,
                template_topk_count=self.hparams.template_topk_count,
                template_min_percent_identity=self.hparams.template_min_percent_identity,
            )
            self.num_chain_types = dataset.num_chain_types
            self.max_copy_number = dataset.max_copy_number

            n_total = len(dataset)
            n_train = max(1, int(n_total * self.hparams.train_fraction))
            n_val = max(1, int(n_total * self.hparams.val_fraction))
            n_test = max(1, n_total - n_train - n_val)
            if n_train + n_val + n_test > n_total:
                n_train = max(1, n_total - n_val - n_test)

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[n_train, n_val, n_total - n_train - n_val],
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )

            # User-requested strict template coverage checks on both training and test splits.
            self._validate_split_template_coverage(self.data_train, "train")
            self._validate_split_template_coverage(self.data_test, "test")

            log.info(
                "Thermal coordinate noise: T=%.1fK, sigma=%.3fA (train=%s, eval=%s); template examples=%d/%d",
                self.hparams.temperature_k,
                self.thermal_noise_sigma_angstrom,
                self.hparams.apply_thermal_noise_train,
                self.hparams.apply_thermal_noise_eval,
                dataset.num_examples_with_template,
                len(dataset),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=partial(
                _collate_rna_batch,
                thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
                add_thermal_noise=self.hparams.apply_thermal_noise_train,
                template_chunk_length=self.hparams.template_chunk_length,
                template_chunk_stride=self.hparams.template_chunk_stride,
                template_chunk_max_windows=self.hparams.template_chunk_max_windows,
            ),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=partial(
                _collate_rna_batch,
                thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
                add_thermal_noise=self.hparams.apply_thermal_noise_eval,
                template_chunk_length=self.hparams.template_chunk_length,
                template_chunk_stride=self.hparams.template_chunk_stride,
                template_chunk_max_windows=self.hparams.template_chunk_max_windows,
            ),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=partial(
                _collate_rna_batch,
                thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
                add_thermal_noise=self.hparams.apply_thermal_noise_eval,
                template_chunk_length=self.hparams.template_chunk_length,
                template_chunk_stride=self.hparams.template_chunk_stride,
                template_chunk_max_windows=self.hparams.template_chunk_max_windows,
            ),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = C147ADataModule()
