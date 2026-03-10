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
                qualified = template_topk_valid.clone()
                qualified_count = int(qualified.sum().item())
                # Never backfill from query/ground-truth coordinates. If coverage is low but at least
                # one real template exists, repeat the best available template candidate.
                if 0 < qualified_count < self.template_topk_count:
                    qualified_idx = torch.nonzero(qualified, as_tuple=False).squeeze(-1)
                    best_local = int(torch.argmax(template_topk_identity[qualified_idx]).item())
                    best_idx = int(qualified_idx[best_local].item())
                    fill_coords = template_topk_coords[best_idx].clone()
                    fill_identity = float(template_topk_identity[best_idx].item())
                    fill_similarity = float(template_topk_similarity[best_idx].item())

                    for k_idx in range(self.template_topk_count):
                        if qualified_count >= self.template_topk_count:
                            break
                        if bool(qualified[k_idx].item()):
                            continue
                        template_topk_coords[k_idx] = fill_coords
                        template_topk_valid[k_idx] = True
                        template_topk_identity[k_idx] = fill_identity
                        template_topk_similarity[k_idx] = fill_similarity
                        qualified[k_idx] = True
                        qualified_count += 1

            qualified = template_topk_valid.clone()
            has_template = bool(has_template or bool(qualified.any().item()))
            if has_template and not bool((template_coords.abs().sum() > 0).item()) and bool(qualified.any().item()):
                template_coords = template_topk_coords[qualified][0].clone()

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
    use_template_only_inputs: bool = False,
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
    if use_template_only_inputs:
        # Strict template-only training/eval mode: do not expose target-derived coords as model inputs.
        coords = template_coords.clone()
        coords = coords * template_mask.unsqueeze(-1).float()

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
        train_use_template_only_inputs: bool = True,
        eval_use_template_only_inputs: bool = True,
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
                        and "topk_sources" in payload
                        and "topk_identity" in payload
                        and "topk_similarity" in payload
                    )
                    topk_store = 0
                    stored_policy = ""
                    stored_allow_self_fallback = True
                    stored_exclude_self = True
                    stored_max_targets: int | None = None
                    stored_max_residues = 0
                    if isinstance(payload, dict):
                        meta = payload.get("meta", {})
                        if isinstance(meta, dict):
                            topk_store = int(meta.get("top_k_store", 0))
                            stored_policy = str(meta.get("selection_policy", ""))
                            stored_allow_self_fallback = bool(meta.get("allow_self_fallback", True))
                            stored_exclude_self = bool(meta.get("exclude_self", True))
                            raw_max_targets = meta.get("max_targets", None)
                            stored_max_targets = None if raw_max_targets is None else int(raw_max_targets)
                            stored_max_residues = int(meta.get("max_residues_per_target", 0))
                    expected_policy = "topk_non_self_by_identity_similarity_no_threshold"
                    needs_policy_upgrade = stored_policy != expected_policy
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
                        or needs_policy_upgrade
                        or stored_allow_self_fallback
                        or (not stored_exclude_self)
                        or target_coverage_insufficient
                        or residue_coverage_insufficient
                    ):
                        should_precompute = True
                        log.info(
                            "Template file at %s requires template payload upgrade "
                            "(need_topk=%d, found_topk=%d, "
                            "need_policy=%s, found_policy=%s, "
                            "need_max_targets=%s, found_max_targets=%s, need_max_residues=%d, found_max_residues=%d, "
                            "allow_self_fallback=%s, exclude_self=%s). "
                            "Recomputing.",
                            template_path,
                            int(self.hparams.template_topk_count),
                            topk_store,
                            expected_policy,
                            stored_policy,
                            str(desired_max_targets),
                            str(stored_max_targets),
                            int(self.hparams.max_residues_per_target),
                            stored_max_residues,
                            stored_allow_self_fallback,
                            stored_exclude_self,
                        )
                except Exception:
                    should_precompute = True
                    log.warning("Failed to read existing template payload at %s. Recomputing.", template_path)

            if should_precompute:
                log.info(
                    "Precomputing template payload at %s (policy=topk_non_self_by_identity_similarity_no_threshold)...",
                    template_path,
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
                    enforce_min_topk=True,
                    allow_self_fallback=False,
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
        dict[str, list[str]],
    ]:
        if not self.hparams.use_template_coords:
            return {}, {}, {}, {}, {}, {}, {}

        template_path = Path(self.hparams.data_dir) / self.hparams.template_file
        if not template_path.exists():
            log.warning("Template file not found at %s. Continuing without template coordinates.", template_path)
            return {}, {}, {}, {}, {}, {}, {}

        payload = torch.load(template_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "templates" in payload:
            templates = payload.get("templates", {})
            available = payload.get("available", {})
            topk_templates = payload.get("topk_templates", {})
            topk_mask = payload.get("topk_mask", {})
            topk_sources = payload.get("topk_sources", {})
            topk_identity = payload.get("topk_identity", {})
            topk_similarity = payload.get("topk_similarity", {})
            return templates, available, topk_templates, topk_mask, topk_identity, topk_similarity, topk_sources

        log.warning("Template file at %s has unexpected format. Continuing without templates.", template_path)
        return {}, {}, {}, {}, {}, {}, {}

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
            qualified = example.template_topk_valid
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
                f"Need >= {required} non-self templates per 512-chunk sequence. "
                f"Examples: {details}"
            )

        log.info(
            "Template coverage check passed for split='%s': min valid templates=%d (required=%d).",
            split_name,
            min_count,
            required,
        )

    def _split_indices(self, split: Dataset) -> tuple[Dataset, list[int]]:
        if isinstance(split, Subset):
            return split.dataset, list(split.indices)
        return split, list(range(len(split)))

    def _split_target_ids(self, split: Dataset) -> set[str]:
        base, indices = self._split_indices(split)
        target_ids: set[str] = set()
        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            target_ids.add(example.target_id)
        return target_ids

    def _validate_disjoint_splits(self) -> tuple[set[str], set[str], set[str]]:
        if self.data_train is None or self.data_val is None or self.data_test is None:
            raise RuntimeError("Datamodule splits are not initialized.")

        train_ids = self._split_target_ids(self.data_train)
        val_ids = self._split_target_ids(self.data_val)
        test_ids = self._split_target_ids(self.data_test)

        train_val = sorted(train_ids.intersection(val_ids))
        train_test = sorted(train_ids.intersection(test_ids))
        val_test = sorted(val_ids.intersection(test_ids))
        if train_val or train_test or val_test:
            raise RuntimeError(
                "Split leakage detected: overlapping targets across splits. "
                f"train∩val={train_val[:5]}, train∩test={train_test[:5]}, val∩test={val_test[:5]}"
            )

        log.info(
            "Split disjointness check passed: train=%d, val=%d, test=%d targets.",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
        return train_ids, val_ids, test_ids

    def _restrict_split_template_sources(
        self,
        split: Dataset,
        split_name: str,
        topk_sources_by_target: dict[str, list[str]],
        allowed_source_ids: set[str],
    ) -> None:
        if not self.hparams.use_template_coords:
            return
        if not topk_sources_by_target:
            raise RuntimeError(
                "Template payload is missing `topk_sources`; cannot enforce split leakage guards for templates."
            )

        required = max(1, int(self.hparams.template_topk_count))
        base, indices = self._split_indices(split)
        pruned_count = 0
        pruned_self_count = 0

        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            source_ids = list(topk_sources_by_target.get(example.target_id, []))
            k_count = int(example.template_topk_valid.shape[0])
            if len(source_ids) < k_count:
                source_ids.extend([""] * (k_count - len(source_ids)))

            for k_idx in range(k_count):
                source_id = source_ids[k_idx]
                keep = bool(example.template_topk_valid[k_idx].item())
                keep = keep and source_id != "" and source_id in allowed_source_ids and source_id != example.target_id
                if not keep:
                    if bool(example.template_topk_valid[k_idx].item()):
                        pruned_count += 1
                        if source_id == example.target_id:
                            pruned_self_count += 1
                    example.template_topk_valid[k_idx] = False
                    example.template_topk_identity[k_idx] = 0.0
                    example.template_topk_similarity[k_idx] = 0.0
                    example.template_topk_coords[k_idx].zero_()
                    source_ids[k_idx] = ""

            qualified = example.template_topk_valid.clone()
            qualified_count = int(qualified.sum().item())

            # Repeat best remaining template candidate if needed; never backfill from target coordinates.
            if 0 < qualified_count < required:
                qualified_idx = torch.nonzero(qualified, as_tuple=False).squeeze(-1)
                best_local = int(torch.argmax(example.template_topk_identity[qualified_idx]).item())
                best_idx = int(qualified_idx[best_local].item())
                fill_coords = example.template_topk_coords[best_idx].clone()
                fill_identity = float(example.template_topk_identity[best_idx].item())
                fill_similarity = float(example.template_topk_similarity[best_idx].item())
                fill_source = source_ids[best_idx] if best_idx < len(source_ids) else ""
                for k_idx in range(k_count):
                    if qualified_count >= required:
                        break
                    if bool(qualified[k_idx].item()):
                        continue
                    example.template_topk_coords[k_idx] = fill_coords
                    example.template_topk_valid[k_idx] = True
                    example.template_topk_identity[k_idx] = fill_identity
                    example.template_topk_similarity[k_idx] = fill_similarity
                    source_ids[k_idx] = fill_source
                    qualified[k_idx] = True
                    qualified_count += 1

            qualified = example.template_topk_valid.clone()
            if bool(qualified.any().item()):
                weights = torch.clamp(example.template_topk_similarity[qualified].to(dtype=torch.float32), min=0.0)
                if float(weights.sum().item()) <= 0.0:
                    weights = torch.ones_like(weights)
                weights = weights / weights.sum()
                coords_valid = example.template_topk_coords[qualified].to(dtype=torch.float32)
                example.template_coords = (weights[:, None, None] * coords_valid).sum(dim=0)
                example.has_template = True
            else:
                example.template_coords.zero_()
                example.has_template = False
            topk_sources_by_target[example.target_id] = source_ids

        log.info(
            "Template source restriction applied for split='%s': pruned %d candidate templates (self-source pruned=%d).",
            split_name,
            pruned_count,
            pruned_self_count,
        )

    def _validate_no_self_template_sources(
        self,
        split: Dataset,
        split_name: str,
        topk_sources_by_target: dict[str, list[str]],
    ) -> None:
        if not self.hparams.use_template_coords:
            return

        base, indices = self._split_indices(split)
        violations: list[tuple[str, int]] = []
        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            source_ids = topk_sources_by_target.get(example.target_id, [])
            k_count = int(example.template_topk_valid.shape[0])
            for k_idx in range(k_count):
                if not bool(example.template_topk_valid[k_idx].item()):
                    continue
                source_id = source_ids[k_idx] if k_idx < len(source_ids) else ""
                if source_id == example.target_id:
                    violations.append((example.target_id, k_idx))
                    if len(violations) >= 10:
                        break
            if len(violations) >= 10:
                break

        if violations:
            details = ", ".join([f"{tid}[k={k_idx}]" for tid, k_idx in violations])
            raise RuntimeError(
                f"Self-source template leakage detected in split='{split_name}'. "
                f"Found template source == target_id in active candidates: {details}"
            )

        log.info("No self-source template candidates detected for split='%s'.", split_name)

    def _filter_split_min_templates(self, split: Dataset, split_name: str, min_required: int = 1) -> Dataset:
        if not self.hparams.use_template_coords:
            return split

        required = max(1, int(min_required))
        base, indices = self._split_indices(split)
        kept_indices: list[int] = []
        dropped = 0

        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            qualified = example.template_topk_valid
            if int(qualified.sum().item()) >= required:
                kept_indices.append(idx)
            else:
                dropped += 1

        if not kept_indices:
            raise RuntimeError(f"All samples dropped from split='{split_name}' after template-only filtering.")

        if dropped > 0:
            log.info(
                "Dropped %d samples from split='%s' due to missing valid templates (required=%d).",
                dropped,
                split_name,
                required,
            )
        return Subset(base, kept_indices)

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
                template_topk_sources_by_target,
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

            train_ids, val_ids, test_ids = self._validate_disjoint_splits()
            all_ids = train_ids | val_ids | test_ids
            self._restrict_split_template_sources(
                self.data_train,
                "train",
                topk_sources_by_target=template_topk_sources_by_target,
                allowed_source_ids=train_ids,
            )
            self._validate_no_self_template_sources(
                self.data_train,
                "train",
                topk_sources_by_target=template_topk_sources_by_target,
            )
            self._restrict_split_template_sources(
                self.data_val,
                "val",
                topk_sources_by_target=template_topk_sources_by_target,
                allowed_source_ids=all_ids,
            )
            self._restrict_split_template_sources(
                self.data_test,
                "test",
                topk_sources_by_target=template_topk_sources_by_target,
                allowed_source_ids=all_ids,
            )
            self.data_train = self._filter_split_min_templates(
                self.data_train,
                "train",
                min_required=int(self.hparams.template_topk_count),
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
                use_template_only_inputs=self.hparams.train_use_template_only_inputs,
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
                use_template_only_inputs=self.hparams.eval_use_template_only_inputs,
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
                use_template_only_inputs=self.hparams.eval_use_template_only_inputs,
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
