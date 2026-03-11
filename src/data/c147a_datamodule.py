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


def _compute_chunk_starts(seq_len: int, chunk_length: int, chunk_stride: int, max_windows: int) -> list[int]:
    length = max(1, int(chunk_length))
    stride = max(1, int(chunk_stride))
    max_w = max(1, int(max_windows))
    if seq_len <= 0:
        return []

    starts: list[int] = []
    start = 0
    while start < seq_len and len(starts) < max_w:
        starts.append(start)
        if start + length >= seq_len:
            break
        start += stride

    last_start = max(0, seq_len - length)
    if last_start not in starts:
        if len(starts) < max_w:
            starts.append(last_start)
        elif starts:
            starts[-1] = last_start
    return sorted(set(starts))[:max_w]


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
    template_topk_residue_idx: torch.Tensor  # (K, L)
    template_chunk_coords: torch.Tensor  # (W, K, C, 3)
    template_chunk_mask: torch.Tensor  # (W, C)
    template_chunk_start: torch.Tensor  # (W,)
    template_chunk_window_valid: torch.Tensor  # (W,)
    template_chunk_valid: torch.Tensor  # (W, K)
    template_chunk_identity: torch.Tensor  # (W, K)
    template_chunk_similarity: torch.Tensor  # (W, K)
    template_chunk_confidence: torch.Tensor  # (W, K)
    template_chunk_source_onehot: torch.Tensor  # (W, K, 2)
    template_chunk_residue_idx: torch.Tensor  # (W, K, C)
    has_template: bool


class RNAIdentityDataset(Dataset[RNAExample]):
    def __init__(
        self,
        labels_path: str | Path,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = 256,
        template_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_available_by_target: Optional[dict[str, bool]] = None,
        template_chunk_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_mask_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_start_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_window_valid_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_valid_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_identity_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_similarity_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_confidence_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_source_onehot_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_chunk_residue_idx_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_count: int = 5,
        template_chunk_length: int = 512,
        template_chunk_stride: int = 256,
        template_chunk_max_windows: int = 64,
        template_min_percent_identity: float = 50.0,
        enforce_template_coverage: bool = True,
        template_force_oracle_only: bool = False,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.max_residues_per_target = max_residues_per_target
        self.max_targets = max_targets
        self.template_coords_by_target = template_coords_by_target or {}
        self.template_available_by_target = template_available_by_target or {}
        self.template_chunk_coords_by_target = template_chunk_coords_by_target or {}
        self.template_chunk_mask_by_target = template_chunk_mask_by_target or {}
        self.template_chunk_start_by_target = template_chunk_start_by_target or {}
        self.template_chunk_window_valid_by_target = template_chunk_window_valid_by_target or {}
        self.template_chunk_valid_by_target = template_chunk_valid_by_target or {}
        self.template_chunk_identity_by_target = template_chunk_identity_by_target or {}
        self.template_chunk_similarity_by_target = template_chunk_similarity_by_target or {}
        self.template_chunk_confidence_by_target = template_chunk_confidence_by_target or {}
        self.template_chunk_source_onehot_by_target = template_chunk_source_onehot_by_target or {}
        self.template_chunk_residue_idx_by_target = template_chunk_residue_idx_by_target or {}
        self.template_topk_count = max(1, int(template_topk_count))
        self.template_chunk_length = max(1, int(template_chunk_length))
        self.template_chunk_stride = max(1, int(template_chunk_stride))
        self.template_chunk_max_windows = max(1, int(template_chunk_max_windows))
        self.template_min_percent_identity = float(template_min_percent_identity)
        self.enforce_template_coverage = bool(enforce_template_coverage)
        self.template_force_oracle_only = bool(template_force_oracle_only)
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
            residue_idx_t = torch.tensor(residues, dtype=torch.long)

            resid_t = torch.tensor(resids, dtype=torch.float32)
            denom = resid_t.max().clamp(min=1.0)
            resid_norm = resid_t / denom

            coords_t = torch.tensor(coords, dtype=torch.float32)
            template_coords = torch.zeros_like(coords_t)
            template_topk_coords = torch.zeros((self.template_topk_count, coords_t.shape[0], 3), dtype=torch.float32)
            template_topk_valid = torch.zeros((self.template_topk_count,), dtype=torch.bool)
            template_topk_identity = torch.zeros((self.template_topk_count,), dtype=torch.float32)
            template_topk_similarity = torch.zeros((self.template_topk_count,), dtype=torch.float32)
            template_topk_residue_idx = torch.full(
                (self.template_topk_count, coords_t.shape[0]),
                4,
                dtype=torch.long,
            )
            template_chunk_coords = torch.zeros(
                (self.template_chunk_max_windows, self.template_topk_count, self.template_chunk_length, 3),
                dtype=torch.float32,
            )
            template_chunk_mask = torch.zeros(
                (self.template_chunk_max_windows, self.template_chunk_length),
                dtype=torch.bool,
            )
            template_chunk_start = torch.zeros((self.template_chunk_max_windows,), dtype=torch.long)
            template_chunk_window_valid = torch.zeros((self.template_chunk_max_windows,), dtype=torch.bool)
            template_chunk_valid = torch.zeros((self.template_chunk_max_windows, self.template_topk_count), dtype=torch.bool)
            template_chunk_identity = torch.zeros(
                (self.template_chunk_max_windows, self.template_topk_count),
                dtype=torch.float32,
            )
            template_chunk_similarity = torch.zeros(
                (self.template_chunk_max_windows, self.template_topk_count),
                dtype=torch.float32,
            )
            template_chunk_confidence = torch.zeros(
                (self.template_chunk_max_windows, self.template_topk_count),
                dtype=torch.float32,
            )
            template_chunk_source_onehot = torch.zeros(
                (self.template_chunk_max_windows, self.template_topk_count, 2),
                dtype=torch.float32,
            )
            template_chunk_residue_idx = torch.full(
                (self.template_chunk_max_windows, self.template_topk_count, self.template_chunk_length),
                4,
                dtype=torch.long,
            )
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

            if target_id in self.template_chunk_coords_by_target:
                chunk_coords_raw = self.template_chunk_coords_by_target[target_id].detach().to(
                    dtype=torch.float32, device="cpu"
                )
                if chunk_coords_raw.ndim == 5 and chunk_coords_raw.shape[-1] == 3:
                    w_take = min(self.template_chunk_max_windows, int(chunk_coords_raw.shape[0]))
                    k_take = min(self.template_topk_count, int(chunk_coords_raw.shape[1]))
                    c_take = min(self.template_chunk_length, int(chunk_coords_raw.shape[2]))
                    template_chunk_coords[:w_take, :k_take, :c_take] = chunk_coords_raw[:w_take, :k_take, :c_take]

                chunk_mask_raw = self.template_chunk_mask_by_target.get(target_id, None)
                if isinstance(chunk_mask_raw, torch.Tensor) and chunk_mask_raw.numel() > 0:
                    chunk_mask_t = chunk_mask_raw.detach().to(dtype=torch.bool, device="cpu")
                    if chunk_mask_t.ndim == 2:
                        w_take = min(self.template_chunk_max_windows, int(chunk_mask_t.shape[0]))
                        c_take = min(self.template_chunk_length, int(chunk_mask_t.shape[1]))
                        template_chunk_mask[:w_take, :c_take] = chunk_mask_t[:w_take, :c_take]

                chunk_start_raw = self.template_chunk_start_by_target.get(target_id, None)
                if isinstance(chunk_start_raw, torch.Tensor) and chunk_start_raw.numel() > 0:
                    chunk_start_t = chunk_start_raw.detach().to(dtype=torch.long, device="cpu")
                    if chunk_start_t.ndim == 1:
                        w_take = min(self.template_chunk_max_windows, int(chunk_start_t.shape[0]))
                        template_chunk_start[:w_take] = chunk_start_t[:w_take]

                chunk_window_valid_raw = self.template_chunk_window_valid_by_target.get(target_id, None)
                if isinstance(chunk_window_valid_raw, torch.Tensor) and chunk_window_valid_raw.numel() > 0:
                    chunk_window_valid_t = chunk_window_valid_raw.detach().to(dtype=torch.bool, device="cpu")
                    if chunk_window_valid_t.ndim == 1:
                        w_take = min(self.template_chunk_max_windows, int(chunk_window_valid_t.shape[0]))
                        template_chunk_window_valid[:w_take] = chunk_window_valid_t[:w_take]

                chunk_valid_raw = self.template_chunk_valid_by_target.get(target_id, None)
                if isinstance(chunk_valid_raw, torch.Tensor) and chunk_valid_raw.numel() > 0:
                    chunk_valid_t = chunk_valid_raw.detach().to(dtype=torch.bool, device="cpu")
                    if chunk_valid_t.ndim == 2:
                        w_take = min(self.template_chunk_max_windows, int(chunk_valid_t.shape[0]))
                        k_take = min(self.template_topk_count, int(chunk_valid_t.shape[1]))
                        template_chunk_valid[:w_take, :k_take] = chunk_valid_t[:w_take, :k_take]

                chunk_identity_raw = self.template_chunk_identity_by_target.get(target_id, None)
                if isinstance(chunk_identity_raw, torch.Tensor) and chunk_identity_raw.numel() > 0:
                    chunk_identity_t = chunk_identity_raw.detach().to(dtype=torch.float32, device="cpu")
                    if chunk_identity_t.ndim == 2:
                        w_take = min(self.template_chunk_max_windows, int(chunk_identity_t.shape[0]))
                        k_take = min(self.template_topk_count, int(chunk_identity_t.shape[1]))
                        template_chunk_identity[:w_take, :k_take] = chunk_identity_t[:w_take, :k_take]

                chunk_similarity_raw = self.template_chunk_similarity_by_target.get(target_id, None)
                if isinstance(chunk_similarity_raw, torch.Tensor) and chunk_similarity_raw.numel() > 0:
                    chunk_similarity_t = chunk_similarity_raw.detach().to(dtype=torch.float32, device="cpu")
                    if chunk_similarity_t.ndim == 2:
                        w_take = min(self.template_chunk_max_windows, int(chunk_similarity_t.shape[0]))
                        k_take = min(self.template_topk_count, int(chunk_similarity_t.shape[1]))
                        template_chunk_similarity[:w_take, :k_take] = chunk_similarity_t[:w_take, :k_take]

                chunk_confidence_raw = self.template_chunk_confidence_by_target.get(target_id, None)
                if isinstance(chunk_confidence_raw, torch.Tensor) and chunk_confidence_raw.numel() > 0:
                    chunk_confidence_t = chunk_confidence_raw.detach().to(dtype=torch.float32, device="cpu")
                    if chunk_confidence_t.ndim == 2:
                        w_take = min(self.template_chunk_max_windows, int(chunk_confidence_t.shape[0]))
                        k_take = min(self.template_topk_count, int(chunk_confidence_t.shape[1]))
                        template_chunk_confidence[:w_take, :k_take] = chunk_confidence_t[:w_take, :k_take]

                chunk_source_onehot_raw = self.template_chunk_source_onehot_by_target.get(target_id, None)
                if isinstance(chunk_source_onehot_raw, torch.Tensor) and chunk_source_onehot_raw.numel() > 0:
                    chunk_source_onehot_t = chunk_source_onehot_raw.detach().to(dtype=torch.float32, device="cpu")
                    if chunk_source_onehot_t.ndim == 3 and int(chunk_source_onehot_t.shape[-1]) == 2:
                        w_take = min(self.template_chunk_max_windows, int(chunk_source_onehot_t.shape[0]))
                        k_take = min(self.template_topk_count, int(chunk_source_onehot_t.shape[1]))
                        template_chunk_source_onehot[:w_take, :k_take] = chunk_source_onehot_t[:w_take, :k_take]

                chunk_residue_idx_raw = self.template_chunk_residue_idx_by_target.get(target_id, None)
                if isinstance(chunk_residue_idx_raw, torch.Tensor) and chunk_residue_idx_raw.numel() > 0:
                    chunk_residue_idx_t = chunk_residue_idx_raw.detach().to(dtype=torch.long, device="cpu")
                    if chunk_residue_idx_t.ndim == 3:
                        w_take = min(self.template_chunk_max_windows, int(chunk_residue_idx_t.shape[0]))
                        k_take = min(self.template_topk_count, int(chunk_residue_idx_t.shape[1]))
                        c_take = min(self.template_chunk_length, int(chunk_residue_idx_t.shape[2]))
                        template_chunk_residue_idx[:w_take, :k_take, :c_take] = chunk_residue_idx_t[
                            :w_take, :k_take, :c_take
                        ]

            if self.template_force_oracle_only:
                seq_len = int(coords_t.shape[0])
                starts = _compute_chunk_starts(
                    seq_len=seq_len,
                    chunk_length=self.template_chunk_length,
                    chunk_stride=self.template_chunk_stride,
                    max_windows=self.template_chunk_max_windows,
                )
                k_count = int(self.template_topk_count)
                c_len = int(self.template_chunk_length)

                template_coords = coords_t.clone()
                template_topk_coords = coords_t.unsqueeze(0).repeat(k_count, 1, 1)
                template_topk_valid = torch.ones((k_count,), dtype=torch.bool)
                template_topk_identity = torch.full((k_count,), 100.0, dtype=torch.float32)
                template_topk_similarity = torch.full((k_count,), 1.0, dtype=torch.float32)
                template_topk_residue_idx = residue_idx_t.unsqueeze(0).repeat(k_count, 1)

                template_chunk_coords.zero_()
                template_chunk_mask.zero_()
                template_chunk_start.zero_()
                template_chunk_window_valid.zero_()
                template_chunk_valid.zero_()
                template_chunk_identity.zero_()
                template_chunk_similarity.zero_()
                template_chunk_confidence.zero_()
                template_chunk_source_onehot.zero_()
                template_chunk_residue_idx.fill_(4)

                for w_idx, start_idx in enumerate(starts):
                    if w_idx >= self.template_chunk_max_windows:
                        break
                    end_idx = min(seq_len, int(start_idx) + c_len)
                    seg_len = max(0, end_idx - int(start_idx))
                    if seg_len <= 0:
                        continue
                    template_chunk_start[w_idx] = int(start_idx)
                    template_chunk_window_valid[w_idx] = True
                    template_chunk_mask[w_idx, :seg_len] = True
                    for k_idx in range(k_count):
                        template_chunk_coords[w_idx, k_idx, :seg_len] = coords_t[int(start_idx) : end_idx]
                        template_chunk_valid[w_idx, k_idx] = True
                        template_chunk_identity[w_idx, k_idx] = 100.0
                        template_chunk_similarity[w_idx, k_idx] = 1.0
                        template_chunk_confidence[w_idx, k_idx] = 0.0
                        template_chunk_source_onehot[w_idx, k_idx, 0] = 1.0
                        template_chunk_source_onehot[w_idx, k_idx, 1] = 0.0
                        template_chunk_residue_idx[w_idx, k_idx, :seg_len] = residue_idx_t[int(start_idx) : end_idx]
                has_template = True

            if self.enforce_template_coverage:
                active_windows = torch.nonzero(template_chunk_window_valid, as_tuple=False).squeeze(-1)
                for w_idx_t in active_windows:
                    w_idx = int(w_idx_t.item())
                    qualified = template_chunk_valid[w_idx].clone()
                    qualified_count = int(qualified.sum().item())
                    if 0 < qualified_count < self.template_topk_count:
                        qualified_idx = torch.nonzero(qualified, as_tuple=False).squeeze(-1)
                        best_local = int(torch.argmax(template_chunk_identity[w_idx, qualified_idx]).item())
                        best_idx = int(qualified_idx[best_local].item())
                        fill_coords = template_chunk_coords[w_idx, best_idx].clone()
                        fill_identity = float(template_chunk_identity[w_idx, best_idx].item())
                        fill_similarity = float(template_chunk_similarity[w_idx, best_idx].item())
                        fill_confidence = float(template_chunk_confidence[w_idx, best_idx].item())
                        fill_source_onehot = template_chunk_source_onehot[w_idx, best_idx].clone()
                        fill_residue_idx = template_chunk_residue_idx[w_idx, best_idx].clone()
                        for k_idx in range(self.template_topk_count):
                            if qualified_count >= self.template_topk_count:
                                break
                            if bool(qualified[k_idx].item()):
                                continue
                            template_chunk_coords[w_idx, k_idx] = fill_coords
                            template_chunk_valid[w_idx, k_idx] = True
                            template_chunk_identity[w_idx, k_idx] = fill_identity
                            template_chunk_similarity[w_idx, k_idx] = fill_similarity
                            template_chunk_confidence[w_idx, k_idx] = fill_confidence
                            template_chunk_source_onehot[w_idx, k_idx] = fill_source_onehot
                            template_chunk_residue_idx[w_idx, k_idx] = fill_residue_idx
                            qualified[k_idx] = True
                            qualified_count += 1

            (
                rebuilt_template_coords,
                rebuilt_topk_coords,
                rebuilt_topk_valid,
                rebuilt_topk_identity,
                rebuilt_topk_similarity,
                rebuilt_topk_residue_idx,
                rebuilt_has_template,
            ) = _rebuild_template_views_from_chunks(
                seq_len=int(coords_t.shape[0]),
                chunk_coords=template_chunk_coords,
                chunk_mask=template_chunk_mask,
                chunk_start=template_chunk_start,
                chunk_window_valid=template_chunk_window_valid,
                chunk_valid=template_chunk_valid,
                chunk_identity=template_chunk_identity,
                chunk_similarity=template_chunk_similarity,
                chunk_residue_idx=template_chunk_residue_idx,
                topk_count=self.template_topk_count,
            )
            template_coords = rebuilt_template_coords
            template_topk_coords = rebuilt_topk_coords
            template_topk_valid = rebuilt_topk_valid
            template_topk_identity = rebuilt_topk_identity
            template_topk_similarity = rebuilt_topk_similarity
            template_topk_residue_idx = rebuilt_topk_residue_idx
            has_template = bool(has_template or rebuilt_has_template)

            if has_template:
                num_examples_with_template += 1

            examples.append(
                RNAExample(
                    target_id=target_id,
                    residue_idx=residue_idx_t,
                    chain_idx=torch.tensor(chains, dtype=torch.long),
                    copy_idx=torch.tensor(copies, dtype=torch.long),
                    resid_norm=resid_norm,
                    coords=coords_t,
                    template_coords=template_coords,
                    template_topk_coords=template_topk_coords,
                    template_topk_valid=template_topk_valid,
                    template_topk_identity=template_topk_identity,
                    template_topk_similarity=template_topk_similarity,
                    template_topk_residue_idx=template_topk_residue_idx,
                    template_chunk_coords=template_chunk_coords,
                    template_chunk_mask=template_chunk_mask,
                    template_chunk_start=template_chunk_start,
                    template_chunk_window_valid=template_chunk_window_valid,
                    template_chunk_valid=template_chunk_valid,
                    template_chunk_identity=template_chunk_identity,
                    template_chunk_similarity=template_chunk_similarity,
                    template_chunk_confidence=template_chunk_confidence,
                    template_chunk_source_onehot=template_chunk_source_onehot,
                    template_chunk_residue_idx=template_chunk_residue_idx,
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


def _rebuild_template_views_from_chunks(
    seq_len: int,
    chunk_coords: torch.Tensor,
    chunk_mask: torch.Tensor,
    chunk_start: torch.Tensor,
    chunk_window_valid: torch.Tensor,
    chunk_valid: torch.Tensor,
    chunk_identity: torch.Tensor,
    chunk_similarity: torch.Tensor,
    chunk_residue_idx: torch.Tensor,
    topk_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    template_coords = torch.zeros((seq_len, 3), dtype=torch.float32)
    template_topk_coords = torch.zeros((topk_count, seq_len, 3), dtype=torch.float32)
    template_topk_valid = torch.zeros((topk_count,), dtype=torch.bool)
    template_topk_identity = torch.zeros((topk_count,), dtype=torch.float32)
    template_topk_similarity = torch.zeros((topk_count,), dtype=torch.float32)
    template_topk_residue_idx = torch.full((topk_count, seq_len), 4, dtype=torch.long)

    active_windows = torch.nonzero(chunk_window_valid, as_tuple=False).squeeze(-1)
    has_template = False

    coord_sum = torch.zeros((seq_len, 3), dtype=torch.float32)
    weight_sum = torch.zeros((seq_len, 1), dtype=torch.float32)
    first_active_window: int | None = None

    for w_idx_t in active_windows:
        w_idx = int(w_idx_t.item())
        valid_row = chunk_valid[w_idx]
        if not bool(valid_row.any().item()):
            continue
        if first_active_window is None:
            first_active_window = w_idx

        start_idx = int(chunk_start[w_idx].item())
        win_len = int(chunk_mask[w_idx].sum().item())
        end_idx = min(seq_len, start_idx + win_len)
        if end_idx <= start_idx:
            continue
        seg_len = end_idx - start_idx
        for k_idx in range(min(topk_count, int(valid_row.shape[0]))):
            if not bool(valid_row[k_idx].item()):
                continue
            has_template = True
            weight = max(1e-4, float(chunk_similarity[w_idx, k_idx].item()))
            coord_sum[start_idx:end_idx] += chunk_coords[w_idx, k_idx, :seg_len] * weight
            weight_sum[start_idx:end_idx] += weight

    valid_weight = weight_sum.squeeze(-1) > 0.0
    if bool(valid_weight.any().item()):
        template_coords[valid_weight] = coord_sum[valid_weight] / weight_sum[valid_weight]

    if first_active_window is not None:
        w_idx = int(first_active_window)
        start_idx = int(chunk_start[w_idx].item())
        win_len = int(chunk_mask[w_idx].sum().item())
        end_idx = min(seq_len, start_idx + win_len)
        if end_idx > start_idx:
            seg_len = end_idx - start_idx
            k_take = min(topk_count, int(chunk_valid.shape[1]))
            template_topk_valid[:k_take] = chunk_valid[w_idx, :k_take]
            template_topk_identity[:k_take] = chunk_identity[w_idx, :k_take]
            template_topk_similarity[:k_take] = chunk_similarity[w_idx, :k_take]
            template_topk_coords[:k_take, start_idx:end_idx] = chunk_coords[w_idx, :k_take, :seg_len]
            template_topk_residue_idx[:k_take, start_idx:end_idx] = chunk_residue_idx[w_idx, :k_take, :seg_len]

    return (
        template_coords,
        template_topk_coords,
        template_topk_valid,
        template_topk_identity,
        template_topk_similarity,
        template_topk_residue_idx,
        has_template,
    )


def _collate_rna_batch(
    batch: list[RNAExample],
    thermal_noise_sigma_angstrom: float = 0.0,
    add_thermal_noise: bool = False,
    use_template_only_inputs: bool = False,
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
    template_topk_residue_idx = torch.full((batch_size, topk_count, max_len), 4, dtype=torch.long)
    chunk_windows = batch[0].template_chunk_coords.shape[0] if batch else 1
    chunk_length = batch[0].template_chunk_coords.shape[2] if batch else 1
    template_chunk_coords = torch.zeros(
        batch_size,
        chunk_windows,
        topk_count,
        chunk_length,
        3,
        dtype=torch.float32,
    )
    template_chunk_mask = torch.zeros(batch_size, chunk_windows, chunk_length, dtype=torch.bool)
    template_chunk_start = torch.zeros(batch_size, chunk_windows, dtype=torch.long)
    template_chunk_window_valid = torch.zeros(batch_size, chunk_windows, dtype=torch.bool)
    template_chunk_valid = torch.zeros(batch_size, chunk_windows, topk_count, dtype=torch.bool)
    template_chunk_identity = torch.zeros(batch_size, chunk_windows, topk_count, dtype=torch.float32)
    template_chunk_similarity = torch.zeros(batch_size, chunk_windows, topk_count, dtype=torch.float32)
    template_chunk_confidence = torch.zeros(batch_size, chunk_windows, topk_count, dtype=torch.float32)
    template_chunk_source_onehot = torch.zeros(batch_size, chunk_windows, topk_count, 2, dtype=torch.float32)
    template_chunk_residue_idx = torch.full(
        (batch_size, chunk_windows, topk_count, chunk_length),
        4,
        dtype=torch.long,
    )

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
        template_mask[i, :seq_len] = item.template_coords.abs().sum(dim=-1) > 0
        template_topk_valid[i] = item.template_topk_valid
        template_topk_identity[i] = item.template_topk_identity
        template_topk_similarity[i] = item.template_topk_similarity
        template_topk_residue_idx[i, :, :seq_len] = item.template_topk_residue_idx
        template_topk_mask[i, :, :seq_len] = item.template_topk_valid.unsqueeze(-1)
        template_chunk_coords[i] = item.template_chunk_coords
        template_chunk_mask[i] = item.template_chunk_mask
        template_chunk_start[i] = item.template_chunk_start
        template_chunk_window_valid[i] = item.template_chunk_window_valid
        template_chunk_valid[i] = item.template_chunk_valid
        template_chunk_identity[i] = item.template_chunk_identity
        template_chunk_similarity[i] = item.template_chunk_similarity
        template_chunk_confidence[i] = item.template_chunk_confidence
        template_chunk_source_onehot[i] = item.template_chunk_source_onehot
        template_chunk_residue_idx[i] = item.template_chunk_residue_idx

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
        chunk_noise = torch.randn_like(template_chunk_coords) * thermal_noise_sigma_angstrom
        chunk_token_mask = template_chunk_mask.unsqueeze(2).unsqueeze(-1).float()
        chunk_cand_mask = template_chunk_valid.unsqueeze(-1).unsqueeze(-1).float()
        chunk_noise = chunk_noise * chunk_token_mask * chunk_cand_mask
        template_chunk_coords = template_chunk_coords + chunk_noise

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
        "template_topk_residue_idx": template_topk_residue_idx,  # top-K template residue ids
        "template_chunk_coords": template_chunk_coords,  # up to W x K chunk candidates
        "template_chunk_mask": template_chunk_mask,  # valid tokens per chunk window
        "template_chunk_start": template_chunk_start,  # start index of each chunk window
        "template_chunk_window_valid": template_chunk_window_valid,  # valid chunk windows
        "template_chunk_valid": template_chunk_valid,  # valid candidates per chunk window
        "template_chunk_identity": template_chunk_identity,  # candidate identity per chunk window
        "template_chunk_similarity": template_chunk_similarity,  # candidate similarity per chunk window
        "template_chunk_confidence": template_chunk_confidence,  # candidate confidence per chunk window (Protenix)
        "template_chunk_source_onehot": template_chunk_source_onehot,  # [is_template, is_protenix] per candidate
        "template_chunk_residue_idx": template_chunk_residue_idx,  # candidate template residue ids per chunk
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
        template_min_similarity: float = 0.0,
        template_max_templates: int = 8,
        template_length_ratio_tolerance: float = 0.3,
        template_topk_count: int = 5,
        template_chunk_length: int = 512,
        template_chunk_stride: int = 256,
        template_chunk_max_windows: int = 64,
        template_protenix_fallback_zip: str = "protenix_finished_chunks_full_4gpu_2775chunks_20260309T215807Z",
        template_protenix_base_confidence: float = 0.85,
        template_force_oracle_only: bool = False,
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
            if bool(self.hparams.template_force_oracle_only):
                log.warning(
                    "template_force_oracle_only=True: skipping template precompute and using target-derived oracle templates for all candidates."
                )
                return
            template_path = Path(self.hparams.data_dir) / self.hparams.template_file
            should_precompute = (not template_path.exists()) and self.hparams.precompute_templates_if_missing

            if template_path.exists() and self.hparams.precompute_templates_if_missing:
                try:
                    payload = torch.load(template_path, map_location="cpu", weights_only=False)
                    has_required_fields = (
                        isinstance(payload, dict)
                        and "templates" in payload
                        and "available" in payload
                        and "chunk_topk_templates" in payload
                        and "chunk_mask" in payload
                        and "chunk_start" in payload
                        and "chunk_window_valid" in payload
                        and "chunk_topk_valid" in payload
                        and "chunk_topk_identity" in payload
                        and "chunk_topk_similarity" in payload
                        and "chunk_topk_confidence" in payload
                        and "chunk_topk_source_onehot" in payload
                        and "chunk_topk_residue_idx" in payload
                        and "chunk_topk_sources" in payload
                        and "protenix_chunks_used" in payload
                    )
                    topk_store = 0
                    stored_chunk_policy = ""
                    stored_alignment_mode = ""
                    stored_allow_self_fallback = True
                    stored_exclude_self = True
                    stored_max_targets: int | None = None
                    stored_max_residues = 0
                    stored_chunk_length = 0
                    stored_chunk_stride = 0
                    stored_chunk_max_windows = 0
                    stored_search_strategy = ""
                    if isinstance(payload, dict):
                        meta = payload.get("meta", {})
                        if isinstance(meta, dict):
                            topk_store = int(meta.get("top_k_store", 0))
                            stored_chunk_policy = str(meta.get("chunk_selection_policy", ""))
                            stored_alignment_mode = str(meta.get("alignment_mode", ""))
                            stored_allow_self_fallback = bool(meta.get("allow_self_fallback", True))
                            stored_exclude_self = bool(meta.get("exclude_self", True))
                            raw_max_targets = meta.get("max_targets", None)
                            stored_max_targets = None if raw_max_targets is None else int(raw_max_targets)
                            stored_max_residues = int(meta.get("max_residues_per_target", 0))
                            stored_chunk_length = int(meta.get("chunk_length", 0))
                            stored_chunk_stride = int(meta.get("chunk_stride", 0))
                            stored_chunk_max_windows = int(meta.get("chunk_max_windows", 0))
                            stored_search_strategy = str(
                                meta.get("search_strategy", meta.get("prefilter_strategy", ""))
                            )
                    expected_chunk_policy = (
                        "chunked_topk_non_self_by_similarity_rank_identity_gate_with_protenix_then_oracle_diversity_fallback"
                    )
                    expected_alignment_mode = "global"
                    expected_search_strategy = "full_exhaustive_alignment"
                    needs_chunk_policy_upgrade = stored_chunk_policy != expected_chunk_policy
                    needs_alignment_upgrade = stored_alignment_mode != expected_alignment_mode
                    needs_search_upgrade = stored_search_strategy != expected_search_strategy
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
                    chunk_shape_mismatch = (
                        stored_chunk_length != int(self.hparams.template_chunk_length)
                        or stored_chunk_stride != int(self.hparams.template_chunk_stride)
                        or stored_chunk_max_windows != int(self.hparams.template_chunk_max_windows)
                    )
                    if (
                        not has_required_fields
                        or topk_store < int(self.hparams.template_topk_count)
                        or needs_chunk_policy_upgrade
                        or needs_alignment_upgrade
                        or needs_search_upgrade
                        or stored_allow_self_fallback
                        or (not stored_exclude_self)
                        or target_coverage_insufficient
                        or residue_coverage_insufficient
                        or chunk_shape_mismatch
                    ):
                        should_precompute = True
                        log.info(
                            "Template file at %s requires template payload upgrade "
                            "(need_topk=%d, found_topk=%d, "
                            "need_chunk_policy=%s, found_chunk_policy=%s, "
                            "need_alignment=%s, found_alignment=%s, "
                            "need_search=%s, found_search=%s, "
                            "need_max_targets=%s, found_max_targets=%s, need_max_residues=%d, found_max_residues=%d, "
                            "need_chunk=(%d,%d,%d), found_chunk=(%d,%d,%d), "
                            "allow_self_fallback=%s, exclude_self=%s). "
                            "Recomputing.",
                            template_path,
                            int(self.hparams.template_topk_count),
                            topk_store,
                            expected_chunk_policy,
                            stored_chunk_policy,
                            expected_alignment_mode,
                            stored_alignment_mode,
                            expected_search_strategy,
                            stored_search_strategy,
                            str(desired_max_targets),
                            str(stored_max_targets),
                            int(self.hparams.max_residues_per_target),
                            stored_max_residues,
                            int(self.hparams.template_chunk_length),
                            int(self.hparams.template_chunk_stride),
                            int(self.hparams.template_chunk_max_windows),
                            stored_chunk_length,
                            stored_chunk_stride,
                            stored_chunk_max_windows,
                            stored_allow_self_fallback,
                            stored_exclude_self,
                        )
                except Exception:
                    should_precompute = True
                    log.warning("Failed to read existing template payload at %s. Recomputing.", template_path)

            if should_precompute:
                log.info(
                    "Precomputing chunk template payload at %s...",
                    template_path,
                )
                precompute_template_coords(
                    labels_path=labels_path,
                    output_path=template_path,
                    top_k_store=self.hparams.template_topk_count,
                    max_residues_per_target=self.hparams.max_residues_per_target,
                    max_targets=self.hparams.max_targets,
                    exclude_self=True,
                    enforce_min_topk=True,
                    chunk_length=self.hparams.template_chunk_length,
                    chunk_stride=self.hparams.template_chunk_stride,
                    chunk_max_windows=self.hparams.template_chunk_max_windows,
                    min_percent_identity=self.hparams.template_min_percent_identity,
                    min_similarity=self.hparams.template_min_similarity,
                    protenix_fallback_zip=Path(self.hparams.data_dir) / self.hparams.template_protenix_fallback_zip,
                    protenix_base_confidence=self.hparams.template_protenix_base_confidence,
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
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, list[list[str]]],
    ]:
        if not self.hparams.use_template_coords:
            return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        if bool(self.hparams.template_force_oracle_only):
            return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        template_path = Path(self.hparams.data_dir) / self.hparams.template_file
        if not template_path.exists():
            log.warning("Template file not found at %s. Continuing without template coordinates.", template_path)
            return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        payload = torch.load(template_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "templates" in payload:
            templates = payload.get("templates", {})
            available = payload.get("available", {})
            chunk_topk_templates = payload.get("chunk_topk_templates", {})
            chunk_mask = payload.get("chunk_mask", {})
            chunk_start = payload.get("chunk_start", {})
            chunk_window_valid = payload.get("chunk_window_valid", {})
            chunk_topk_valid = payload.get("chunk_topk_valid", {})
            chunk_topk_identity = payload.get("chunk_topk_identity", {})
            chunk_topk_similarity = payload.get("chunk_topk_similarity", {})
            chunk_topk_confidence = payload.get("chunk_topk_confidence", {})
            chunk_topk_source_onehot = payload.get("chunk_topk_source_onehot", {})
            chunk_topk_residue_idx = payload.get("chunk_topk_residue_idx", {})
            chunk_topk_sources = payload.get("chunk_topk_sources", {})
            return (
                templates,
                available,
                chunk_topk_templates,
                chunk_mask,
                chunk_start,
                chunk_window_valid,
                chunk_topk_valid,
                chunk_topk_identity,
                chunk_topk_similarity,
                chunk_topk_confidence,
                chunk_topk_source_onehot,
                chunk_topk_residue_idx,
                chunk_topk_sources,
            )

        log.warning("Template file at %s has unexpected format. Continuing without templates.", template_path)
        return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

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
        failing: list[tuple[str, int, int, int]] = []
        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            active_windows = torch.nonzero(example.template_chunk_window_valid, as_tuple=False).squeeze(-1)
            if int(active_windows.numel()) <= 0:
                failing.append((example.target_id, 0, 0, 0))
                if len(failing) >= 10:
                    break
                continue
            for w_idx in active_windows.tolist():
                count = int(example.template_chunk_valid[w_idx].sum().item())
                min_count = min(min_count, count)
                if count < required:
                    failing.append((example.target_id, int(w_idx), count, int(active_windows.numel())))
                    if len(failing) >= 10:
                        break
            if len(failing) >= 10:
                break

        if failing:
            details = "; ".join(
                [f"{tid}(window={w_idx},valid={cnt},total_windows={total_w})" for tid, w_idx, cnt, total_w in failing]
            )
            raise RuntimeError(
                f"Template coverage check failed for split='{split_name}'. "
                f"Need >= {required} non-self templates per active 512/256 chunk window. "
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
                f"train_intersect_val={train_val[:5]}, "
                f"train_intersect_test={train_test[:5]}, "
                f"val_intersect_test={val_test[:5]}"
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
        chunk_sources_by_target: dict[str, list[list[str]]],
        allowed_source_ids: set[str],
    ) -> None:
        if not self.hparams.use_template_coords:
            return
        if not chunk_sources_by_target:
            raise RuntimeError(
                "Template payload is missing `chunk_topk_sources`; cannot enforce split leakage guards for templates."
            )

        required = max(1, int(self.hparams.template_topk_count))
        base, indices = self._split_indices(split)
        pruned_count = 0
        pruned_self_count = 0

        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            w_count = int(example.template_chunk_valid.shape[0])
            k_count = int(example.template_chunk_valid.shape[1])
            source_windows = [list(row) for row in chunk_sources_by_target.get(example.target_id, [])]
            if len(source_windows) < w_count:
                source_windows.extend([[""] * k_count for _ in range(w_count - len(source_windows))])

            for w_idx in range(w_count):
                if not bool(example.template_chunk_window_valid[w_idx].item()):
                    continue
                row_sources = source_windows[w_idx] if w_idx < len(source_windows) else []
                if len(row_sources) < k_count:
                    row_sources.extend([""] * (k_count - len(row_sources)))

                for k_idx in range(k_count):
                    source_id = row_sources[k_idx]
                    keep = bool(example.template_chunk_valid[w_idx, k_idx].item())
                    is_protenix = str(source_id).startswith("protenix:")
                    is_oracle = str(source_id).startswith("oracle:")
                    if is_protenix or is_oracle:
                        keep = keep
                    else:
                        keep = keep and source_id != "" and source_id in allowed_source_ids and source_id != example.target_id
                    if not keep:
                        if bool(example.template_chunk_valid[w_idx, k_idx].item()):
                            pruned_count += 1
                            if source_id == example.target_id:
                                pruned_self_count += 1
                        example.template_chunk_valid[w_idx, k_idx] = False
                        example.template_chunk_identity[w_idx, k_idx] = 0.0
                        example.template_chunk_similarity[w_idx, k_idx] = 0.0
                        example.template_chunk_confidence[w_idx, k_idx] = 0.0
                        example.template_chunk_source_onehot[w_idx, k_idx].zero_()
                        example.template_chunk_coords[w_idx, k_idx].zero_()
                        example.template_chunk_residue_idx[w_idx, k_idx].fill_(4)
                        row_sources[k_idx] = ""

                qualified = example.template_chunk_valid[w_idx].clone()
                qualified_count = int(qualified.sum().item())
                if 0 < qualified_count < required:
                    qualified_idx = torch.nonzero(qualified, as_tuple=False).squeeze(-1)
                    best_local = int(torch.argmax(example.template_chunk_identity[w_idx, qualified_idx]).item())
                    best_idx = int(qualified_idx[best_local].item())
                    fill_coords = example.template_chunk_coords[w_idx, best_idx].clone()
                    fill_identity = float(example.template_chunk_identity[w_idx, best_idx].item())
                    fill_similarity = float(example.template_chunk_similarity[w_idx, best_idx].item())
                    fill_confidence = float(example.template_chunk_confidence[w_idx, best_idx].item())
                    fill_source_onehot = example.template_chunk_source_onehot[w_idx, best_idx].clone()
                    fill_residue_idx = example.template_chunk_residue_idx[w_idx, best_idx].clone()
                    fill_source = row_sources[best_idx] if best_idx < len(row_sources) else ""
                    for k_idx in range(k_count):
                        if qualified_count >= required:
                            break
                        if bool(qualified[k_idx].item()):
                            continue
                        example.template_chunk_coords[w_idx, k_idx] = fill_coords
                        example.template_chunk_valid[w_idx, k_idx] = True
                        example.template_chunk_identity[w_idx, k_idx] = fill_identity
                        example.template_chunk_similarity[w_idx, k_idx] = fill_similarity
                        example.template_chunk_confidence[w_idx, k_idx] = fill_confidence
                        example.template_chunk_source_onehot[w_idx, k_idx] = fill_source_onehot
                        example.template_chunk_residue_idx[w_idx, k_idx] = fill_residue_idx
                        row_sources[k_idx] = fill_source
                        qualified[k_idx] = True
                        qualified_count += 1

                source_windows[w_idx] = row_sources[:k_count]

            (
                rebuilt_template_coords,
                rebuilt_topk_coords,
                rebuilt_topk_valid,
                rebuilt_topk_identity,
                rebuilt_topk_similarity,
                rebuilt_topk_residue_idx,
                rebuilt_has_template,
            ) = _rebuild_template_views_from_chunks(
                seq_len=int(example.coords.shape[0]),
                chunk_coords=example.template_chunk_coords,
                chunk_mask=example.template_chunk_mask,
                chunk_start=example.template_chunk_start,
                chunk_window_valid=example.template_chunk_window_valid,
                chunk_valid=example.template_chunk_valid,
                chunk_identity=example.template_chunk_identity,
                chunk_similarity=example.template_chunk_similarity,
                chunk_residue_idx=example.template_chunk_residue_idx,
                topk_count=int(example.template_topk_valid.shape[0]),
            )
            example.template_coords = rebuilt_template_coords
            example.template_topk_coords = rebuilt_topk_coords
            example.template_topk_valid = rebuilt_topk_valid
            example.template_topk_identity = rebuilt_topk_identity
            example.template_topk_similarity = rebuilt_topk_similarity
            example.template_topk_residue_idx = rebuilt_topk_residue_idx
            example.has_template = rebuilt_has_template
            chunk_sources_by_target[example.target_id] = source_windows[:w_count]

        log.info(
            "Chunk template source restriction applied for split='%s': pruned %d candidate templates (self-source pruned=%d).",
            split_name,
            pruned_count,
            pruned_self_count,
        )

    def _validate_no_self_template_sources(
        self,
        split: Dataset,
        split_name: str,
        chunk_sources_by_target: dict[str, list[list[str]]],
    ) -> None:
        if not self.hparams.use_template_coords:
            return

        base, indices = self._split_indices(split)
        violations: list[tuple[str, int]] = []
        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            source_windows = chunk_sources_by_target.get(example.target_id, [])
            w_count = int(example.template_chunk_valid.shape[0])
            k_count = int(example.template_chunk_valid.shape[1])
            for w_idx in range(w_count):
                if not bool(example.template_chunk_window_valid[w_idx].item()):
                    continue
                row_sources = source_windows[w_idx] if w_idx < len(source_windows) else []
                for k_idx in range(k_count):
                    if not bool(example.template_chunk_valid[w_idx, k_idx].item()):
                        continue
                    source_id = row_sources[k_idx] if k_idx < len(row_sources) else ""
                    if (not str(source_id).startswith("protenix:")) and source_id == example.target_id:
                        violations.append((example.target_id, k_idx))
                        if len(violations) >= 10:
                            break
                if len(violations) >= 10:
                    break
            if len(violations) >= 10:
                break

        if violations:
            details = ", ".join([f"{tid}[k={k_idx}]" for tid, k_idx in violations])
            raise RuntimeError(
                f"Self-source template leakage detected in split='{split_name}'. "
                f"Found chunk template source == target_id in active candidates: {details}"
            )

        log.info("No self-source chunk template candidates detected for split='%s'.", split_name)

    def _filter_split_min_templates(self, split: Dataset, split_name: str, min_required: int = 1) -> Dataset:
        if not self.hparams.use_template_coords:
            return split

        required = max(1, int(min_required))
        base, indices = self._split_indices(split)
        kept_indices: list[int] = []
        dropped = 0

        for idx in indices:
            example: RNAExample = base[idx]  # type: ignore[assignment]
            active_windows = torch.nonzero(example.template_chunk_window_valid, as_tuple=False).squeeze(-1)
            if int(active_windows.numel()) <= 0:
                dropped += 1
                continue
            has_all = True
            for w_idx in active_windows.tolist():
                if int(example.template_chunk_valid[w_idx].sum().item()) < required:
                    has_all = False
                    break
            if has_all:
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
                template_chunk_coords_by_target,
                template_chunk_mask_by_target,
                template_chunk_start_by_target,
                template_chunk_window_valid_by_target,
                template_chunk_valid_by_target,
                template_chunk_identity_by_target,
                template_chunk_similarity_by_target,
                template_chunk_confidence_by_target,
                template_chunk_source_onehot_by_target,
                template_chunk_residue_idx_by_target,
                template_chunk_sources_by_target,
            ) = self._load_template_payload()
            dataset = RNAIdentityDataset(
                labels_path=labels_path,
                max_residues_per_target=self.hparams.max_residues_per_target,
                max_targets=self.hparams.max_targets,
                template_coords_by_target=template_coords_by_target,
                template_available_by_target=template_available_by_target,
                template_chunk_coords_by_target=template_chunk_coords_by_target,
                template_chunk_mask_by_target=template_chunk_mask_by_target,
                template_chunk_start_by_target=template_chunk_start_by_target,
                template_chunk_window_valid_by_target=template_chunk_window_valid_by_target,
                template_chunk_valid_by_target=template_chunk_valid_by_target,
                template_chunk_identity_by_target=template_chunk_identity_by_target,
                template_chunk_similarity_by_target=template_chunk_similarity_by_target,
                template_chunk_confidence_by_target=template_chunk_confidence_by_target,
                template_chunk_source_onehot_by_target=template_chunk_source_onehot_by_target,
                template_chunk_residue_idx_by_target=template_chunk_residue_idx_by_target,
                template_topk_count=self.hparams.template_topk_count,
                template_chunk_length=self.hparams.template_chunk_length,
                template_chunk_stride=self.hparams.template_chunk_stride,
                template_chunk_max_windows=self.hparams.template_chunk_max_windows,
                template_min_percent_identity=self.hparams.template_min_percent_identity,
                template_force_oracle_only=self.hparams.template_force_oracle_only,
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
            if bool(self.hparams.template_force_oracle_only):
                log.warning(
                    "Oracle-only diagnostic mode enabled: bypassing split template-source restriction and self-source leakage checks."
                )
            else:
                all_ids = train_ids | val_ids | test_ids
                self._restrict_split_template_sources(
                    self.data_train,
                    "train",
                    chunk_sources_by_target=template_chunk_sources_by_target,
                    allowed_source_ids=train_ids,
                )
                self._validate_no_self_template_sources(
                    self.data_train,
                    "train",
                    chunk_sources_by_target=template_chunk_sources_by_target,
                )
                self._restrict_split_template_sources(
                    self.data_val,
                    "val",
                    chunk_sources_by_target=template_chunk_sources_by_target,
                    allowed_source_ids=all_ids,
                )
                self._restrict_split_template_sources(
                    self.data_test,
                    "test",
                    chunk_sources_by_target=template_chunk_sources_by_target,
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
