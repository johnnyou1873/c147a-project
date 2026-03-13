from __future__ import annotations

import csv
import logging
import math
import zlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import Dataset, Subset, random_split

from src.data.eternafold_bpp import (
    EternaFoldCacheWarmer,
    load_or_compute_eternafold_bpp_banded,
    prune_stale_eternafold_cache_locks,
    resolve_eternafold_binary,
    resolve_eternafold_cache_dir,
    resolve_eternafold_parameters,
)
from src.data.kaggle_sequence_metadata import (
    canonicalize_rna_sequence,
    load_kaggle_sequence_records,
    resolve_sequences_path,
)
from src.data.precomputed_rna_msa import build_precomputed_rna_msa_tensors
from src.data.precompute_full_length_templates import (
    FULL_LENGTH_SELECTION_POLICY,
    precompute_full_length_template_coords,
)
from src.data.precompute_templates import _build_oracle_diverse_candidate
from src.data.rna_datamodule_base import RESNAME_TO_IDX, RNADataModuleBase


log = logging.getLogger(__name__)

_LEGACY_FULL_LENGTH_SELECTION_POLICIES = {
    "full_length_topk_non_self_by_similarity_rank_identity_gate",
}

_FullTemplatePayload = tuple[
    dict[str, torch.Tensor],
    dict[str, bool],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, list[str]],
]


def _resolve_msa_rows(topk_count: int, rna_msa_max_rows: Optional[int], use_rna_msa_features: bool = True) -> int:
    if not use_rna_msa_features:
        return 1
    return max(1, int(rna_msa_max_rows) if rna_msa_max_rows is not None else int(topk_count) + 1)


def _empty_full_template_payload() -> _FullTemplatePayload:
    return {}, {}, {}, {}, {}, {}, {}, {}


def _parse_optional_coord_triplet(row: dict[str, Any]) -> tuple[float, float, float, bool]:
    raw_values = [str(row.get(key, "")).strip() for key in ("x_1", "y_1", "z_1")]
    if any(value == "" for value in raw_values):
        return 0.0, 0.0, 0.0, False

    try:
        coords = [float(value) for value in raw_values]
    except ValueError:
        return 0.0, 0.0, 0.0, False

    if not all(math.isfinite(value) for value in coords):
        return 0.0, 0.0, 0.0, False
    return float(coords[0]), float(coords[1]), float(coords[2]), True


@dataclass
class RNAFullTemplateExample:
    target_id: str
    residue_idx: torch.Tensor
    chain_idx: torch.Tensor
    copy_idx: torch.Tensor
    resid_norm: torch.Tensor
    coords: torch.Tensor
    target_mask: torch.Tensor
    template_coords: torch.Tensor
    template_topk_coords: torch.Tensor
    template_topk_valid: torch.Tensor
    template_topk_identity: torch.Tensor
    template_topk_similarity: torch.Tensor
    template_topk_residue_idx: torch.Tensor
    template_topk_sources: list[str]
    rna_msa_tokens: Optional[torch.Tensor]
    rna_msa_mask: Optional[torch.Tensor]
    rna_msa_row_valid: Optional[torch.Tensor]
    rna_msa_profile: Optional[torch.Tensor]
    rna_bpp_banded: Optional[torch.Tensor]
    has_template: bool


@dataclass
class _FullTemplateTargetRecord:
    target_id: str
    sequence: str
    residue_idx: torch.Tensor
    chain_idx: torch.Tensor
    copy_idx: torch.Tensor
    resid_norm: torch.Tensor
    coords: torch.Tensor
    target_mask: torch.Tensor


@dataclass
class _FullTemplateTemplateState:
    template_coords: torch.Tensor
    template_topk_coords: torch.Tensor
    template_topk_valid: torch.Tensor
    template_topk_identity: torch.Tensor
    template_topk_similarity: torch.Tensor
    template_topk_residue_idx: torch.Tensor
    template_topk_sources: list[str]
    has_template: bool


def _rebuild_template_consensus(
    template_topk_coords: torch.Tensor,
    template_topk_valid: torch.Tensor,
    template_topk_similarity: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    seq_len = int(template_topk_coords.shape[1])
    if not bool(template_topk_valid.any().item()):
        return torch.zeros((seq_len, 3), dtype=torch.float32), False

    valid_idx = torch.nonzero(template_topk_valid, as_tuple=False).squeeze(-1)
    coords = template_topk_coords[valid_idx].to(dtype=torch.float32)
    weights = template_topk_similarity[valid_idx].to(dtype=torch.float32).clamp(min=1e-4)
    weights = weights / weights.sum().clamp(min=1e-6)
    consensus = (coords * weights[:, None, None]).sum(dim=0)
    return consensus, True


def _oracle_template_seed(target_id: str, slot_idx: int) -> int:
    raw = f"{target_id}:{int(slot_idx)}".encode("utf-8")
    return int(zlib.adler32(raw) & 0xFFFFFFFF)


def _fill_bpp_mask(mask: torch.Tensor, batch_idx: int, seq_len: int, bpp_take: int) -> None:
    if bpp_take <= 0 or seq_len <= 1:
        return
    row_idx = torch.arange(seq_len, device=mask.device)
    col_idx = torch.arange(bpp_take, device=mask.device)
    valid_span = torch.clamp(seq_len - row_idx - 1, min=0, max=bpp_take)
    mask[batch_idx, :seq_len, :bpp_take] = col_idx.unsqueeze(0) < valid_span.unsqueeze(1)


def _fill_missing_template_slots(
    *,
    target_id: str,
    template_topk_coords: torch.Tensor,
    template_topk_valid: torch.Tensor,
    template_topk_identity: torch.Tensor,
    template_topk_similarity: torch.Tensor,
    template_topk_residue_idx: torch.Tensor,
    template_topk_sources: list[str],
    required: int,
) -> None:
    valid_count = int(template_topk_valid.sum().item())
    if valid_count <= 0 or valid_count >= int(required):
        return

    valid_idx = torch.nonzero(template_topk_valid, as_tuple=False).squeeze(-1)
    ranking = template_topk_similarity[valid_idx] + 0.01 * template_topk_identity[valid_idx]
    best_idx = int(valid_idx[int(torch.argmax(ranking).item())].item())
    base_coords = template_topk_coords[best_idx].detach().cpu().numpy()
    base_identity = float(template_topk_identity[best_idx].item())
    base_similarity = float(template_topk_similarity[best_idx].item())
    base_residue_idx = template_topk_residue_idx[best_idx].clone()

    for k_idx in range(int(template_topk_valid.shape[0])):
        if valid_count >= int(required):
            break
        if bool(template_topk_valid[k_idx].item()):
            continue
        diverse_coords = _build_oracle_diverse_candidate(
            base_coords=base_coords,
            chunk_len=int(template_topk_coords.shape[1]),
            variant_rank=k_idx,
            seed_base=_oracle_template_seed(target_id, k_idx),
        )
        template_topk_coords[k_idx] = torch.from_numpy(diverse_coords.astype("float32", copy=False))
        template_topk_valid[k_idx] = True
        template_topk_identity[k_idx] = base_identity
        template_topk_similarity[k_idx] = base_similarity
        template_topk_residue_idx[k_idx] = base_residue_idx
        template_topk_sources[k_idx] = f"oracle:{target_id}:{k_idx}"
        valid_count += 1


def _collate_full_template_batch(
    batch: list[RNAFullTemplateExample],
    thermal_noise_sigma_angstrom: float = 0.0,
    add_thermal_noise: bool = False,
    use_template_only_inputs: bool = False,
    use_rna_msa_features: bool = True,
    rna_msa_max_rows: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item.coords.shape[0] for item in batch)
    topk_count = batch[0].template_topk_coords.shape[0] if batch else 1
    msa_rows = _resolve_msa_rows(
        topk_count=topk_count,
        rna_msa_max_rows=rna_msa_max_rows,
        use_rna_msa_features=use_rna_msa_features,
    )
    bpp_span = batch[0].rna_bpp_banded.shape[1] if batch else 1

    residue_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    chain_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    copy_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    resid = torch.zeros(batch_size, max_len, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_topk_coords = torch.zeros(batch_size, topk_count, max_len, 3, dtype=torch.float32)
    template_topk_mask = torch.zeros(batch_size, topk_count, max_len, dtype=torch.bool)
    template_topk_valid = torch.zeros(batch_size, topk_count, dtype=torch.bool)
    template_topk_identity = torch.zeros(batch_size, topk_count, dtype=torch.float32)
    template_topk_similarity = torch.zeros(batch_size, topk_count, dtype=torch.float32)
    template_topk_residue_idx = torch.full((batch_size, topk_count, max_len), 4, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    target_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    template_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    rna_msa_tokens = torch.full((batch_size, msa_rows, max_len), 4, dtype=torch.long)
    rna_msa_mask = torch.zeros(batch_size, msa_rows, max_len, dtype=torch.bool)
    rna_msa_row_valid = torch.zeros(batch_size, msa_rows, dtype=torch.bool)
    rna_msa_profile = torch.zeros(batch_size, max_len, 5, dtype=torch.float32)
    rna_bpp_banded = torch.zeros(batch_size, max_len, bpp_span, dtype=torch.float32)
    rna_bpp_mask = torch.zeros(batch_size, max_len, bpp_span, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = int(item.coords.shape[0])
        residue_idx[i, :seq_len] = item.residue_idx
        chain_idx[i, :seq_len] = item.chain_idx
        copy_idx[i, :seq_len] = item.copy_idx
        resid[i, :seq_len] = item.resid_norm
        coords[i, :seq_len] = item.coords
        target_mask[i, :seq_len] = item.target_mask
        template_coords[i, :seq_len] = item.template_coords
        template_topk_coords[i, :, :seq_len] = item.template_topk_coords
        template_topk_valid[i] = item.template_topk_valid
        template_topk_identity[i] = item.template_topk_identity
        template_topk_similarity[i] = item.template_topk_similarity
        template_topk_residue_idx[i, :, :seq_len] = item.template_topk_residue_idx
        template_topk_mask[i, :, :seq_len] = item.template_topk_valid.unsqueeze(-1)
        mask[i, :seq_len] = True
        template_mask[i, :seq_len] = bool(item.has_template)

        bpp_take = min(bpp_span, int(item.rna_bpp_banded.shape[1]))
        rna_bpp_banded[i, :seq_len, :bpp_take] = item.rna_bpp_banded[:seq_len, :bpp_take]
        _fill_bpp_mask(rna_bpp_mask, i, seq_len, bpp_take)

        if use_rna_msa_features:
            if (
                item.rna_msa_tokens is None
                or item.rna_msa_mask is None
                or item.rna_msa_row_valid is None
                or item.rna_msa_profile is None
            ):
                raise RuntimeError(
                    f"Missing FASTA-backed RNA MSA tensors for target '{item.target_id}'. "
                    "Full-template loading does not support template-derived RNA MSA fallback."
                )
            rows_take = min(msa_rows, int(item.rna_msa_tokens.shape[0]))
            rna_msa_tokens[i, :rows_take, :seq_len] = item.rna_msa_tokens[:rows_take, :seq_len]
            rna_msa_mask[i, :rows_take, :seq_len] = item.rna_msa_mask[:rows_take, :seq_len]
            rna_msa_row_valid[i, :rows_take] = item.rna_msa_row_valid[:rows_take]
            rna_msa_profile[i, :seq_len] = item.rna_msa_profile[:seq_len]

    target_coords = coords.clone()
    if use_template_only_inputs:
        coords = template_coords.clone()
        coords = coords * template_mask.unsqueeze(-1).float()

    if add_thermal_noise and thermal_noise_sigma_angstrom > 0.0:
        coord_noise_mask = template_mask if use_template_only_inputs else target_mask
        noise = torch.randn_like(coords) * thermal_noise_sigma_angstrom
        noise = noise * coord_noise_mask.unsqueeze(-1).float()
        coords = coords + noise

        template_noise = torch.randn_like(template_coords) * thermal_noise_sigma_angstrom
        template_noise = template_noise * template_mask.unsqueeze(-1).float()
        template_coords = template_coords + template_noise

        topk_noise = torch.randn_like(template_topk_coords) * thermal_noise_sigma_angstrom
        topk_noise = topk_noise * template_topk_mask.unsqueeze(-1).float()
        template_topk_coords = template_topk_coords + topk_noise

    return {
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "copy_idx": copy_idx,
        "resid": resid,
        "coords": coords,
        "target_mask": target_mask,
        "template_coords": template_coords,
        "template_mask": template_mask,
        "template_topk_coords": template_topk_coords,
        "template_topk_mask": template_topk_mask,
        "template_topk_valid": template_topk_valid,
        "template_topk_identity": template_topk_identity,
        "template_topk_similarity": template_topk_similarity,
        "template_topk_residue_idx": template_topk_residue_idx,
        "rna_msa_tokens": rna_msa_tokens,
        "rna_msa_mask": rna_msa_mask,
        "rna_msa_row_valid": rna_msa_row_valid,
        "rna_msa_profile": rna_msa_profile,
        "rna_bpp_banded": rna_bpp_banded,
        "rna_bpp_mask": rna_bpp_mask,
        "target_coords": target_coords,
        "mask": mask,
    }


class RNAFullTemplateDataset(Dataset[RNAFullTemplateExample]):
    def __init__(
        self,
        labels_path: str | Path,
        sequences_path: Optional[str | Path] = None,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = 256,
        template_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_available_by_target: Optional[dict[str, bool]] = None,
        template_topk_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_valid_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_identity_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_similarity_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_residue_idx_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_topk_sources_by_target: Optional[dict[str, list[str]]] = None,
        template_topk_count: int = 5,
        enforce_template_coverage: bool = True,
        template_force_oracle_only: bool = False,
        use_rna_bpp_features: bool = True,
        rna_bpp_max_span: int = 256,
        rna_bpp_cutoff: float = 1e-4,
        rna_bpp_binary_path: str | Path | None = None,
        rna_bpp_parameters_path: str | Path | None = None,
        rna_bpp_cache_dir: str | Path | None = None,
        rna_bpp_num_threads: int = 16,
        use_rna_msa_features: bool = True,
        rna_msa_max_rows: Optional[int] = None,
        rna_msa_fasta_dir: Optional[str | Path] = None,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.sequences_path = resolve_sequences_path(self.labels_path, sequences_path)
        self.max_residues_per_target = max_residues_per_target
        self.max_targets = max_targets
        self.template_coords_by_target = template_coords_by_target or {}
        self.template_available_by_target = template_available_by_target or {}
        self.template_topk_coords_by_target = template_topk_coords_by_target or {}
        self.template_topk_valid_by_target = template_topk_valid_by_target or {}
        self.template_topk_identity_by_target = template_topk_identity_by_target or {}
        self.template_topk_similarity_by_target = template_topk_similarity_by_target or {}
        self.template_topk_residue_idx_by_target = template_topk_residue_idx_by_target or {}
        self.template_topk_sources_by_target = template_topk_sources_by_target or {}
        self.template_topk_count = max(1, int(template_topk_count))
        self.enforce_template_coverage = bool(enforce_template_coverage)
        self.template_force_oracle_only = bool(template_force_oracle_only)
        self.use_rna_bpp_features = bool(use_rna_bpp_features)
        self.rna_bpp_max_span = max(1, int(rna_bpp_max_span))
        self.rna_bpp_cutoff = float(rna_bpp_cutoff)
        self.rna_bpp_num_threads = max(1, int(rna_bpp_num_threads))
        self.rna_bpp_binary_path = (
            resolve_eternafold_binary(rna_bpp_binary_path) if self.use_rna_bpp_features else None
        )
        self.rna_bpp_parameters_path = (
            resolve_eternafold_parameters(
                parameters_path=rna_bpp_parameters_path,
                binary_path=self.rna_bpp_binary_path.parent if self.rna_bpp_binary_path is not None else None,
            )
            if self.use_rna_bpp_features
            else None
        )
        self.rna_bpp_cache_dir = (
            resolve_eternafold_cache_dir(rna_bpp_cache_dir) if self.use_rna_bpp_features else None
        )
        self._rna_bpp_warmer: Optional[EternaFoldCacheWarmer] = None
        self._rna_bpp_stale_locks_pruned = False
        self.use_rna_msa_features = bool(use_rna_msa_features)
        self.rna_msa_max_rows = rna_msa_max_rows
        self.rna_msa_fasta_dir = Path(rna_msa_fasta_dir) if rna_msa_fasta_dir else None
        if self.use_rna_msa_features and self.rna_msa_fasta_dir is None:
            raise ValueError("use_rna_msa_features=True requires `rna_msa_fasta_dir` to be set.")
        self.sequence_records = load_kaggle_sequence_records(self.sequences_path) if self.sequences_path is not None else {}
        self._example_cache: dict[int, RNAFullTemplateExample] = {}
        self._template_state_cache: dict[str, _FullTemplateTemplateState] = {}
        self._rna_msa_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.records, self.num_chain_types, self.max_copy_number, self.num_examples_with_template = (
            self._load_target_records()
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Windows dataloader workers pickle the dataset. The cache warmer owns
        # thread primitives and must stay only in the parent process.
        state["_rna_bpp_warmer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._rna_bpp_warmer = None

    @property
    def msa_rows(self) -> int:
        return _resolve_msa_rows(
            topk_count=self.template_topk_count,
            rna_msa_max_rows=self.rna_msa_max_rows,
            use_rna_msa_features=self.use_rna_msa_features,
        )

    def _load_target_records(self) -> tuple[list[_FullTemplateTargetRecord], int, int, int]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Could not find labels file: {self.labels_path}")

        rows_by_target: dict[str, list[tuple[int, int, str, int, str, int, float, float, float, bool]]] = {}
        skipped_overlong_targets: set[str] = set()
        chain_values: set[str] = set()
        max_copy = 0
        with self.labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_idx, row in enumerate(reader):
                full_id = row.get("ID", "")
                if "_" not in full_id:
                    continue
                target_id, pos_token = full_id.rsplit("_", 1)
                if target_id in skipped_overlong_targets:
                    continue
                if target_id not in rows_by_target:
                    if self.max_targets is not None and len(rows_by_target) >= self.max_targets:
                        continue
                    rows_by_target[target_id] = []
                target_rows = rows_by_target[target_id]
                if len(target_rows) >= self.max_residues_per_target:
                    rows_by_target.pop(target_id, None)
                    skipped_overlong_targets.add(target_id)
                    continue

                resname = str(row.get("resname", "N")).upper()
                chain = str(row.get("chain", "UNK"))
                try:
                    pos_idx = int(float(pos_token))
                    resid = int(float(row.get("resid", 0)))
                    copy = int(float(row.get("copy", 0)))
                except ValueError:
                    continue
                x, y, z, has_coords = _parse_optional_coord_triplet(row)

                target_rows.append((pos_idx, line_idx, resname, resid, chain, copy, x, y, z, has_coords))
                chain_values.add(chain)
                max_copy = max(max_copy, copy)

        if skipped_overlong_targets:
            log.info(
                "Skipped %d targets exceeding max_residues_per_target=%d while loading %s.",
                len(skipped_overlong_targets),
                int(self.max_residues_per_target),
                self.labels_path,
            )

        if self.sequence_records:
            for target_id in rows_by_target.keys():
                record = self.sequence_records.get(target_id)
                if record is None:
                    raise ValueError(
                        f"Target '{target_id}' was present in {self.labels_path} but missing from {self.sequences_path}."
                    )
                for segment in record.segments:
                    if segment.chain_id:
                        chain_values.add(segment.chain_id)

        chain_to_idx = {chain: idx for idx, chain in enumerate(sorted(chain_values))}
        records: list[_FullTemplateTargetRecord] = []
        num_examples_with_template = 0

        for target_id, rows in rows_by_target.items():
            rows.sort(key=lambda item: (item[0], item[1]))
            label_sequence = canonicalize_rna_sequence("".join(row[2] for row in rows))
            sequence_record = self.sequence_records.get(target_id)
            if sequence_record is not None:
                full_sequence = sequence_record.sequence
                if len(full_sequence) != len(rows):
                    raise ValueError(
                        f"Sequence length mismatch for target '{target_id}': labels provide {len(rows)} residues, "
                        f"but {self.sequences_path} provides {len(full_sequence)}."
                    )
                expected_sequence = full_sequence
                if label_sequence != expected_sequence:
                    raise ValueError(
                        f"Sequence mismatch for target '{target_id}' between {self.labels_path} and {self.sequences_path}."
                    )
            else:
                expected_sequence = label_sequence

            if self._target_has_template(target_id):
                num_examples_with_template += 1

            resid_t = torch.tensor([row[3] for row in rows], dtype=torch.float32)

            records.append(
                _FullTemplateTargetRecord(
                    target_id=target_id,
                    sequence=expected_sequence,
                    residue_idx=self._residue_idx_from_sequence(expected_sequence),
                    chain_idx=torch.tensor([chain_to_idx[row[4]] for row in rows], dtype=torch.long),
                    copy_idx=torch.tensor([max(0, row[5]) for row in rows], dtype=torch.long),
                    resid_norm=resid_t / resid_t.max().clamp(min=1.0),
                    coords=torch.tensor([[row[6], row[7], row[8]] for row in rows], dtype=torch.float32),
                    target_mask=torch.tensor([bool(row[9]) for row in rows], dtype=torch.bool),
                )
            )

        if not records:
            raise RuntimeError(f"No usable samples were found in {self.labels_path}")
        self._validate_msa_files(records)
        log.info("Indexed %d full-template targets from %s; examples will materialize lazily.", len(records), self.labels_path)
        return records, max(1, len(chain_to_idx)), max_copy, num_examples_with_template

    def __len__(self) -> int:
        return len(self.records)

    def _target_has_template(self, target_id: str) -> bool:
        if self.template_force_oracle_only:
            return True
        valid_t = self.template_topk_valid_by_target.get(target_id)
        if isinstance(valid_t, torch.Tensor) and bool(valid_t.detach().to(dtype=torch.bool, device="cpu").any().item()):
            return True
        return bool(self.template_available_by_target.get(target_id, False))

    def get_target_id(self, index: int) -> str:
        return self.records[int(index)].target_id

    def _residue_idx_from_sequence(self, sequence: str) -> torch.Tensor:
        return torch.tensor([RESNAME_TO_IDX.get(base, 4) for base in sequence], dtype=torch.long)

    def _validate_msa_files(self, records: list[_FullTemplateTargetRecord]) -> None:
        if not self.use_rna_msa_features:
            return
        missing: list[str] = []
        for record in records:
            msa_path = self.rna_msa_fasta_dir / f"{record.target_id}.MSA.fasta"
            if not msa_path.exists():
                missing.append(str(msa_path))
                if len(missing) >= 10:
                    break
        if missing:
            details = "; ".join(missing)
            raise FileNotFoundError(f"Missing required RNA MSA FASTA files. Examples: {details}")

    def _get_template_state(self, index: int) -> _FullTemplateTemplateState:
        record = self.records[int(index)]
        cached_state = self._template_state_cache.get(record.target_id)
        if cached_state is not None:
            return cached_state

        seq_len = len(record.sequence)
        residue_idx_t = record.residue_idx
        coords_t = record.coords
        template_coords = torch.zeros((seq_len, 3), dtype=torch.float32)
        template_topk_coords = torch.zeros((self.template_topk_count, seq_len, 3), dtype=torch.float32)
        template_topk_valid = torch.zeros((self.template_topk_count,), dtype=torch.bool)
        template_topk_identity = torch.zeros((self.template_topk_count,), dtype=torch.float32)
        template_topk_similarity = torch.zeros((self.template_topk_count,), dtype=torch.float32)
        template_topk_residue_idx = torch.full((self.template_topk_count, seq_len), 4, dtype=torch.long)
        template_topk_sources = [""] * self.template_topk_count

        target_id = record.target_id
        if target_id in self.template_coords_by_target:
            tmpl = self.template_coords_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
            if tmpl.ndim == 2 and int(tmpl.shape[-1]) == 3:
                take = min(seq_len, int(tmpl.shape[0]))
                template_coords[:take] = tmpl[:take]

        if target_id in self.template_topk_coords_by_target:
            topk_t = self.template_topk_coords_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
            if topk_t.ndim == 3 and int(topk_t.shape[-1]) == 3:
                k_take = min(self.template_topk_count, int(topk_t.shape[0]))
                l_take = min(seq_len, int(topk_t.shape[1]))
                template_topk_coords[:k_take, :l_take] = topk_t[:k_take, :l_take]

        if target_id in self.template_topk_valid_by_target:
            valid_t = self.template_topk_valid_by_target[target_id].detach().to(dtype=torch.bool, device="cpu")
            k_take = min(self.template_topk_count, int(valid_t.shape[0]))
            template_topk_valid[:k_take] = valid_t[:k_take]

        if target_id in self.template_topk_identity_by_target:
            ident_t = self.template_topk_identity_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
            k_take = min(self.template_topk_count, int(ident_t.shape[0]))
            template_topk_identity[:k_take] = ident_t[:k_take]

        if target_id in self.template_topk_similarity_by_target:
            sim_t = self.template_topk_similarity_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
            k_take = min(self.template_topk_count, int(sim_t.shape[0]))
            template_topk_similarity[:k_take] = sim_t[:k_take]

        if target_id in self.template_topk_residue_idx_by_target:
            ridx_t = self.template_topk_residue_idx_by_target[target_id].detach().to(dtype=torch.long, device="cpu")
            if ridx_t.ndim == 2:
                k_take = min(self.template_topk_count, int(ridx_t.shape[0]))
                l_take = min(seq_len, int(ridx_t.shape[1]))
                template_topk_residue_idx[:k_take, :l_take] = ridx_t[:k_take, :l_take]

        raw_sources = list(self.template_topk_sources_by_target.get(target_id, []))
        for k_idx in range(min(self.template_topk_count, len(raw_sources))):
            template_topk_sources[k_idx] = str(raw_sources[k_idx])

        if self.template_force_oracle_only:
            template_coords = coords_t.clone()
            template_topk_coords = coords_t.unsqueeze(0).repeat(self.template_topk_count, 1, 1)
            template_topk_valid = torch.ones((self.template_topk_count,), dtype=torch.bool)
            template_topk_identity = torch.full((self.template_topk_count,), 100.0, dtype=torch.float32)
            template_topk_similarity = torch.full((self.template_topk_count,), 1.0, dtype=torch.float32)
            template_topk_residue_idx = residue_idx_t.unsqueeze(0).repeat(self.template_topk_count, 1)
            template_topk_sources = [f"oracle:{target_id}:{k_idx}" for k_idx in range(self.template_topk_count)]

        if self.enforce_template_coverage:
            _fill_missing_template_slots(
                target_id=target_id,
                template_topk_coords=template_topk_coords,
                template_topk_valid=template_topk_valid,
                template_topk_identity=template_topk_identity,
                template_topk_similarity=template_topk_similarity,
                template_topk_residue_idx=template_topk_residue_idx,
                template_topk_sources=template_topk_sources,
                required=self.template_topk_count,
            )

        template_coords, has_template = _rebuild_template_consensus(
            template_topk_coords=template_topk_coords,
            template_topk_valid=template_topk_valid,
            template_topk_similarity=template_topk_similarity,
        )
        state = _FullTemplateTemplateState(
            template_coords=template_coords,
            template_topk_coords=template_topk_coords,
            template_topk_valid=template_topk_valid,
            template_topk_identity=template_topk_identity,
            template_topk_similarity=template_topk_similarity,
            template_topk_residue_idx=template_topk_residue_idx,
            template_topk_sources=template_topk_sources,
            has_template=bool(has_template or self.template_available_by_target.get(target_id, False)),
        )
        self._template_state_cache[target_id] = state
        return state

    def get_valid_template_count(self, index: int) -> int:
        return int(self._get_template_state(index).template_topk_valid.sum().item())

    def restrict_template_sources(self, index: int, allowed_source_ids: set[str], required: int) -> tuple[int, int]:
        record = self.records[int(index)]
        state = self._get_template_state(index)
        pruned_count = 0
        pruned_self_count = 0

        for k_idx in range(int(state.template_topk_valid.shape[0])):
            source_id = state.template_topk_sources[k_idx] if k_idx < len(state.template_topk_sources) else ""
            keep = bool(state.template_topk_valid[k_idx].item())
            is_oracle = str(source_id).startswith("oracle:")
            is_protenix = str(source_id).startswith("protenix:")
            if not (is_oracle or is_protenix):
                keep = keep and source_id != "" and source_id in allowed_source_ids and source_id != record.target_id
            if not keep:
                if bool(state.template_topk_valid[k_idx].item()):
                    pruned_count += 1
                    if source_id == record.target_id:
                        pruned_self_count += 1
                state.template_topk_valid[k_idx] = False
                state.template_topk_coords[k_idx].zero_()
                state.template_topk_identity[k_idx] = 0.0
                state.template_topk_similarity[k_idx] = 0.0
                state.template_topk_residue_idx[k_idx].fill_(4)
                if k_idx < len(state.template_topk_sources):
                    state.template_topk_sources[k_idx] = ""

        _fill_missing_template_slots(
            target_id=record.target_id,
            template_topk_coords=state.template_topk_coords,
            template_topk_valid=state.template_topk_valid,
            template_topk_identity=state.template_topk_identity,
            template_topk_similarity=state.template_topk_similarity,
            template_topk_residue_idx=state.template_topk_residue_idx,
            template_topk_sources=state.template_topk_sources,
            required=required,
        )

        state.template_coords, rebuilt_has_template = _rebuild_template_consensus(
            template_topk_coords=state.template_topk_coords,
            template_topk_valid=state.template_topk_valid,
            template_topk_similarity=state.template_topk_similarity,
        )
        state.has_template = bool(rebuilt_has_template)
        self._example_cache.pop(int(index), None)
        return pruned_count, pruned_self_count

    def has_self_template_source(self, index: int) -> Optional[int]:
        record = self.records[int(index)]
        state = self._get_template_state(index)
        for k_idx in range(int(state.template_topk_valid.shape[0])):
            if not bool(state.template_topk_valid[k_idx].item()):
                continue
            source_id = state.template_topk_sources[k_idx] if k_idx < len(state.template_topk_sources) else ""
            if (not str(source_id).startswith("oracle:")) and (not str(source_id).startswith("protenix:")) and source_id == record.target_id:
                return k_idx
        return None

    def _build_example(self, index: int) -> RNAFullTemplateExample:
        cached_example = self._example_cache.get(int(index))
        if cached_example is not None:
            return cached_example

        record = self.records[int(index)]
        template_state = self._get_template_state(index)
        example = RNAFullTemplateExample(
            target_id=record.target_id,
            residue_idx=record.residue_idx.clone(),
            chain_idx=record.chain_idx.clone(),
            copy_idx=record.copy_idx.clone(),
            resid_norm=record.resid_norm.clone(),
            coords=record.coords.clone(),
            target_mask=record.target_mask.clone(),
            template_coords=template_state.template_coords.clone(),
            template_topk_coords=template_state.template_topk_coords.clone(),
            template_topk_valid=template_state.template_topk_valid.clone(),
            template_topk_identity=template_state.template_topk_identity.clone(),
            template_topk_similarity=template_state.template_topk_similarity.clone(),
            template_topk_residue_idx=template_state.template_topk_residue_idx.clone(),
            template_topk_sources=list(template_state.template_topk_sources),
            rna_msa_tokens=None,
            rna_msa_mask=None,
            rna_msa_row_valid=None,
            rna_msa_profile=None,
            rna_bpp_banded=None,
            has_template=bool(template_state.has_template),
        )
        self._example_cache[int(index)] = example
        return example

    def _load_rna_bpp_banded(self, example: RNAFullTemplateExample) -> torch.Tensor:
        if example.rna_bpp_banded is not None:
            return example.rna_bpp_banded

        seq_len = int(example.coords.shape[0])
        if not self.use_rna_bpp_features:
            example.rna_bpp_banded = torch.zeros((seq_len, self.rna_bpp_max_span), dtype=torch.float32)
            return example.rna_bpp_banded

        example.rna_bpp_banded = load_or_compute_eternafold_bpp_banded(
            target_id=example.target_id,
            residue_idx=example.residue_idx,
            max_span=self.rna_bpp_max_span,
            cutoff=self.rna_bpp_cutoff,
            binary_path=self.rna_bpp_binary_path,
            parameters_path=self.rna_bpp_parameters_path,
            cache_dir=self.rna_bpp_cache_dir,
        )
        return example.rna_bpp_banded

    def _load_rna_msa_features(self, example: RNAFullTemplateExample) -> None:
        if not self.use_rna_msa_features:
            return
        if (
            example.rna_msa_tokens is not None
            and example.rna_msa_mask is not None
            and example.rna_msa_row_valid is not None
            and example.rna_msa_profile is not None
        ):
            return
        cached_msa = self._rna_msa_cache.get(example.target_id)
        if cached_msa is not None:
            (
                example.rna_msa_tokens,
                example.rna_msa_mask,
                example.rna_msa_row_valid,
                example.rna_msa_profile,
            ) = cached_msa
            return
        (
            example.rna_msa_tokens,
            example.rna_msa_mask,
            example.rna_msa_row_valid,
            example.rna_msa_profile,
        ) = build_precomputed_rna_msa_tensors(
            msa_path=self.rna_msa_fasta_dir / f"{example.target_id}.MSA.fasta",
            query_tokens=example.residue_idx,
            max_rows=self.msa_rows,
        )
        self._rna_msa_cache[example.target_id] = (
            example.rna_msa_tokens,
            example.rna_msa_mask,
            example.rna_msa_row_valid,
            example.rna_msa_profile,
        )

    def warm_rna_bpp_cache(self, indices: Optional[Iterable[int]] = None) -> None:
        if not self.use_rna_bpp_features:
            return
        selected_indices = list(range(len(self.records))) if indices is None else [int(example_idx) for example_idx in indices]
        if not selected_indices:
            return
        if not self._rna_bpp_stale_locks_pruned:
            prune_stale_eternafold_cache_locks(self.rna_bpp_cache_dir)
            self._rna_bpp_stale_locks_pruned = True
        if self._rna_bpp_warmer is None:
            self._rna_bpp_warmer = EternaFoldCacheWarmer(
                max_workers=self.rna_bpp_num_threads,
                max_span=self.rna_bpp_max_span,
                cutoff=self.rna_bpp_cutoff,
                binary_path=self.rna_bpp_binary_path,
                parameters_path=self.rna_bpp_parameters_path,
                cache_dir=self.rna_bpp_cache_dir,
            )
        self._rna_bpp_warmer.submit_many(
            (
                self.records[idx].target_id,
                self.records[idx].residue_idx,
            )
            for idx in selected_indices
        )
        log.info(
            "Queued EternaFold warmup for %d training targets with %d workers; cache=%s.",
            len(selected_indices),
            self.rna_bpp_num_threads,
            self.rna_bpp_cache_dir,
        )

    def shutdown_background_workers(self, wait: bool = False) -> None:
        if self._rna_bpp_warmer is not None:
            self._rna_bpp_warmer.shutdown(wait=wait)
            self._rna_bpp_warmer = None

    def __getitem__(self, index: int) -> RNAFullTemplateExample:
        example = self._build_example(int(index))
        self._load_rna_bpp_banded(example)
        self._load_rna_msa_features(example)
        return example


class C147AFullTemplateDataModule(RNADataModuleBase):
    """Separate AF3-style datamodule with full-length templates and Kaggle FASTA MSAs."""

    @property
    def _uses_full_templates(self) -> bool:
        return bool(self.hparams.use_template_coords) and (not bool(self.hparams.template_force_oracle_only))

    def _required_msa_dir(self) -> Path:
        fasta_dir = str(getattr(self.hparams, "rna_msa_fasta_dir", "")).strip()
        if not fasta_dir:
            raise ValueError("use_rna_msa_features=True requires `rna_msa_fasta_dir` to be set.")
        fasta_path = Path(fasta_dir)
        if not fasta_path.exists():
            raise FileNotFoundError(f"RNA MSA FASTA directory does not exist: {fasta_path}")
        return fasta_path

    def _validate_rna_msa_configuration(self) -> None:
        if not bool(self.hparams.use_rna_msa_features):
            return
        log.info("RNA MSA: using Kaggle FASTA MSAs from %s.", self._required_msa_dir())

    def _build_collate_fn(self, add_thermal_noise: bool, use_template_only_inputs: bool) -> Any:
        return partial(
            _collate_full_template_batch,
            thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
            add_thermal_noise=add_thermal_noise,
            use_template_only_inputs=use_template_only_inputs,
            use_rna_msa_features=self.hparams.use_rna_msa_features,
            rna_msa_max_rows=self.hparams.rna_msa_max_rows,
        )

    def _dataset_kwargs(
        self,
        labels_path: Path,
        sequences_path: Optional[Path],
        template_payload: _FullTemplatePayload,
    ) -> dict[str, Any]:
        (
            template_coords_by_target,
            template_available_by_target,
            template_topk_coords_by_target,
            template_topk_valid_by_target,
            template_topk_identity_by_target,
            template_topk_similarity_by_target,
            template_topk_residue_idx_by_target,
            template_topk_sources_by_target,
        ) = template_payload
        return {
            "labels_path": labels_path,
            "sequences_path": sequences_path,
            "max_residues_per_target": self.hparams.max_residues_per_target,
            "max_targets": self.hparams.max_targets,
            "template_coords_by_target": template_coords_by_target,
            "template_available_by_target": template_available_by_target,
            "template_topk_coords_by_target": template_topk_coords_by_target,
            "template_topk_valid_by_target": template_topk_valid_by_target,
            "template_topk_identity_by_target": template_topk_identity_by_target,
            "template_topk_similarity_by_target": template_topk_similarity_by_target,
            "template_topk_residue_idx_by_target": template_topk_residue_idx_by_target,
            "template_topk_sources_by_target": template_topk_sources_by_target,
            "template_topk_count": self.hparams.template_topk_count,
            "enforce_template_coverage": True,
            "template_force_oracle_only": self.hparams.template_force_oracle_only,
            "use_rna_bpp_features": self.hparams.use_rna_bpp_features,
            "rna_bpp_max_span": self.hparams.rna_bpp_max_span,
            "rna_bpp_cutoff": self.hparams.rna_bpp_cutoff,
            "rna_bpp_binary_path": self.hparams.rna_bpp_binary_path,
            "rna_bpp_parameters_path": self.hparams.rna_bpp_parameters_path,
            "rna_bpp_cache_dir": Path(self.hparams.data_dir) / self.hparams.rna_bpp_cache_dir,
            "rna_bpp_num_threads": self.hparams.rna_bpp_num_threads,
            "use_rna_msa_features": self.hparams.use_rna_msa_features,
            "rna_msa_max_rows": self.hparams.rna_msa_max_rows,
            "rna_msa_fasta_dir": self.hparams.rna_msa_fasta_dir,
        }

    def _template_compatibility_message(
        self,
        template_path: Path,
        template_search_pool_max_targets: Optional[int],
    ) -> tuple[bool, Optional[str], Optional[str]]:
        payload = self._read_template_file_payload(template_path)
        has_required_fields = (
            isinstance(payload, dict)
            and "templates" in payload
            and "available" in payload
            and "template_topk_coords" in payload
            and "template_topk_valid" in payload
            and "template_topk_identity" in payload
            and "template_topk_similarity" in payload
            and "template_topk_residue_idx" in payload
            and "template_topk_sources" in payload
        )
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        topk_store = int(meta.get("top_k_store", 0)) if isinstance(meta, dict) else 0
        selection_policy = str(meta.get("selection_policy", "")) if isinstance(meta, dict) else ""
        alignment_mode = str(meta.get("alignment_mode", "")) if isinstance(meta, dict) else ""
        search_strategy = str(meta.get("search_strategy", "")) if isinstance(meta, dict) else ""
        exclude_self = bool(meta.get("exclude_self", True)) if isinstance(meta, dict) else True
        raw_query_max_targets = meta.get("query_max_targets", meta.get("max_targets", None)) if isinstance(meta, dict) else None
        raw_search_pool_max_targets = (
            meta.get("search_pool_max_targets", meta.get("max_targets", None)) if isinstance(meta, dict) else None
        )
        stored_query_max_targets = None if raw_query_max_targets is None else int(raw_query_max_targets)
        stored_search_pool_max_targets = None if raw_search_pool_max_targets is None else int(raw_search_pool_max_targets)
        stored_max_residues = int(meta.get("max_residues_per_target", 0)) if isinstance(meta, dict) else 0
        stored_length_ratio_tolerance = float(meta.get("length_ratio_tolerance", 0.0)) if isinstance(meta, dict) else 0.0
        stored_min_percent_identity = float(meta.get("min_percent_identity", 0.0)) if isinstance(meta, dict) else 0.0
        stored_min_similarity = float(meta.get("min_similarity", 0.0)) if isinstance(meta, dict) else 0.0

        desired_query_max_targets = self.hparams.max_targets
        desired_search_pool_max_targets = template_search_pool_max_targets
        selection_policy_compatible = selection_policy in {
            FULL_LENGTH_SELECTION_POLICY,
            *_LEGACY_FULL_LENGTH_SELECTION_POLICIES,
        }
        query_coverage_insufficient = (
            (desired_query_max_targets is None and stored_query_max_targets is not None)
            or (
                desired_query_max_targets is not None
                and stored_query_max_targets is not None
                and int(stored_query_max_targets) < int(desired_query_max_targets)
            )
        )
        search_pool_coverage_insufficient = (
            (desired_search_pool_max_targets is None and stored_search_pool_max_targets is not None)
            or (
                desired_search_pool_max_targets is not None
                and stored_search_pool_max_targets is not None
                and int(stored_search_pool_max_targets) < int(desired_search_pool_max_targets)
            )
        )
        ratio_mismatch = abs(stored_length_ratio_tolerance - float(self.hparams.template_length_ratio_tolerance)) > 1e-8
        identity_mismatch = abs(stored_min_percent_identity - float(self.hparams.template_min_percent_identity)) > 1e-8
        similarity_mismatch = abs(stored_min_similarity - float(self.hparams.template_min_similarity)) > 1e-8

        if (
            (not has_required_fields)
            or topk_store < int(self.hparams.template_topk_count)
            or (not selection_policy_compatible)
            or alignment_mode != "global"
            or search_strategy != "full_exhaustive_alignment"
            or (not exclude_self)
            or query_coverage_insufficient
            or search_pool_coverage_insufficient
            or stored_max_residues < int(self.hparams.max_residues_per_target)
            or ratio_mismatch
            or identity_mismatch
            or similarity_mismatch
        ):
            message = (
                "Full-length template file at %s requires payload upgrade "
                "(need_topk=%d, found_topk=%d, need_policy=%s, found_policy=%s, "
                "need_alignment=global, found_alignment=%s, need_search=full_exhaustive_alignment, "
                "found_search=%s, exclude_self=%s, need_query_max_targets=%s, found_query_max_targets=%s, "
                "need_search_pool_max_targets=%s, found_search_pool_max_targets=%s, need_max_residues=%d, "
                "found_max_residues=%d, need_length_ratio_tolerance=%.3f, found_length_ratio_tolerance=%.3f, "
                "need_min_percent_identity=%.1f, found_min_percent_identity=%.1f, "
                "need_min_similarity=%.3f, found_min_similarity=%.3f). Recomputing."
            ) % (
                template_path,
                int(self.hparams.template_topk_count),
                topk_store,
                FULL_LENGTH_SELECTION_POLICY,
                selection_policy,
                alignment_mode,
                search_strategy,
                exclude_self,
                str(desired_query_max_targets),
                str(stored_query_max_targets),
                str(desired_search_pool_max_targets),
                str(stored_search_pool_max_targets),
                int(self.hparams.max_residues_per_target),
                stored_max_residues,
                float(self.hparams.template_length_ratio_tolerance),
                stored_length_ratio_tolerance,
                float(self.hparams.template_min_percent_identity),
                stored_min_percent_identity,
                float(self.hparams.template_min_similarity),
                stored_min_similarity,
            )
            return False, selection_policy, message

        return True, selection_policy, None

    def _split_target_id(self, base: Dataset, idx: int) -> str:
        if isinstance(base, RNAFullTemplateDataset):
            return base.get_target_id(idx)
        return base[idx].target_id

    def _split_template_count(self, base: Dataset, idx: int) -> int:
        if isinstance(base, RNAFullTemplateDataset):
            return base.get_valid_template_count(idx)
        return int(base[idx].template_topk_valid.sum().item())

    def prepare_data(self) -> None:
        self._validate_rna_msa_configuration()
        self._validate_rna_bpp_configuration()
        labels_path, sequences_path = self._resolved_labels_and_sequences()
        if not labels_path.exists():
            raise FileNotFoundError(f"Expected labels file at: {labels_path}")
        if sequences_path is not None and not sequences_path.exists():
            raise FileNotFoundError(f"Expected sequences file at: {sequences_path}")

        template_search_pool_max_targets = self.hparams.template_generation_max_targets
        if not self.hparams.use_template_coords:
            return
        if not self._uses_full_templates:
            log.warning(
                "template_force_oracle_only=True: skipping full-length template precompute and using oracle templates."
            )
            return

        template_path = self._template_path()
        should_precompute = (not template_path.exists()) and self.hparams.precompute_templates_if_missing

        if template_path.exists() and self.hparams.precompute_templates_if_missing:
            try:
                compatible, selection_policy, incompatibility_message = self._template_compatibility_message(
                    template_path=template_path,
                    template_search_pool_max_targets=template_search_pool_max_targets,
                )
                if not compatible:
                    should_precompute = True
                    log.info("%s", incompatibility_message)
                elif selection_policy != FULL_LENGTH_SELECTION_POLICY:
                    log.info(
                        "Using legacy full-length template payload at %s without recomputing "
                        "(policy=%s, requested_policy=%s).",
                        template_path,
                        selection_policy,
                        FULL_LENGTH_SELECTION_POLICY,
                    )
            except Exception:
                should_precompute = True
                log.warning("Failed to read existing full-length template payload at %s. Recomputing.", template_path)

        if should_precompute:
            log.info(
                "Precomputing full-length template payload at %s (query_max_targets=%s, search_pool_max_targets=%s)...",
                template_path,
                str(self.hparams.max_targets),
                str(template_search_pool_max_targets),
            )
            precompute_full_length_template_coords(
                labels_path=labels_path,
                sequences_path=sequences_path,
                output_path=template_path,
                top_k_store=self.hparams.template_topk_count,
                max_residues_per_target=self.hparams.max_residues_per_target,
                query_max_targets=self.hparams.max_targets,
                search_pool_max_targets=template_search_pool_max_targets,
                exclude_self=True,
                min_percent_identity=self.hparams.template_min_percent_identity,
                min_similarity=self.hparams.template_min_similarity,
                length_ratio_tolerance=self.hparams.template_length_ratio_tolerance,
                num_threads=self.hparams.template_precompute_num_threads,
            )

    def _load_full_template_payload(
        self,
    ) -> _FullTemplatePayload:
        if not self._uses_full_templates:
            return _empty_full_template_payload()

        template_path = self._template_path()
        if not template_path.exists():
            log.warning(
                "Full-length template file not found at %s. Continuing without template coordinates.",
                template_path,
            )
            return _empty_full_template_payload()

        payload = self._read_template_file_payload(template_path)
        if isinstance(payload, dict) and "templates" in payload:
            return (
                payload.get("templates", {}),
                payload.get("available", {}),
                payload.get("template_topk_coords", {}),
                payload.get("template_topk_valid", {}),
                payload.get("template_topk_identity", {}),
                payload.get("template_topk_similarity", {}),
                payload.get("template_topk_residue_idx", {}),
                payload.get("template_topk_sources", {}),
            )

        log.warning(
            "Full-length template file at %s has unexpected format. Continuing without templates.",
            template_path,
        )
        return _empty_full_template_payload()

    def _split_target_ids(self, split: Dataset) -> set[str]:
        base, indices = self._split_indices(split)
        if isinstance(base, RNAFullTemplateDataset):
            return {base.get_target_id(idx) for idx in indices}
        return super()._split_target_ids(split)

    def _restrict_split_template_sources(
        self,
        split: Dataset,
        split_name: str,
        template_sources_by_target: dict[str, list[str]],
        allowed_source_ids: set[str],
    ) -> None:
        if not self.hparams.use_template_coords:
            return
        if not template_sources_by_target:
            raise RuntimeError(
                "Full-length template payload is missing `template_topk_sources`; cannot enforce split leakage guards."
            )

        required = max(1, int(self.hparams.template_topk_count))
        base, indices = self._split_indices(split)
        pruned_count = 0
        pruned_self_count = 0

        for idx in indices:
            if isinstance(base, RNAFullTemplateDataset):
                local_pruned, local_self_pruned = base.restrict_template_sources(
                    index=idx,
                    allowed_source_ids=allowed_source_ids,
                    required=required,
                )
                target_id = base.get_target_id(idx)
                template_sources_by_target[target_id] = list(base._get_template_state(idx).template_topk_sources)
                pruned_count += local_pruned
                pruned_self_count += local_self_pruned
                continue

            example = base[idx]
            template_sources_by_target[example.target_id] = list(template_sources_by_target.get(example.target_id, []))

        log.info(
            "Full-length template source restriction applied for split='%s': pruned %d templates (self-source pruned=%d).",
            split_name,
            pruned_count,
            pruned_self_count,
        )

    def _validate_no_self_template_sources(
        self,
        split: Dataset,
        split_name: str,
        template_sources_by_target: dict[str, list[str]],
    ) -> None:
        if not self.hparams.use_template_coords:
            return

        base, indices = self._split_indices(split)
        violations: list[tuple[str, int]] = []
        for idx in indices:
            if isinstance(base, RNAFullTemplateDataset):
                target_id = base.get_target_id(idx)
                k_idx = base.has_self_template_source(idx)
                if k_idx is not None:
                    violations.append((target_id, k_idx))
                    if len(violations) >= 10:
                        break
                continue
            example = base[idx]
            sources = template_sources_by_target.get(example.target_id, [])
            for k_idx in range(int(example.template_topk_valid.shape[0])):
                if not bool(example.template_topk_valid[k_idx].item()):
                    continue
                source_id = sources[k_idx] if k_idx < len(sources) else ""
                if (not str(source_id).startswith("oracle:")) and (not str(source_id).startswith("protenix:")) and source_id == example.target_id:
                    violations.append((example.target_id, k_idx))
                    if len(violations) >= 10:
                        break
            if len(violations) >= 10:
                break

        if violations:
            details = ", ".join([f"{tid}[k={k_idx}]" for tid, k_idx in violations])
            raise RuntimeError(
                f"Self-source full-length template leakage detected in split='{split_name}': {details}"
            )

        log.info("No self-source full-length template candidates detected for split='%s'.", split_name)

    def _filter_split_min_templates(self, split: Dataset, split_name: str, min_required: int = 1) -> Dataset:
        if not self.hparams.use_template_coords:
            return split

        required = max(1, int(min_required))
        base, indices = self._split_indices(split)
        kept_indices: list[int] = []
        dropped = 0

        for idx in indices:
            count = self._split_template_count(base, idx)
            if count >= required:
                kept_indices.append(idx)
            else:
                dropped += 1

        if not kept_indices:
            raise RuntimeError(f"All samples dropped from split='{split_name}' after full-length template filtering.")

        if dropped > 0:
            log.info(
                "Dropped %d samples from split='%s' due to missing full-length templates (required=%d).",
                dropped,
                split_name,
                required,
            )
        return Subset(base, kept_indices)

    def _validate_split_template_coverage(self, split: Dataset, split_name: str) -> None:
        if not self.hparams.use_template_coords:
            return

        required = 1
        base, indices = self._split_indices(split)
        if not indices:
            return

        min_count = int(self.hparams.template_topk_count)
        failing: list[tuple[str, int]] = []
        for idx in indices:
            target_id = self._split_target_id(base, idx)
            count = self._split_template_count(base, idx)
            min_count = min(min_count, count)
            if count < required:
                failing.append((target_id, count))
                if len(failing) >= 10:
                    break

        if failing:
            details = "; ".join([f"{tid}(valid_templates={cnt})" for tid, cnt in failing])
            message = (
                f"Full-length template coverage check failed for split='{split_name}'. "
                f"Need >= {required} non-self full-length templates per target. Examples: {details}"
            )
            if split_name.strip().lower() == "train":
                raise RuntimeError(message)
            log.warning("%s Continuing because this split is not used to determine train viability.", message)
            return

        log.info(
            "Full-length template coverage check passed for split='%s': min valid templates=%d (required=%d).",
            split_name,
            min_count,
            required,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self._validate_rna_msa_configuration()
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None and self.data_val is None and self.data_test is None:
            labels_path, sequences_path = self._resolved_labels_and_sequences()
            template_payload = self._load_full_template_payload()
            dataset = RNAFullTemplateDataset(**self._dataset_kwargs(labels_path, sequences_path, template_payload))
            self._rna_dataset = dataset
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
                    "Oracle-only diagnostic mode enabled: bypassing split full-length template-source restriction and leakage checks."
                )
            else:
                all_ids = train_ids | val_ids | test_ids
                template_topk_sources_by_target = template_payload[7]
                self._restrict_split_template_sources(
                    self.data_train,
                    "train",
                    template_sources_by_target=template_topk_sources_by_target,
                    allowed_source_ids=all_ids,
                )
                self._validate_no_self_template_sources(
                    self.data_train,
                    "train",
                    template_sources_by_target=template_topk_sources_by_target,
                )
                self._restrict_split_template_sources(
                    self.data_val,
                    "val",
                    template_sources_by_target=template_topk_sources_by_target,
                    allowed_source_ids=all_ids,
                )
                self._restrict_split_template_sources(
                    self.data_test,
                    "test",
                    template_sources_by_target=template_topk_sources_by_target,
                    allowed_source_ids=all_ids,
                )

            self.data_train = self._filter_split_min_templates(self.data_train, "train", min_required=1)
            train_base, train_indices = self._split_indices(self.data_train)
            if train_base is dataset:
                self._warm_relevant_rna_bpp_cache(dataset, indices=train_indices)
            else:
                self._warm_relevant_rna_bpp_cache(dataset)
            self._validate_split_template_coverage(self.data_train, "train")
            self._validate_split_template_coverage(self.data_test, "test")

            log.info(
                "Full-length template datamodule ready: T=%.1fK, sigma=%.3fA (train=%s, eval=%s); template examples=%d/%d",
                self.hparams.temperature_k,
                self.thermal_noise_sigma_angstrom,
                self.hparams.apply_thermal_noise_train,
                self.hparams.apply_thermal_noise_eval,
                dataset.num_examples_with_template,
                len(dataset),
            )

if __name__ == "__main__":
    _ = C147AFullTemplateDataModule()
