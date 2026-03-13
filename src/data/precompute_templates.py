from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.Align import PairwiseAligner

from src.data.kaggle_sequence_metadata import (
    canonicalize_rna_sequence,
    load_kaggle_sequence_records,
    resolve_sequences_path,
)


log = logging.getLogger(__name__)

RNA_RESIDUES = {"A", "C", "G", "U", "T"}
RESNAME_TO_BASE = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U"}
RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
SOURCE_UNKNOWN = 0
SOURCE_TEMPLATE = 1
SOURCE_PROTENIX = 2
CHUNK_SELECTION_POLICY = (
    "chunked_topk_non_self_by_similarity_rank_identity_gate_with_protenix_then_oracle_diversity_fallback"
)


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _make_aligner(mode: str = "global") -> PairwiseAligner:
    # Mirrors src/kaggle.ipynb search/adaptation scoring.
    aligner = PairwiseAligner()
    aligner.mode = str(mode)
    aligner.match_score = 2
    aligner.mismatch_score = -1.5
    aligner.open_gap_score = -8
    aligner.extend_gap_score = -0.4
    for attr in [
        "open_left_deletion_score",
        "extend_left_deletion_score",
        "open_right_deletion_score",
        "extend_right_deletion_score",
        "open_left_insertion_score",
        "extend_left_insertion_score",
        "open_right_insertion_score",
        "extend_right_insertion_score",
    ]:
        setattr(aligner, attr, -8 if "open" in attr else -0.4)
    return aligner


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


def _split_target_and_pos(full_id: str) -> tuple[str, int]:
    if "_" not in full_id:
        return full_id, 0
    prefix, suffix = full_id.rsplit("_", 1)
    try:
        pos = int(float(suffix))
    except ValueError:
        pos = 0
    return prefix, pos


def _canonical_resname(raw: str) -> str:
    token = str(raw).upper()
    if token in RESNAME_TO_BASE:
        return RESNAME_TO_BASE[token]
    if token in RNA_RESIDUES:
        return token
    return "N"


def _canonical_base(base: str) -> str:
    token = str(base).upper()
    if token == "T":
        return "U"
    if token in {"A", "C", "G", "U", "N"}:
        return token
    return "N"


def _base_to_residue_idx(base: str) -> int:
    token = _canonical_base(base)
    return int(RESNAME_TO_IDX.get(token, 4))


def _fill_missing_coords(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    coords[np.abs(coords) > 1e10] = np.nan
    n = coords.shape[0]
    for i in range(n):
        if np.isfinite(coords[i, 0]):
            continue
        prev_idx = next((j for j in range(i - 1, -1, -1) if np.isfinite(coords[j, 0])), -1)
        next_idx = next((j for j in range(i + 1, n) if np.isfinite(coords[j, 0])), -1)
        if prev_idx >= 0 and next_idx >= 0:
            w = float(i - prev_idx) / float(next_idx - prev_idx)
            coords[i] = (1.0 - w) * coords[prev_idx] + w * coords[next_idx]
        elif prev_idx >= 0:
            coords[i] = coords[prev_idx] + np.array([3.0, 0.0, 0.0], dtype=np.float64)
        elif next_idx >= 0:
            coords[i] = coords[next_idx] + np.array([3.0, 0.0, 0.0], dtype=np.float64)
        else:
            coords[i] = np.array([float(i) * 3.0, 0.0, 0.0], dtype=np.float64)
    return np.nan_to_num(coords, nan=0.0)


def _coords_are_valid(coords: np.ndarray) -> bool:
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        return False
    spread = np.max(coords, axis=0) - np.min(coords, axis=0)
    if np.any(spread < 2.0):
        return False
    rms = float(np.sqrt(np.mean(coords**2)))
    if rms < 1.0:
        return False
    return True


def _load_sequences_and_coords(
    labels_path: Path,
    max_residues_per_target: int = 5120,
    max_targets: int | None = None,
    sequences_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    rows_by_target: dict[str, list[tuple[int, int, str, float, float, float]]] = {}
    skipped_overlong_targets: set[str] = set()
    resolved_sequences_path = resolve_sequences_path(labels_path=labels_path, sequences_path=sequences_path)
    sequence_records = (
        load_kaggle_sequence_records(resolved_sequences_path)
        if resolved_sequences_path is not None
        else {}
    )

    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_idx, row in enumerate(reader):
            target_id, pos = _split_target_and_pos(str(row.get("ID", "")))
            if not target_id:
                continue
            if target_id in skipped_overlong_targets:
                continue
            if target_id not in rows_by_target:
                if max_targets is not None and len(rows_by_target) >= max_targets:
                    continue
                rows_by_target[target_id] = []
            if len(rows_by_target[target_id]) >= max_residues_per_target:
                rows_by_target.pop(target_id, None)
                skipped_overlong_targets.add(target_id)
                continue

            resname = _canonical_resname(str(row.get("resname", "N")))
            try:
                x = float(row.get("x_1", 0.0))
                y = float(row.get("y_1", 0.0))
                z = float(row.get("z_1", 0.0))
            except ValueError:
                x, y, z = 0.0, 0.0, 0.0

            rows_by_target[target_id].append((pos, line_idx, resname, x, y, z))

    if skipped_overlong_targets:
        log.info(
            "Skipped %d targets exceeding max_residues_per_target=%d while loading %s.",
            len(skipped_overlong_targets),
            int(max_residues_per_target),
            labels_path,
        )

    sequences: dict[str, str] = {}
    coords_by_target: dict[str, np.ndarray] = {}
    for target_id, rows in rows_by_target.items():
        rows.sort(key=lambda r: (r[0], r[1]))  # ID suffix order, then file order.
        label_seq = canonicalize_rna_sequence("".join(r[2] for r in rows))
        if resolved_sequences_path is not None:
            record = sequence_records.get(target_id)
            if record is None:
                raise ValueError(f"Target '{target_id}' was present in {labels_path} but missing from {resolved_sequences_path}.")
            full_seq = record.sequence
            if len(full_seq) != len(rows):
                raise ValueError(
                    f"Sequence length mismatch for target '{target_id}': labels provide {len(rows)} residues, "
                    f"but {resolved_sequences_path} provides {len(full_seq)}."
                )
            seq = full_seq
            if label_seq != seq:
                raise ValueError(
                    f"Sequence mismatch for target '{target_id}' between {labels_path} and {resolved_sequences_path}."
                )
        else:
            seq = label_seq
        coords = np.asarray([[r[3], r[4], r[5]] for r in rows], dtype=np.float64)
        coords = _fill_missing_coords(coords)
        sequences[target_id] = seq
        coords_by_target[target_id] = coords

    return sequences, coords_by_target


def _compute_identity_percent(query_seq: str, template_seq: str, alignment: Any) -> float:
    identical = 0
    for (qs, qe), (ts, te) in zip(*alignment.aligned):
        for qp, tp in zip(range(qs, qe), range(ts, te)):
            if query_seq[qp] == template_seq[tp]:
                identical += 1
    return 100.0 * float(identical) / float(max(1, len(query_seq)))


def _adapt_template_to_query(
    query_seq: str,
    template_seq: str,
    template_coords: np.ndarray,
    alignment: Any,
) -> np.ndarray:
    out = np.full((len(query_seq), 3), np.nan, dtype=np.float64)
    for (qs, qe), (ts, te) in zip(*alignment.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs):
            out[qs:qe] = chunk

    for i in range(len(out)):
        if np.isfinite(out[i, 0]):
            continue
        prev_idx = next((j for j in range(i - 1, -1, -1) if np.isfinite(out[j, 0])), -1)
        next_idx = next((j for j in range(i + 1, len(out)) if np.isfinite(out[j, 0])), -1)
        if prev_idx >= 0 and next_idx >= 0:
            w = float(i - prev_idx) / float(next_idx - prev_idx)
            out[i] = (1.0 - w) * out[prev_idx] + w * out[next_idx]
        elif prev_idx >= 0:
            out[i] = out[prev_idx] + np.array([3.0, 0.0, 0.0], dtype=np.float64)
        elif next_idx >= 0:
            out[i] = out[next_idx] + np.array([3.0, 0.0, 0.0], dtype=np.float64)
        else:
            out[i] = np.array([float(i) * 3.0, 0.0, 0.0], dtype=np.float64)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _adapt_template_residue_idx_to_query(
    query_seq: str,
    template_seq: str,
    alignment: Any,
) -> np.ndarray:
    out = np.full((len(query_seq),), 4, dtype=np.int64)
    for (qs, qe), (ts, te) in zip(*alignment.aligned):
        seg_len = int(min(qe - qs, te - ts))
        if seg_len <= 0:
            continue
        for offset in range(seg_len):
            out[int(qs) + offset] = _base_to_residue_idx(template_seq[int(ts) + offset])
    return out


def _query_residue_idx_for_chunk(query_seq: str, chunk_len: int, chunk_length: int) -> np.ndarray:
    out = np.full((chunk_length,), 4, dtype=np.int64)
    if chunk_len <= 0:
        return out
    chunk_seq = query_seq[:chunk_len]
    for i, base in enumerate(chunk_seq):
        out[i] = _base_to_residue_idx(base)
    return out


def _build_oracle_chunk_candidate(
    coords_by_target: dict[str, np.ndarray],
    target_id: str,
    start_idx: int,
    chunk_len: int,
    chunk_length: int,
) -> np.ndarray | None:
    target_coords = coords_by_target.get(target_id)
    if target_coords is None:
        return None
    if target_coords.ndim != 2 or target_coords.shape[1] != 3:
        return None
    if chunk_len <= 0:
        return None

    start_i = max(0, int(start_idx))
    end_i = min(start_i + int(chunk_len), int(target_coords.shape[0]))
    usable_len = max(0, end_i - start_i)
    if usable_len <= 0:
        return None

    out = np.zeros((1, int(chunk_length), 3), dtype=np.float32)
    out[0, :usable_len] = target_coords[start_i:end_i].astype(np.float32, copy=False)
    return out


def _rotmat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    vec = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    vec = vec / norm
    x, y, z = vec
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    cc = 1.0 - c
    return np.asarray(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def _apply_hinge_transform(coords: np.ndarray, rng: np.random.Generator, deg: float = 22.0) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32).copy()
    length = int(out.shape[0])
    if length < 30:
        return out
    pivot = int(rng.integers(10, length - 10))
    angle_rad = np.deg2rad(float(rng.uniform(-deg, deg)))
    rotation = _rotmat(rng.normal(size=3), angle_rad).astype(np.float32, copy=False)
    pivot_point = out[pivot].copy()
    out[pivot + 1 :] = (out[pivot + 1 :] - pivot_point) @ rotation.T + pivot_point
    return out


def _jitter_chain_transform(
    coords: np.ndarray,
    rng: np.random.Generator,
    deg: float = 12.0,
    trans: float = 1.5,
) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32).copy()
    length = int(out.shape[0])
    if length <= 0:
        return out
    global_center = out.mean(axis=0, keepdims=True)
    angle_rad = np.deg2rad(float(rng.uniform(-deg, deg)))
    rotation = _rotmat(rng.normal(size=3), angle_rad).astype(np.float32, copy=False)
    shift = rng.normal(size=3).astype(np.float32, copy=False)
    shift_norm = float(np.linalg.norm(shift))
    if shift_norm > 1e-12:
        shift = shift / shift_norm
    shift *= float(rng.uniform(0.0, trans))
    center = out.mean(axis=0, keepdims=True)
    out = (out - center) @ rotation.T + center + shift
    out -= out.mean(axis=0, keepdims=True) - global_center
    return out


def _smooth_wiggle_transform(coords: np.ndarray, rng: np.random.Generator, amp: float = 0.8) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float32).copy()
    length = int(out.shape[0])
    if length < 20:
        return out
    ctrl = np.linspace(0, length - 1, 6)
    disp = rng.normal(0.0, amp, size=(6, 3)).astype(np.float32, copy=False)
    t = np.arange(length)
    delta = np.vstack([np.interp(t, ctrl, disp[:, axis]) for axis in range(3)]).T.astype(np.float32, copy=False)
    out += delta
    return out


def _adaptive_rna_constraints_chunk(
    coords: np.ndarray,
    confidence: float = 1.0,
    passes: int = 2,
) -> np.ndarray:
    out = np.asarray(coords, dtype=np.float64).copy()
    length = int(out.shape[0])
    if length <= 0:
        return np.asarray(coords, dtype=np.float32)

    strength = max(0.75 * (1.0 - min(float(confidence), 0.97)), 0.02)
    for _ in range(max(1, int(passes))):
        if length >= 2:
            d = out[1:] - out[:-1]
            dist = np.linalg.norm(d, axis=1) + 1e-6
            adjust = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            out[:-1] -= adjust
            out[1:] += adjust
        if length >= 3:
            d2 = out[2:] - out[:-2]
            d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adjust2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            out[:-2] -= adjust2
            out[2:] += adjust2
            out[1:-1] += (0.06 * strength) * (0.5 * (out[:-2] + out[2:]) - out[1:-1])
        if length >= 25:
            if length > 220:
                idx = np.linspace(0, length - 1, min(length, 160)).astype(np.int64)
            else:
                idx = np.arange(length, dtype=np.int64)
            points = out[idx]
            diff = points[:, None, :] - points[None, :, :]
            dm = np.linalg.norm(diff, axis=2) + 1e-6
            sep = np.abs(idx[:, None] - idx[None, :])
            mask = (sep > 2) & (dm < 3.2)
            if np.any(mask):
                repulse = diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]
                out[idx] += (0.015 * strength) * repulse.sum(axis=1)
    return out.astype(np.float32, copy=False)


def _oracle_seed(target_id: str, start_idx: int, window_idx: int) -> int:
    raw = f"{target_id}:{int(start_idx)}:{int(window_idx)}".encode("utf-8")
    return int(zlib.adler32(raw) & 0xFFFFFFFF)


def _build_oracle_diverse_candidate(
    base_coords: np.ndarray,
    chunk_len: int,
    variant_rank: int,
    seed_base: int,
) -> np.ndarray:
    out = np.zeros_like(base_coords, dtype=np.float32)
    usable_len = min(int(chunk_len), int(base_coords.shape[0]), int(out.shape[0]))
    if usable_len <= 0:
        return out

    work = np.asarray(base_coords[:usable_len], dtype=np.float32).copy()
    rank_i = max(0, int(variant_rank))
    rng = np.random.default_rng(np.uint64(seed_base + 1009 * rank_i))

    if rank_i == 0:
        transformed = work
    elif rank_i == 1:
        transformed = work + rng.normal(0.0, 0.01, size=work.shape).astype(np.float32, copy=False)
    elif rank_i == 2:
        transformed = _apply_hinge_transform(work, rng)
    elif rank_i == 3:
        transformed = _jitter_chain_transform(work, rng)
    else:
        transformed = _smooth_wiggle_transform(work, rng)

    transformed = _adaptive_rna_constraints_chunk(transformed, confidence=1.0, passes=2)
    out[:usable_len] = transformed[:usable_len]
    return out


def _load_protenix_chunk_candidates(
    protenix_zip_path: Path,
) -> dict[str, dict[int, np.ndarray]]:
    if not protenix_zip_path.exists():
        log.warning("Protenix fallback path not found at %s. Proceeding without fallback.", protenix_zip_path)
        return {}

    candidates_by_target: dict[str, dict[int, np.ndarray]] = {}
    
    def _consume_payload(raw: bytes) -> None:
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
                if "coords" not in npz or "target_id" not in npz or "chunk_start" not in npz:
                    return
                coords = np.asarray(npz["coords"], dtype=np.float32)
                if coords.ndim == 2 and coords.shape[-1] == 3:
                    coords = coords[None, ...]
                if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] <= 0 or coords.shape[1] <= 0:
                    return
                coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                target_id_raw = np.asarray(npz["target_id"]).reshape(-1)
                chunk_start_raw = np.asarray(npz["chunk_start"]).reshape(-1)
                if target_id_raw.size <= 0 or chunk_start_raw.size <= 0:
                    return
                target_id = str(target_id_raw[0])
                chunk_start = int(chunk_start_raw[0])
                per_target = candidates_by_target.setdefault(target_id, {})
                existing = per_target.get(chunk_start)
                if existing is None:
                    per_target[chunk_start] = coords
                    return
                if existing.ndim != 3 or existing.shape[-1] != 3:
                    per_target[chunk_start] = coords
                    return
                if existing.shape[1] != coords.shape[1]:
                    usable_len = min(int(existing.shape[1]), int(coords.shape[1]))
                    if usable_len <= 0:
                        per_target[chunk_start] = coords
                        return
                    existing = existing[:, :usable_len, :]
                    coords = coords[:, :usable_len, :]
                per_target[chunk_start] = np.concatenate([existing, coords], axis=0).astype(np.float32, copy=False)
        except Exception:
            return

    if protenix_zip_path.is_dir():
        try:
            for npz_path in protenix_zip_path.rglob("*.npz"):
                try:
                    _consume_payload(npz_path.read_bytes())
                except Exception:
                    continue
        except Exception as exc:
            log.warning("Failed to read Protenix fallback directory %s: %s", protenix_zip_path, exc)
            return {}
    else:
        try:
            with zipfile.ZipFile(protenix_zip_path, "r") as archive:
                for member in archive.namelist():
                    if not member.endswith(".npz"):
                        continue
                    try:
                        with archive.open(member) as handle:
                            _consume_payload(handle.read())
                    except Exception:
                        continue
        except Exception as exc:
            log.warning("Failed to load Protenix fallback archive %s: %s", protenix_zip_path, exc)
            return {}

    log.info(
        "Loaded Protenix fallback chunks: %d targets from %s.",
        len(candidates_by_target),
        protenix_zip_path,
    )
    return candidates_by_target


def _lookup_protenix_candidates_for_chunk(
    protenix_candidates_by_target: dict[str, dict[int, np.ndarray]],
    target_id: str,
    start_idx: int,
    chunk_stride: int,
) -> np.ndarray | None:
    per_target = protenix_candidates_by_target.get(target_id)
    if not per_target:
        return None
    if start_idx in per_target:
        return per_target[start_idx]
    nearest_start = min(per_target.keys(), key=lambda s: abs(int(s) - int(start_idx)))
    if abs(int(nearest_start) - int(start_idx)) > max(1, int(chunk_stride)):
        return None
    return per_target[nearest_start]


_WORKER_ALIGNER_LOCAL: PairwiseAligner | None = None
_WORKER_SEQUENCES: dict[str, str] | None = None
_WORKER_COORDS_BY_TARGET: dict[str, np.ndarray] | None = None
_WORKER_VALID_TARGET_IDS: list[str] | None = None
_WORKER_TOP_K_STORE = 5
_WORKER_EXCLUDE_SELF = True
_WORKER_ENFORCE_MIN_TOPK = True
_WORKER_CHUNK_LENGTH = 512
_WORKER_CHUNK_STRIDE = 256
_WORKER_CHUNK_MAX_WINDOWS = 64
_WORKER_MIN_PERCENT_IDENTITY = 50.0
_WORKER_MIN_SIMILARITY = 0.0


def _init_precompute_worker(
    sequences: dict[str, str],
    coords_by_target: dict[str, np.ndarray],
    valid_target_ids: list[str],
    top_k_store: int,
    exclude_self: bool,
    enforce_min_topk: bool,
    chunk_length: int,
    chunk_stride: int,
    chunk_max_windows: int,
    min_percent_identity: float,
    min_similarity: float,
) -> None:
    global _WORKER_ALIGNER_LOCAL
    global _WORKER_SEQUENCES
    global _WORKER_COORDS_BY_TARGET
    global _WORKER_VALID_TARGET_IDS
    global _WORKER_TOP_K_STORE
    global _WORKER_EXCLUDE_SELF
    global _WORKER_ENFORCE_MIN_TOPK
    global _WORKER_CHUNK_LENGTH
    global _WORKER_CHUNK_STRIDE
    global _WORKER_CHUNK_MAX_WINDOWS
    global _WORKER_MIN_PERCENT_IDENTITY
    global _WORKER_MIN_SIMILARITY

    _WORKER_ALIGNER_LOCAL = _make_aligner(mode="global")
    _WORKER_SEQUENCES = sequences
    _WORKER_COORDS_BY_TARGET = coords_by_target
    _WORKER_VALID_TARGET_IDS = valid_target_ids
    _WORKER_TOP_K_STORE = max(1, int(top_k_store))
    _WORKER_EXCLUDE_SELF = bool(exclude_self)
    _WORKER_ENFORCE_MIN_TOPK = bool(enforce_min_topk)
    _WORKER_CHUNK_LENGTH = max(1, int(chunk_length))
    _WORKER_CHUNK_STRIDE = max(1, int(chunk_stride))
    _WORKER_CHUNK_MAX_WINDOWS = max(1, int(chunk_max_windows))
    _WORKER_MIN_PERCENT_IDENTITY = float(min_percent_identity)
    _WORKER_MIN_SIMILARITY = float(min_similarity)


def _first_alignment(aligner: PairwiseAligner, query_seq: str, template_seq: str) -> Any | None:
    try:
        alns = aligner.align(query_seq, template_seq)
    except Exception:
        return None
    try:
        if len(alns) <= 0:
            return None
    except Exception:
        pass
    try:
        return alns[0]
    except Exception:
        return None


def _candidate_template_ids(
    query_tid: str,
    valid_target_ids: list[str],
    exclude_self: bool,
) -> list[str]:
    # Full exhaustive search across all eligible targets.
    return [tid for tid in valid_target_ids if (not exclude_self or tid != query_tid)]


def _select_topk_chunk_templates(
    query_tid: str,
    query_chunk_seq: str,
    sequences: dict[str, str],
    valid_target_ids: list[str],
    aligner_local: PairwiseAligner,
    exclude_self: bool,
    top_k_store: int,
    min_percent_identity: float,
    min_similarity: float,
) -> tuple[list[tuple[str, float, float, Any]], int, int]:
    candidates: list[tuple[str, float, float, Any]] = []
    query_len = len(query_chunk_seq)
    if query_len <= 0:
        return [], 0, 0
    _ = float(min_similarity)  # Similarity is used for ranking only; identity is the hard filter.

    candidate_target_ids = _candidate_template_ids(
        query_tid=query_tid,
        valid_target_ids=valid_target_ids,
        exclude_self=exclude_self,
    )
    aligned_attempts = 0

    for template_tid in candidate_target_ids:
        template_seq = sequences[template_tid]
        if not template_seq:
            continue
        aligned_attempts += 1
        aln = _first_alignment(aligner_local, query_chunk_seq, template_seq)
        if aln is None:
            continue
        norm_similarity = float(aln.score) / float(2.0 * max(1, query_len))
        pct_identity = _compute_identity_percent(query_chunk_seq, template_seq, aln)
        if pct_identity + 1e-8 < float(min_percent_identity):
            continue
        candidates.append((template_tid, norm_similarity, pct_identity, aln))

    dedup: dict[str, tuple[str, float, float, Any]] = {}
    for item in sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True):
        if item[0] not in dedup:
            dedup[item[0]] = item
    return list(dedup.values())[: int(top_k_store)], int(len(candidate_target_ids)), int(aligned_attempts)


def _compute_single_chunk_worker(
    query_tid: str,
    window_idx: int,
    start_idx: int,
    end_idx: int,
) -> tuple[
    str,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    int,
    int,
    str,
]:
    if _WORKER_ALIGNER_LOCAL is None:
        raise RuntimeError("Precompute worker was not initialized.")
    if _WORKER_SEQUENCES is None or _WORKER_COORDS_BY_TARGET is None or _WORKER_VALID_TARGET_IDS is None:
        raise RuntimeError("Precompute worker data was not initialized.")

    aligner_local = _WORKER_ALIGNER_LOCAL
    sequences = _WORKER_SEQUENCES
    coords_by_target = _WORKER_COORDS_BY_TARGET
    valid_target_ids = _WORKER_VALID_TARGET_IDS

    query_seq = sequences[query_tid]
    chunk_len = max(0, min(int(end_idx), len(query_seq)) - int(start_idx))
    if chunk_len <= 0:
        raise RuntimeError(f"Invalid chunk bounds for target='{query_tid}', window={window_idx}.")
    query_chunk_seq = query_seq[int(start_idx) : int(start_idx) + chunk_len]

    used_self_fallback = 0
    used_repeated_fill = 0

    selected_chunk, shortlist_count, aligned_attempts = _select_topk_chunk_templates(
        query_tid=query_tid,
        query_chunk_seq=query_chunk_seq,
        sequences=sequences,
        valid_target_ids=valid_target_ids,
        aligner_local=aligner_local,
        exclude_self=bool(_WORKER_EXCLUDE_SELF),
        top_k_store=int(_WORKER_TOP_K_STORE),
        min_percent_identity=float(_WORKER_MIN_PERCENT_IDENTITY),
        min_similarity=float(_WORKER_MIN_SIMILARITY),
    )

    chunk_topk_coords = np.zeros((_WORKER_TOP_K_STORE, _WORKER_CHUNK_LENGTH, 3), dtype=np.float32)
    chunk_topk_valid = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.bool_)
    chunk_topk_identity = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32)
    chunk_topk_similarity = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32)
    chunk_topk_residue_idx = np.full((_WORKER_TOP_K_STORE, _WORKER_CHUNK_LENGTH), 4, dtype=np.int64)
    chunk_topk_sources: list[str] = [""] * _WORKER_TOP_K_STORE

    for k_idx, (template_tid, sim, pct_identity, aln) in enumerate(selected_chunk[: _WORKER_TOP_K_STORE]):
        adapted_chunk = _adapt_template_to_query(
            query_seq=query_chunk_seq,
            template_seq=sequences[template_tid],
            template_coords=coords_by_target[template_tid],
            alignment=aln,
        )
        adapted_residue_idx = _adapt_template_residue_idx_to_query(
            query_seq=query_chunk_seq,
            template_seq=sequences[template_tid],
            alignment=aln,
        )
        chunk_topk_coords[k_idx, :chunk_len] = adapted_chunk[:chunk_len]
        chunk_topk_valid[k_idx] = True
        chunk_topk_identity[k_idx] = float(pct_identity)
        chunk_topk_similarity[k_idx] = float(sim)
        chunk_topk_residue_idx[k_idx, :chunk_len] = adapted_residue_idx[:chunk_len]
        chunk_topk_sources[k_idx] = template_tid

    return (
        query_tid,
        int(window_idx),
        int(start_idx),
        int(chunk_len),
        chunk_topk_coords,
        chunk_topk_valid,
        chunk_topk_identity,
        chunk_topk_similarity,
        chunk_topk_residue_idx,
        chunk_topk_sources,
        used_self_fallback,
        used_repeated_fill,
        int(shortlist_count),
        int(aligned_attempts),
        f"pid:{os.getpid()}",
    )


def precompute_template_coords(
    labels_path: str | Path,
    output_path: str | Path,
    sequences_path: str | Path | None = None,
    top_k_store: int = 5,
    max_residues_per_target: int = 5120,
    max_targets: int | None = None,
    query_max_targets: int | None = None,
    search_pool_max_targets: int | None = None,
    exclude_self: bool = True,
    enforce_min_topk: bool = True,
    chunk_length: int = 512,
    chunk_stride: int = 256,
    chunk_max_windows: int = 64,
    min_percent_identity: float = 50.0,
    min_similarity: float = 0.0,
    protenix_fallback_zip: str | Path | None = None,
    protenix_base_confidence: float = 0.85,
    num_threads: int = 0,
) -> dict[str, Any]:
    labels = Path(labels_path)
    sequences_path_resolved = resolve_sequences_path(labels_path=labels, sequences_path=sequences_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    query_max_targets_i = query_max_targets if query_max_targets is not None else max_targets
    search_pool_max_targets_i = search_pool_max_targets if search_pool_max_targets is not None else max_targets

    sequences, coords_by_target = _load_sequences_and_coords(
        labels_path=labels,
        max_residues_per_target=max_residues_per_target,
        max_targets=search_pool_max_targets_i,
        sequences_path=sequences_path_resolved,
    )
    valid_target_ids = [tid for tid, c in coords_by_target.items() if _coords_are_valid(c)]
    query_target_ids = (
        valid_target_ids
        if query_max_targets_i is None
        else valid_target_ids[: max(0, int(query_max_targets_i))]
    )

    top_k_store_i = max(1, int(top_k_store))
    chunk_length_i = max(1, int(chunk_length))
    chunk_stride_i = max(1, int(chunk_stride))
    chunk_max_windows_i = max(1, int(chunk_max_windows))
    min_percent_identity_i = float(min_percent_identity)
    min_similarity_i = float(min_similarity)
    protenix_base_confidence_i = float(protenix_base_confidence)
    protenix_zip_i = Path(protenix_fallback_zip) if protenix_fallback_zip is not None else None
    protenix_candidates_by_target = (
        _load_protenix_chunk_candidates(protenix_zip_i) if protenix_zip_i is not None else {}
    )

    templates: dict[str, torch.Tensor] = {}
    available: dict[str, bool] = {}
    chunk_topk_templates: dict[str, torch.Tensor] = {}
    chunk_mask: dict[str, torch.Tensor] = {}
    chunk_start: dict[str, torch.Tensor] = {}
    chunk_window_valid: dict[str, torch.Tensor] = {}
    chunk_topk_valid: dict[str, torch.Tensor] = {}
    chunk_topk_identity: dict[str, torch.Tensor] = {}
    chunk_topk_similarity: dict[str, torch.Tensor] = {}
    chunk_topk_confidence: dict[str, torch.Tensor] = {}
    chunk_topk_residue_idx: dict[str, torch.Tensor] = {}
    chunk_topk_source_type: dict[str, torch.Tensor] = {}
    chunk_topk_source_onehot: dict[str, torch.Tensor] = {}
    chunk_topk_sources: dict[str, list[list[str]]] = {}
    protenix_chunks_used: dict[str, dict[str, torch.Tensor]] = {}

    chunk_coords_np: dict[str, np.ndarray] = {}
    chunk_mask_np: dict[str, np.ndarray] = {}
    chunk_start_np: dict[str, np.ndarray] = {}
    chunk_window_valid_np: dict[str, np.ndarray] = {}
    chunk_valid_np: dict[str, np.ndarray] = {}
    chunk_identity_np: dict[str, np.ndarray] = {}
    chunk_similarity_np: dict[str, np.ndarray] = {}
    chunk_confidence_np: dict[str, np.ndarray] = {}
    chunk_residue_idx_np: dict[str, np.ndarray] = {}
    chunk_source_type_np: dict[str, np.ndarray] = {}
    chunk_sources_np: dict[str, list[list[str]]] = {}

    target_used_self_fallback: dict[str, bool] = {}
    target_used_repeated_fill: dict[str, bool] = {}
    num_chunk_windows_with_self_fallback = 0
    num_chunk_windows_with_repeated_fill = 0
    total_candidate_targets = 0
    total_alignment_attempts = 0
    num_chunks_with_protenix_request = 0
    num_chunks_with_protenix_library_hit = 0
    num_chunks_with_protenix_library_miss = 0
    num_chunks_with_oracle_fallback = 0

    total_search_pool_targets = len(valid_target_ids)
    total_query_targets = len(query_target_ids)
    if total_search_pool_targets == 0:
        raise RuntimeError("No valid targets found in labels file for template precompute.")
    if total_query_targets == 0:
        raise RuntimeError(
            "No query targets selected for template precompute. "
            f"query_max_targets={query_max_targets_i}, search_pool_targets={total_search_pool_targets}."
        )
    if abs(min_similarity_i) > 1e-8:
        log.info(
            "template_min_similarity=%.3f was provided, but similarity thresholding is disabled; "
            "similarity is used only for ranking identity-qualified templates.",
            min_similarity_i,
        )

    chunk_tasks: list[tuple[str, int, int, int]] = []
    total_windows_generated = 0
    min_windows_per_target: int | None = None
    max_windows_per_target = 0
    for query_tid in query_target_ids:
        q_len = len(sequences[query_tid])
        starts = _compute_chunk_starts(q_len, chunk_length_i, chunk_stride_i, chunk_max_windows_i)
        num_windows = int(len(starts))
        total_windows_generated += num_windows
        if min_windows_per_target is None or num_windows < min_windows_per_target:
            min_windows_per_target = num_windows
        max_windows_per_target = max(max_windows_per_target, num_windows)

        chunk_coords_np[query_tid] = np.zeros(
            (chunk_max_windows_i, top_k_store_i, chunk_length_i, 3),
            dtype=np.float32,
        )
        chunk_mask_np[query_tid] = np.zeros((chunk_max_windows_i, chunk_length_i), dtype=np.bool_)
        chunk_start_np[query_tid] = np.zeros((chunk_max_windows_i,), dtype=np.int64)
        chunk_window_valid_np[query_tid] = np.zeros((chunk_max_windows_i,), dtype=np.bool_)
        chunk_valid_np[query_tid] = np.zeros((chunk_max_windows_i, top_k_store_i), dtype=np.bool_)
        chunk_identity_np[query_tid] = np.zeros((chunk_max_windows_i, top_k_store_i), dtype=np.float32)
        chunk_similarity_np[query_tid] = np.zeros((chunk_max_windows_i, top_k_store_i), dtype=np.float32)
        chunk_confidence_np[query_tid] = np.zeros((chunk_max_windows_i, top_k_store_i), dtype=np.float32)
        chunk_residue_idx_np[query_tid] = np.full((chunk_max_windows_i, top_k_store_i, chunk_length_i), 4, dtype=np.int64)
        chunk_source_type_np[query_tid] = np.zeros((chunk_max_windows_i, top_k_store_i), dtype=np.int64)
        chunk_sources_np[query_tid] = [[""] * top_k_store_i for _ in range(chunk_max_windows_i)]
        target_used_self_fallback[query_tid] = False
        target_used_repeated_fill[query_tid] = False

        for window_idx, start_idx in enumerate(starts):
            end_idx = min(start_idx + chunk_length_i, q_len)
            if end_idx <= start_idx:
                continue
            chunk_tasks.append((query_tid, int(window_idx), int(start_idx), int(end_idx)))

    total_chunk_tasks = len(chunk_tasks)
    if total_chunk_tasks == 0:
        raise RuntimeError("No valid chunk windows were generated for template precompute.")

    requested_threads = int(num_threads)
    worker_threads = min(16, max(1, os.cpu_count() or 1)) if requested_threads <= 0 else max(1, requested_threads)
    worker_threads = min(worker_threads, max(1, total_chunk_tasks))
    avg_windows_per_target = float(total_windows_generated) / float(max(1, total_query_targets))
    min_windows_log = int(min_windows_per_target if min_windows_per_target is not None else 0)

    worker_ids_seen: set[str] = set()
    progress_start = time.perf_counter()
    progress_every = max(1, min(500, total_chunk_tasks // 100 if total_chunk_tasks >= 100 else 20))
    progress_time_every = 10.0
    last_logged_done = 0
    last_logged_time = progress_start
    last_logged_alignment_attempts = 0
    last_logged_candidate_targets = 0

    log.info(
        "Template precompute workset: query_targets=%d search_pool_targets=%d chunks=%d "
        "windows/target(min/avg/max)=%d/%.2f/%d chunk_len=%d stride=%d top_k=%d threads=%d.",
        total_query_targets,
        total_search_pool_targets,
        total_chunk_tasks,
        min_windows_log,
        avg_windows_per_target,
        int(max_windows_per_target),
        int(chunk_length_i),
        int(chunk_stride_i),
        int(top_k_store_i),
        int(worker_threads),
    )

    def _log_progress(done: int, force: bool = False) -> None:
        nonlocal last_logged_done, last_logged_time
        nonlocal last_logged_alignment_attempts, last_logged_candidate_targets
        now = time.perf_counter()
        hit_chunk_boundary = (done % progress_every) == 0
        hit_time_boundary = (now - last_logged_time) >= progress_time_every
        if (not force) and (not hit_chunk_boundary) and (not hit_time_boundary):
            return
        if force and done == last_logged_done:
            return
        elapsed = now - progress_start
        interval_elapsed = max(1e-9, now - last_logged_time)
        rate = done / max(elapsed, 1e-9)
        interval_done = max(0, done - last_logged_done)
        interval_rate = float(interval_done) / interval_elapsed
        remaining = max(0, total_chunk_tasks - done)
        eta = remaining / max(rate, 1e-9)
        pct = 100.0 * float(done) / float(max(1, total_chunk_tasks))
        avg_candidates = float(total_candidate_targets) / float(max(1, done))
        avg_alignments = float(total_alignment_attempts) / float(max(1, done))
        align_rate_overall = float(total_alignment_attempts) / max(elapsed, 1e-9)
        align_rate_interval = float(max(0, total_alignment_attempts - last_logged_alignment_attempts)) / interval_elapsed
        candidate_rate_interval = float(max(0, total_candidate_targets - last_logged_candidate_targets)) / interval_elapsed
        log.info(
            "Template precompute progress: %d/%d chunks (%.1f%%) elapsed=%s eta=%s "
            "chunk_rate=%.2f/s recent=%.2f/s align_rate=%.1f/s recent=%.1f/s "
            "avg_align=%.1f/chunk avg_candidates=%.1f/chunk cand_rate_recent=%.1f/s "
            "workers=%d/%d fallback(req=%d lib_hit=%d lib_miss=%d oracle=%d) fills(repeated=%d)",
            done,
            total_chunk_tasks,
            pct,
            _format_hms(elapsed),
            _format_hms(eta),
            rate,
            interval_rate,
            align_rate_overall,
            align_rate_interval,
            avg_alignments,
            avg_candidates,
            candidate_rate_interval,
            len(worker_ids_seen),
            worker_threads,
            num_chunks_with_protenix_request,
            num_chunks_with_protenix_library_hit,
            num_chunks_with_protenix_library_miss,
            num_chunks_with_oracle_fallback,
            num_chunk_windows_with_repeated_fill,
        )
        last_logged_done = done
        last_logged_time = now
        last_logged_alignment_attempts = total_alignment_attempts
        last_logged_candidate_targets = total_candidate_targets

    def _store_chunk_result(
        result: tuple[
            str,
            int,
            int,
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            list[str],
            int,
            int,
            int,
            int,
            str,
        ]
    ) -> None:
        nonlocal num_chunk_windows_with_self_fallback, num_chunk_windows_with_repeated_fill
        nonlocal total_candidate_targets, total_alignment_attempts
        nonlocal num_chunks_with_protenix_request, num_chunks_with_protenix_library_hit, num_chunks_with_protenix_library_miss
        nonlocal num_chunks_with_oracle_fallback
        (
            query_tid,
            window_idx,
            start_idx,
            chunk_len,
            chunk_tensor,
            chunk_valid_tensor,
            chunk_id_tensor,
            chunk_sim_tensor,
            chunk_residue_idx_tensor,
            chunk_source_ids,
            self_fallback_flag,
            repeated_fill_flag,
            shortlisted_count,
            aligned_attempts,
            worker_name,
        ) = result
        worker_ids_seen.add(worker_name)
        chunk_tensor = chunk_tensor.astype(np.float32, copy=False)
        chunk_valid_tensor = chunk_valid_tensor.astype(np.bool_, copy=False)
        chunk_id_tensor = chunk_id_tensor.astype(np.float32, copy=False)
        chunk_sim_tensor = chunk_sim_tensor.astype(np.float32, copy=False)
        chunk_residue_idx_tensor = chunk_residue_idx_tensor.astype(np.int64, copy=False)

        chunk_source_ids = list(chunk_source_ids[:top_k_store_i])
        if len(chunk_source_ids) < top_k_store_i:
            chunk_source_ids.extend([""] * (top_k_store_i - len(chunk_source_ids)))

        source_type = np.zeros((top_k_store_i,), dtype=np.int64)
        confidence = np.zeros((top_k_store_i,), dtype=np.float32)
        for k_idx in range(top_k_store_i):
            if bool(chunk_valid_tensor[k_idx]) and str(chunk_source_ids[k_idx]):
                source_type[k_idx] = SOURCE_TEMPLATE

        missing_idx = np.where(~chunk_valid_tensor)[0].tolist()
        requested_protenix = bool(missing_idx)
        protenix_candidates_used: np.ndarray | None = None
        library_hit_for_chunk = False
        oracle_used_for_chunk = False
        if missing_idx:
            query_seq = sequences[query_tid]
            query_chunk_seq = query_seq[int(start_idx) : int(start_idx) + int(chunk_len)]
            query_chunk_residue_idx = _query_residue_idx_for_chunk(
                query_seq=query_chunk_seq,
                chunk_len=int(chunk_len),
                chunk_length=chunk_length_i,
            )
            protenix_candidates = _lookup_protenix_candidates_for_chunk(
                protenix_candidates_by_target=protenix_candidates_by_target,
                target_id=query_tid,
                start_idx=int(start_idx),
                chunk_stride=chunk_stride_i,
            )
            if protenix_candidates is not None and protenix_candidates.size > 0:
                library_hit_for_chunk = True

            if protenix_candidates is not None and protenix_candidates.size > 0:
                protenix_candidates_used = np.asarray(protenix_candidates, dtype=np.float32)
                num_samples = int(protenix_candidates.shape[0])
                max_fill = min(len(missing_idx), num_samples)
                for fill_rank, k_idx in enumerate(missing_idx[:max_fill]):
                    sample_idx = int(fill_rank)
                    sample_coords = np.asarray(protenix_candidates[sample_idx], dtype=np.float32)
                    usable_len = min(int(chunk_len), int(sample_coords.shape[0]), int(chunk_length_i))
                    if usable_len > 0:
                        chunk_tensor[k_idx, :usable_len] = sample_coords[:usable_len]
                        if usable_len < int(chunk_length_i):
                            chunk_tensor[k_idx, usable_len:] = 0.0
                        chunk_residue_idx_tensor[k_idx] = 4
                        chunk_residue_idx_tensor[k_idx, : int(chunk_len)] = query_chunk_residue_idx[: int(chunk_len)]
                    chunk_valid_tensor[k_idx] = True
                    chunk_id_tensor[k_idx] = 0.0
                    chunk_sim_tensor[k_idx] = 0.0
                    source_type[k_idx] = SOURCE_PROTENIX
                    confidence[k_idx] = max(0.05, min(1.0, protenix_base_confidence_i - 0.1 * float(sample_idx)))
                    chunk_source_ids[k_idx] = f"protenix:{query_tid}:{start_idx}:{sample_idx}"
                missing_idx = np.where(~chunk_valid_tensor)[0].tolist()
            if missing_idx:
                oracle_candidates = _build_oracle_chunk_candidate(
                    coords_by_target=coords_by_target,
                    target_id=query_tid,
                    start_idx=int(start_idx),
                    chunk_len=int(chunk_len),
                    chunk_length=int(chunk_length_i),
                )
                if oracle_candidates is not None and oracle_candidates.size > 0:
                    oracle_used_for_chunk = True
                    base_oracle_coords = np.asarray(oracle_candidates[0], dtype=np.float32)
                    seed_base = _oracle_seed(query_tid, int(start_idx), int(window_idx))
                    for fill_rank, k_idx in enumerate(missing_idx):
                        sample_coords = _build_oracle_diverse_candidate(
                            base_coords=base_oracle_coords,
                            chunk_len=int(chunk_len),
                            variant_rank=int(fill_rank),
                            seed_base=int(seed_base),
                        )
                        usable_len = min(int(chunk_len), int(sample_coords.shape[0]), int(chunk_length_i))
                        if usable_len > 0:
                            chunk_tensor[k_idx, :usable_len] = sample_coords[:usable_len]
                            if usable_len < int(chunk_length_i):
                                chunk_tensor[k_idx, usable_len:] = 0.0
                            chunk_residue_idx_tensor[k_idx] = 4
                            chunk_residue_idx_tensor[k_idx, : int(chunk_len)] = query_chunk_residue_idx[: int(chunk_len)]
                        chunk_valid_tensor[k_idx] = True
                        chunk_id_tensor[k_idx] = 100.0
                        chunk_sim_tensor[k_idx] = 1.0
                        source_type[k_idx] = SOURCE_TEMPLATE
                        confidence[k_idx] = 0.0
                        chunk_source_ids[k_idx] = f"oracle:{query_tid}:{start_idx}:{int(fill_rank)}"
                    missing_idx = np.where(~chunk_valid_tensor)[0].tolist()

        if missing_idx:
            valid_idx = np.where(chunk_valid_tensor)[0]
            if valid_idx.size > 0:
                best_idx = int(valid_idx[np.argmax(chunk_sim_tensor[valid_idx] + 0.01 * chunk_id_tensor[valid_idx])])
                for k_idx in missing_idx:
                    chunk_tensor[k_idx] = chunk_tensor[best_idx]
                    chunk_valid_tensor[k_idx] = True
                    chunk_id_tensor[k_idx] = chunk_id_tensor[best_idx]
                    chunk_sim_tensor[k_idx] = chunk_sim_tensor[best_idx]
                    chunk_residue_idx_tensor[k_idx] = chunk_residue_idx_tensor[best_idx]
                    source_type[k_idx] = source_type[best_idx]
                    confidence[k_idx] = confidence[best_idx]
                    chunk_source_ids[k_idx] = chunk_source_ids[best_idx]
                    target_used_repeated_fill[query_tid] = True
                    num_chunk_windows_with_repeated_fill += 1

        if bool(enforce_min_topk) and int(np.sum(chunk_valid_tensor)) < int(top_k_store_i):
            raise RuntimeError(
                f"Chunk template coverage failed for target='{query_tid}', window={window_idx}, "
                f"valid={int(np.sum(chunk_valid_tensor))}, required={top_k_store_i}"
            )

        chunk_coords_np[query_tid][window_idx] = chunk_tensor
        chunk_valid_np[query_tid][window_idx] = chunk_valid_tensor
        chunk_identity_np[query_tid][window_idx] = chunk_id_tensor
        chunk_similarity_np[query_tid][window_idx] = chunk_sim_tensor
        chunk_confidence_np[query_tid][window_idx] = confidence
        chunk_residue_idx_np[query_tid][window_idx] = chunk_residue_idx_tensor
        chunk_source_type_np[query_tid][window_idx] = source_type
        chunk_sources_np[query_tid][window_idx] = list(chunk_source_ids)
        chunk_start_np[query_tid][window_idx] = int(start_idx)
        chunk_window_valid_np[query_tid][window_idx] = True
        chunk_mask_np[query_tid][window_idx, : int(chunk_len)] = True

        if int(self_fallback_flag) > 0:
            target_used_self_fallback[query_tid] = True
            num_chunk_windows_with_self_fallback += 1
        if int(repeated_fill_flag) > 0:
            target_used_repeated_fill[query_tid] = True
            num_chunk_windows_with_repeated_fill += 1
        total_candidate_targets += int(shortlisted_count)
        total_alignment_attempts += int(aligned_attempts)

        if requested_protenix:
            num_chunks_with_protenix_request += 1
            if library_hit_for_chunk:
                num_chunks_with_protenix_library_hit += 1
            else:
                num_chunks_with_protenix_library_miss += 1
            if oracle_used_for_chunk:
                num_chunks_with_oracle_fallback += 1
            if protenix_candidates_used is not None:
                per_target_used = protenix_chunks_used.setdefault(query_tid, {})
                start_key = str(int(start_idx))
                if start_key not in per_target_used:
                    per_target_used[start_key] = torch.from_numpy(
                        protenix_candidates_used.astype(np.float32, copy=False)
                    )
    log.info(
        "Template precompute started for %d query targets (%d chunks) with %d search-pool targets using %d process(es) in chunk-parallel mode "
        "(alignment=global, full_exhaustive_search=true, identity_gate=%.1f, ranking=similarity_desc, "
        "protenix_cache=%s, fallback=oracle_diversity_transforms).",
        total_query_targets,
        total_chunk_tasks,
        total_search_pool_targets,
        worker_threads,
        min_percent_identity_i,
        str(protenix_zip_i) if protenix_zip_i is not None else "disabled",
    )

    if worker_threads > 1 and total_chunk_tasks > 1:
        with ProcessPoolExecutor(
            max_workers=worker_threads,
            initializer=_init_precompute_worker,
            initargs=(
                sequences,
                coords_by_target,
                valid_target_ids,
                int(top_k_store_i),
                bool(exclude_self),
                bool(enforce_min_topk),
                int(chunk_length_i),
                int(chunk_stride_i),
                int(chunk_max_windows_i),
                float(min_percent_identity_i),
                float(min_similarity_i),
            ),
        ) as executor:
            futures = [executor.submit(_compute_single_chunk_worker, *task) for task in chunk_tasks]
            done = 0
            for future in as_completed(futures):
                _store_chunk_result(future.result())
                done += 1
                _log_progress(done)
            _log_progress(done, force=True)
    else:
        _init_precompute_worker(
            sequences=sequences,
            coords_by_target=coords_by_target,
            valid_target_ids=valid_target_ids,
            top_k_store=int(top_k_store_i),
            exclude_self=bool(exclude_self),
            enforce_min_topk=bool(enforce_min_topk),
            chunk_length=int(chunk_length_i),
            chunk_stride=int(chunk_stride_i),
            chunk_max_windows=int(chunk_max_windows_i),
            min_percent_identity=float(min_percent_identity_i),
            min_similarity=float(min_similarity_i),
        )
        for idx, task in enumerate(chunk_tasks, start=1):
            _store_chunk_result(_compute_single_chunk_worker(*task))
            _log_progress(idx)
        _log_progress(total_chunk_tasks, force=True)

    for query_tid in query_target_ids:
        q_len = len(sequences[query_tid])
        consensus = np.zeros((q_len, 3), dtype=np.float32)
        weight_sum = np.zeros((q_len, 1), dtype=np.float32)

        c_coords = chunk_coords_np[query_tid]
        c_mask = chunk_mask_np[query_tid]
        c_start = chunk_start_np[query_tid]
        c_window_valid = chunk_window_valid_np[query_tid]
        c_valid = chunk_valid_np[query_tid]
        c_sim = chunk_similarity_np[query_tid]
        c_conf = chunk_confidence_np[query_tid]
        c_source_type = chunk_source_type_np[query_tid]

        for w_idx in range(chunk_max_windows_i):
            if not bool(c_window_valid[w_idx]):
                continue
            start_idx = int(c_start[w_idx])
            win_len = int(np.sum(c_mask[w_idx]))
            end_idx = min(q_len, start_idx + win_len)
            if end_idx <= start_idx:
                continue
            seg_len = end_idx - start_idx
            for k_idx in range(top_k_store_i):
                if not bool(c_valid[w_idx, k_idx]):
                    continue
                if int(c_source_type[w_idx, k_idx]) == SOURCE_PROTENIX:
                    weight = max(1e-4, float(c_conf[w_idx, k_idx]))
                else:
                    weight = max(1e-4, float(c_sim[w_idx, k_idx]))
                consensus[start_idx:end_idx] += weight * c_coords[w_idx, k_idx, :seg_len]
                weight_sum[start_idx:end_idx] += weight

        consensus = np.where(weight_sum > 0.0, consensus / np.maximum(weight_sum, 1e-6), consensus)
        is_available = bool(np.any(c_valid))
        source_onehot = np.zeros((chunk_max_windows_i, top_k_store_i, 2), dtype=np.float32)
        source_onehot[..., 0] = (c_source_type == SOURCE_TEMPLATE).astype(np.float32)
        source_onehot[..., 1] = (c_source_type == SOURCE_PROTENIX).astype(np.float32)

        templates[query_tid] = torch.from_numpy(consensus.astype(np.float32, copy=False))
        available[query_tid] = is_available
        chunk_topk_templates[query_tid] = torch.from_numpy(c_coords.astype(np.float32, copy=False))
        chunk_mask[query_tid] = torch.from_numpy(c_mask.astype(np.bool_, copy=False))
        chunk_start[query_tid] = torch.from_numpy(c_start.astype(np.int64, copy=False))
        chunk_window_valid[query_tid] = torch.from_numpy(c_window_valid.astype(np.bool_, copy=False))
        chunk_topk_valid[query_tid] = torch.from_numpy(c_valid.astype(np.bool_, copy=False))
        chunk_topk_identity[query_tid] = torch.from_numpy(chunk_identity_np[query_tid].astype(np.float32, copy=False))
        chunk_topk_similarity[query_tid] = torch.from_numpy(c_sim.astype(np.float32, copy=False))
        chunk_topk_confidence[query_tid] = torch.from_numpy(c_conf.astype(np.float32, copy=False))
        chunk_topk_residue_idx[query_tid] = torch.from_numpy(chunk_residue_idx_np[query_tid].astype(np.int64, copy=False))
        chunk_topk_source_type[query_tid] = torch.from_numpy(c_source_type.astype(np.int64, copy=False))
        chunk_topk_source_onehot[query_tid] = torch.from_numpy(source_onehot.astype(np.float32, copy=False))
        chunk_topk_sources[query_tid] = [list(row) for row in chunk_sources_np[query_tid]]

    used_workers = len(worker_ids_seen)
    log.info(
        "Template precompute worker utilization: used %d/%d process(es).",
        used_workers,
        worker_threads,
    )
    avg_candidates = float(total_candidate_targets) / float(max(1, total_chunk_tasks))
    avg_aligned = float(total_alignment_attempts) / float(max(1, total_chunk_tasks))
    log.info(
        "Template precompute candidate accounting: total_candidates=%d total_alignments=%d "
        "(avg candidates %.2f/chunk, avg alignments %.2f/chunk).",
        total_candidate_targets,
        total_alignment_attempts,
        avg_candidates,
        avg_aligned,
    )
    log.info(
        "Fallback reuse: requests=%d library_hits=%d library_misses=%d oracle_fallbacks=%d.",
        num_chunks_with_protenix_request,
        num_chunks_with_protenix_library_hit,
        num_chunks_with_protenix_library_miss,
        num_chunks_with_oracle_fallback,
    )

    num_targets_with_self_fallback = int(sum(1 for v in target_used_self_fallback.values() if v))
    num_targets_with_repeated_templates = int(sum(1 for v in target_used_repeated_fill.values() if v))

    payload = {
        "templates": templates,
        "available": available,
        "chunk_topk_templates": chunk_topk_templates,
        "chunk_mask": chunk_mask,
        "chunk_start": chunk_start,
        "chunk_window_valid": chunk_window_valid,
        "chunk_topk_valid": chunk_topk_valid,
        "chunk_topk_identity": chunk_topk_identity,
        "chunk_topk_similarity": chunk_topk_similarity,
        "chunk_topk_confidence": chunk_topk_confidence,
        "chunk_topk_residue_idx": chunk_topk_residue_idx,
        "chunk_topk_source_type": chunk_topk_source_type,
        "chunk_topk_source_onehot": chunk_topk_source_onehot,
        "chunk_topk_sources": chunk_topk_sources,
        "protenix_chunks_used": protenix_chunks_used,
        "meta": {
            "labels_path": str(labels.resolve()),
            "sequences_path": None if sequences_path_resolved is None else str(sequences_path_resolved.resolve()),
            "chunk_selection_policy": CHUNK_SELECTION_POLICY,
            "alignment_mode": "global",
            "top_k_store": int(top_k_store_i),
            "max_residues_per_target": int(max_residues_per_target),
            "max_targets": None if query_max_targets_i is None else int(query_max_targets_i),
            "query_max_targets": None if query_max_targets_i is None else int(query_max_targets_i),
            "search_pool_max_targets": None
            if search_pool_max_targets_i is None
            else int(search_pool_max_targets_i),
            "exclude_self": bool(exclude_self),
            "enforce_min_topk": bool(enforce_min_topk),
            "allow_self_fallback": False,
            "min_percent_identity": float(min_percent_identity_i),
            "min_similarity": float(min_similarity_i),
            "similarity_threshold_enforced": False,
            "chunk_length": int(chunk_length_i),
            "chunk_stride": int(chunk_stride_i),
            "chunk_max_windows": int(chunk_max_windows_i),
            "search_strategy": "full_exhaustive_alignment",
            "protenix_fallback_zip": None if protenix_zip_i is None else str(protenix_zip_i),
            "protenix_base_confidence": float(protenix_base_confidence_i),
            "num_targets": int(total_query_targets),
            "num_query_targets": int(total_query_targets),
            "num_search_pool_targets": int(total_search_pool_targets),
            "num_chunk_tasks": int(total_chunk_tasks),
            "num_threads": int(worker_threads),
            "executor_type": "process_chunk_parallel",
            "num_candidate_targets_total": int(total_candidate_targets),
            "num_alignment_attempts_total": int(total_alignment_attempts),
            "num_chunks_with_protenix_request": int(num_chunks_with_protenix_request),
            "num_chunks_with_protenix_library_hit": int(num_chunks_with_protenix_library_hit),
            "num_chunks_with_protenix_library_miss": int(num_chunks_with_protenix_library_miss),
            "num_chunks_with_oracle_fallback": int(num_chunks_with_oracle_fallback),
            "num_targets_with_self_fallback": int(num_targets_with_self_fallback),
            "num_targets_with_repeated_templates": int(num_targets_with_repeated_templates),
            "num_chunk_windows_with_self_fallback": int(num_chunk_windows_with_self_fallback),
            "num_chunk_windows_with_repeated_fill": int(num_chunk_windows_with_repeated_fill),
        },
    }
    torch.save(payload, output)
    log.info(
        "Saved chunk template payload to %s (targets=%d, self_fallback=%d, repeated_fill=%d).",
        output,
        total_query_targets,
        num_targets_with_self_fallback,
        num_targets_with_repeated_templates,
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute chunked templates from train_labels.csv")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--labels-file", type=str, default="train_labels.csv")
    parser.add_argument("--sequences-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="template_coords.pt")
    parser.add_argument("--top-k-store", type=int, default=5)
    parser.add_argument("--max-residues-per-target", type=int, default=5120)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=0,
        help="Legacy cap applied to both query targets and search pool targets when explicit query/search caps are unset.",
    )
    parser.add_argument(
        "--query-max-targets",
        type=int,
        default=0,
        help="Cap number of query targets to chunk/search/store. <=0 means no cap.",
    )
    parser.add_argument(
        "--search-pool-max-targets",
        type=int,
        default=0,
        help="Cap number of targets loaded into the template search pool. <=0 means no cap.",
    )
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--chunk-stride", type=int, default=256)
    parser.add_argument("--chunk-max-windows", type=int, default=64)
    parser.add_argument("--min-percent-identity", type=float, default=50.0)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument(
        "--protenix-fallback-zip",
        type=str,
        default="protenix_finished_chunks_full_4gpu_2775chunks_20260309T215807Z",
        help="Optional archive or directory of precomputed Protenix chunk coordinates for fallback.",
    )
    parser.add_argument(
        "--protenix-base-confidence",
        type=float,
        default=0.85,
        help="Base confidence feature value for the first Protenix fallback candidate.",
    )
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--include-self", action="store_true", help="Allow self-target as template.")
    parser.add_argument("--disable-enforce-min-topk", action="store_true", help="Do not enforce exact top-K coverage.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()

    data_dir = Path(args.data_dir)
    labels_path = data_dir / args.labels_file
    sequences_path = (data_dir / args.sequences_file) if str(args.sequences_file).strip() else None
    output_path = data_dir / args.output_file
    max_targets = None if args.max_targets <= 0 else args.max_targets
    query_max_targets = None if args.query_max_targets <= 0 else args.query_max_targets
    search_pool_max_targets = None if args.search_pool_max_targets <= 0 else args.search_pool_max_targets

    precompute_template_coords(
        labels_path=labels_path,
        sequences_path=sequences_path,
        output_path=output_path,
        top_k_store=args.top_k_store,
        max_residues_per_target=args.max_residues_per_target,
        max_targets=max_targets,
        query_max_targets=query_max_targets,
        search_pool_max_targets=search_pool_max_targets,
        exclude_self=not args.include_self,
        enforce_min_topk=not args.disable_enforce_min_topk,
        chunk_length=args.chunk_length,
        chunk_stride=args.chunk_stride,
        chunk_max_windows=args.chunk_max_windows,
        min_percent_identity=args.min_percent_identity,
        min_similarity=args.min_similarity,
        protenix_fallback_zip=(data_dir / args.protenix_fallback_zip) if args.protenix_fallback_zip else None,
        protenix_base_confidence=args.protenix_base_confidence,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
