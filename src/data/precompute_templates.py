from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.Align import PairwiseAligner


log = logging.getLogger(__name__)

RNA_RESIDUES = {"A", "C", "G", "U", "T"}
RESNAME_TO_BASE = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U"}


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _make_aligner(mode: str = "local") -> PairwiseAligner:
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
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    rows_by_target: dict[str, list[tuple[int, int, str, float, float, float]]] = {}

    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_idx, row in enumerate(reader):
            target_id, pos = _split_target_and_pos(str(row.get("ID", "")))
            if not target_id:
                continue
            if target_id not in rows_by_target:
                if max_targets is not None and len(rows_by_target) >= max_targets:
                    continue
                rows_by_target[target_id] = []
            if len(rows_by_target[target_id]) >= max_residues_per_target:
                continue

            resname = _canonical_resname(str(row.get("resname", "N")))
            try:
                x = float(row.get("x_1", 0.0))
                y = float(row.get("y_1", 0.0))
                z = float(row.get("z_1", 0.0))
            except ValueError:
                x, y, z = 0.0, 0.0, 0.0

            rows_by_target[target_id].append((pos, line_idx, resname, x, y, z))

    sequences: dict[str, str] = {}
    coords_by_target: dict[str, np.ndarray] = {}
    for target_id, rows in rows_by_target.items():
        rows.sort(key=lambda r: (r[0], r[1]))  # ID suffix order, then file order.
        seq = "".join(r[2] for r in rows)
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


_WORKER_ALIGNER_LOCAL: PairwiseAligner | None = None
_WORKER_SEQUENCES: dict[str, str] | None = None
_WORKER_COORDS_BY_TARGET: dict[str, np.ndarray] | None = None
_WORKER_VALID_TARGET_IDS: list[str] | None = None
_WORKER_TOP_K_STORE = 5
_WORKER_EXCLUDE_SELF = True
_WORKER_ENFORCE_MIN_TOPK = True
_WORKER_CHUNK_LENGTH = 512
_WORKER_CHUNK_STRIDE = 256
_WORKER_CHUNK_MAX_WINDOWS = 20


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

    _WORKER_ALIGNER_LOCAL = _make_aligner(mode="local")
    _WORKER_SEQUENCES = sequences
    _WORKER_COORDS_BY_TARGET = coords_by_target
    _WORKER_VALID_TARGET_IDS = valid_target_ids
    _WORKER_TOP_K_STORE = max(1, int(top_k_store))
    _WORKER_EXCLUDE_SELF = bool(exclude_self)
    _WORKER_ENFORCE_MIN_TOPK = bool(enforce_min_topk)
    _WORKER_CHUNK_LENGTH = max(1, int(chunk_length))
    _WORKER_CHUNK_STRIDE = max(1, int(chunk_stride))
    _WORKER_CHUNK_MAX_WINDOWS = max(1, int(chunk_max_windows))


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


def _select_topk_chunk_templates(
    query_tid: str,
    query_chunk_seq: str,
    sequences: dict[str, str],
    valid_target_ids: list[str],
    aligner_local: PairwiseAligner,
    exclude_self: bool,
    top_k_store: int,
) -> list[tuple[str, float, float, Any]]:
    candidates: list[tuple[str, float, float, Any]] = []
    query_len = len(query_chunk_seq)
    if query_len <= 0:
        return []

    for template_tid in valid_target_ids:
        if exclude_self and template_tid == query_tid:
            continue
        template_seq = sequences[template_tid]
        if not template_seq:
            continue
        aln = _first_alignment(aligner_local, query_chunk_seq, template_seq)
        if aln is None:
            continue
        norm_similarity = float(aln.score) / float(2.0 * max(1, query_len))
        pct_identity = _compute_identity_percent(query_chunk_seq, template_seq, aln)
        candidates.append((template_tid, norm_similarity, pct_identity, aln))

    dedup: dict[str, tuple[str, float, float, Any]] = {}
    for item in sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True):
        if item[0] not in dedup:
            dedup[item[0]] = item
    return list(dedup.values())[: int(top_k_store)]


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
    list[str],
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

    selected_chunk = _select_topk_chunk_templates(
        query_tid=query_tid,
        query_chunk_seq=query_chunk_seq,
        sequences=sequences,
        valid_target_ids=valid_target_ids,
        aligner_local=aligner_local,
        exclude_self=bool(_WORKER_EXCLUDE_SELF),
        top_k_store=int(_WORKER_TOP_K_STORE),
    )

    if _WORKER_ENFORCE_MIN_TOPK and len(selected_chunk) < _WORKER_TOP_K_STORE:
        raise RuntimeError(
            f"Chunk template coverage failed for target='{query_tid}', window={window_idx}, "
            f"valid={len(selected_chunk)}, required={_WORKER_TOP_K_STORE}"
        )

    chunk_topk_coords = np.zeros((_WORKER_TOP_K_STORE, _WORKER_CHUNK_LENGTH, 3), dtype=np.float32)
    chunk_topk_valid = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.bool_)
    chunk_topk_identity = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32)
    chunk_topk_similarity = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32)
    chunk_topk_sources: list[str] = [""] * _WORKER_TOP_K_STORE

    for k_idx, (template_tid, sim, pct_identity, aln) in enumerate(selected_chunk[: _WORKER_TOP_K_STORE]):
        adapted_chunk = _adapt_template_to_query(
            query_seq=query_chunk_seq,
            template_seq=sequences[template_tid],
            template_coords=coords_by_target[template_tid],
            alignment=aln,
        )
        chunk_topk_coords[k_idx, :chunk_len] = adapted_chunk[:chunk_len]
        chunk_topk_valid[k_idx] = True
        chunk_topk_identity[k_idx] = float(pct_identity)
        chunk_topk_similarity[k_idx] = float(sim)
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
        chunk_topk_sources,
        used_self_fallback,
        used_repeated_fill,
        f"pid:{os.getpid()}",
    )


def precompute_template_coords(
    labels_path: str | Path,
    output_path: str | Path,
    top_k_store: int = 5,
    max_residues_per_target: int = 5120,
    max_targets: int | None = None,
    exclude_self: bool = True,
    enforce_min_topk: bool = True,
    chunk_length: int = 512,
    chunk_stride: int = 256,
    chunk_max_windows: int = 20,
    num_threads: int = 0,
) -> dict[str, Any]:
    labels = Path(labels_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sequences, coords_by_target = _load_sequences_and_coords(
        labels_path=labels,
        max_residues_per_target=max_residues_per_target,
        max_targets=max_targets,
    )
    valid_target_ids = [tid for tid, c in coords_by_target.items() if _coords_are_valid(c)]

    top_k_store_i = max(1, int(top_k_store))
    chunk_length_i = max(1, int(chunk_length))
    chunk_stride_i = max(1, int(chunk_stride))
    chunk_max_windows_i = max(1, int(chunk_max_windows))

    templates: dict[str, torch.Tensor] = {}
    available: dict[str, bool] = {}
    chunk_topk_templates: dict[str, torch.Tensor] = {}
    chunk_mask: dict[str, torch.Tensor] = {}
    chunk_start: dict[str, torch.Tensor] = {}
    chunk_window_valid: dict[str, torch.Tensor] = {}
    chunk_topk_valid: dict[str, torch.Tensor] = {}
    chunk_topk_identity: dict[str, torch.Tensor] = {}
    chunk_topk_similarity: dict[str, torch.Tensor] = {}
    chunk_topk_sources: dict[str, list[list[str]]] = {}

    chunk_coords_np: dict[str, np.ndarray] = {}
    chunk_mask_np: dict[str, np.ndarray] = {}
    chunk_start_np: dict[str, np.ndarray] = {}
    chunk_window_valid_np: dict[str, np.ndarray] = {}
    chunk_valid_np: dict[str, np.ndarray] = {}
    chunk_identity_np: dict[str, np.ndarray] = {}
    chunk_similarity_np: dict[str, np.ndarray] = {}
    chunk_sources_np: dict[str, list[list[str]]] = {}

    target_used_self_fallback: dict[str, bool] = {}
    target_used_repeated_fill: dict[str, bool] = {}
    num_chunk_windows_with_self_fallback = 0
    num_chunk_windows_with_repeated_fill = 0

    total_targets = len(valid_target_ids)
    if total_targets == 0:
        raise RuntimeError("No valid targets found in labels file for template precompute.")

    chunk_tasks: list[tuple[str, int, int, int]] = []
    for query_tid in valid_target_ids:
        q_len = len(sequences[query_tid])
        starts = _compute_chunk_starts(q_len, chunk_length_i, chunk_stride_i, chunk_max_windows_i)

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

    worker_ids_seen: set[str] = set()
    progress_start = time.perf_counter()
    progress_every = max(1, min(500, total_chunk_tasks // 100 if total_chunk_tasks >= 100 else 20))
    last_logged_done = 0

    def _log_progress(done: int, force: bool = False) -> None:
        nonlocal last_logged_done
        if (not force) and (done % progress_every != 0):
            return
        if force and done == last_logged_done:
            return
        elapsed = time.perf_counter() - progress_start
        rate = done / max(elapsed, 1e-9)
        remaining = max(0, total_chunk_tasks - done)
        eta = remaining / max(rate, 1e-9)
        pct = 100.0 * float(done) / float(max(1, total_chunk_tasks))
        log.info(
            "Template precompute progress: %d/%d chunks (%.1f%%) elapsed=%s eta=%s rate=%.2f chunks/s",
            done,
            total_chunk_tasks,
            pct,
            _format_hms(elapsed),
            _format_hms(eta),
            rate,
        )
        last_logged_done = done

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
            list[str],
            int,
            int,
            str,
        ]
    ) -> None:
        nonlocal num_chunk_windows_with_self_fallback, num_chunk_windows_with_repeated_fill
        (
            query_tid,
            window_idx,
            start_idx,
            chunk_len,
            chunk_tensor,
            chunk_valid_tensor,
            chunk_id_tensor,
            chunk_sim_tensor,
            chunk_source_ids,
            self_fallback_flag,
            repeated_fill_flag,
            worker_name,
        ) = result
        worker_ids_seen.add(worker_name)
        chunk_coords_np[query_tid][window_idx] = chunk_tensor.astype(np.float32, copy=False)
        chunk_valid_np[query_tid][window_idx] = chunk_valid_tensor.astype(np.bool_, copy=False)
        chunk_identity_np[query_tid][window_idx] = chunk_id_tensor.astype(np.float32, copy=False)
        chunk_similarity_np[query_tid][window_idx] = chunk_sim_tensor.astype(np.float32, copy=False)
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

    log.info(
        "Template precompute started for %d targets (%d chunks) using %d process(es) in chunk-parallel mode (alignment=local).",
        total_targets,
        total_chunk_tasks,
        worker_threads,
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
        )
        for idx, task in enumerate(chunk_tasks, start=1):
            _store_chunk_result(_compute_single_chunk_worker(*task))
            _log_progress(idx)
        _log_progress(total_chunk_tasks, force=True)

    for query_tid in valid_target_ids:
        q_len = len(sequences[query_tid])
        consensus = np.zeros((q_len, 3), dtype=np.float32)
        weight_sum = np.zeros((q_len, 1), dtype=np.float32)

        c_coords = chunk_coords_np[query_tid]
        c_mask = chunk_mask_np[query_tid]
        c_start = chunk_start_np[query_tid]
        c_window_valid = chunk_window_valid_np[query_tid]
        c_valid = chunk_valid_np[query_tid]
        c_sim = chunk_similarity_np[query_tid]

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
                weight = max(1e-4, float(c_sim[w_idx, k_idx]))
                consensus[start_idx:end_idx] += weight * c_coords[w_idx, k_idx, :seg_len]
                weight_sum[start_idx:end_idx] += weight

        consensus = np.where(weight_sum > 0.0, consensus / np.maximum(weight_sum, 1e-6), consensus)
        is_available = bool(np.any(c_valid))

        templates[query_tid] = torch.from_numpy(consensus.astype(np.float32, copy=False))
        available[query_tid] = is_available
        chunk_topk_templates[query_tid] = torch.from_numpy(c_coords.astype(np.float32, copy=False))
        chunk_mask[query_tid] = torch.from_numpy(c_mask.astype(np.bool_, copy=False))
        chunk_start[query_tid] = torch.from_numpy(c_start.astype(np.int64, copy=False))
        chunk_window_valid[query_tid] = torch.from_numpy(c_window_valid.astype(np.bool_, copy=False))
        chunk_topk_valid[query_tid] = torch.from_numpy(c_valid.astype(np.bool_, copy=False))
        chunk_topk_identity[query_tid] = torch.from_numpy(chunk_identity_np[query_tid].astype(np.float32, copy=False))
        chunk_topk_similarity[query_tid] = torch.from_numpy(c_sim.astype(np.float32, copy=False))
        chunk_topk_sources[query_tid] = [list(row) for row in chunk_sources_np[query_tid]]

    used_workers = len(worker_ids_seen)
    log.info(
        "Template precompute worker utilization: used %d/%d process(es).",
        used_workers,
        worker_threads,
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
        "chunk_topk_sources": chunk_topk_sources,
        "meta": {
            "labels_path": str(labels.resolve()),
            "chunk_selection_policy": "chunked_topk_non_self_by_identity_similarity_no_threshold",
            "alignment_mode": "local",
            "top_k_store": int(top_k_store_i),
            "max_residues_per_target": int(max_residues_per_target),
            "max_targets": None if max_targets is None else int(max_targets),
            "exclude_self": bool(exclude_self),
            "enforce_min_topk": bool(enforce_min_topk),
            "allow_self_fallback": False,
            "chunk_length": int(chunk_length_i),
            "chunk_stride": int(chunk_stride_i),
            "chunk_max_windows": int(chunk_max_windows_i),
            "num_targets": int(total_targets),
            "num_chunk_tasks": int(total_chunk_tasks),
            "num_threads": int(worker_threads),
            "executor_type": "process_chunk_parallel",
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
        total_targets,
        num_targets_with_self_fallback,
        num_targets_with_repeated_templates,
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute chunked templates from train_labels.csv")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--labels-file", type=str, default="train_labels.csv")
    parser.add_argument("--output-file", type=str, default="template_coords.pt")
    parser.add_argument("--top-k-store", type=int, default=5)
    parser.add_argument("--max-residues-per-target", type=int, default=5120)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--chunk-stride", type=int, default=256)
    parser.add_argument("--chunk-max-windows", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--include-self", action="store_true", help="Allow self-target as template.")
    parser.add_argument("--disable-enforce-min-topk", action="store_true", help="Do not enforce exact top-K coverage.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()

    data_dir = Path(args.data_dir)
    labels_path = data_dir / args.labels_file
    output_path = data_dir / args.output_file
    max_targets = None if args.max_targets <= 0 else args.max_targets

    precompute_template_coords(
        labels_path=labels_path,
        output_path=output_path,
        top_k_store=args.top_k_store,
        max_residues_per_target=args.max_residues_per_target,
        max_targets=max_targets,
        exclude_self=not args.include_self,
        enforce_min_topk=not args.disable_enforce_min_topk,
        chunk_length=args.chunk_length,
        chunk_stride=args.chunk_stride,
        chunk_max_windows=args.chunk_max_windows,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
