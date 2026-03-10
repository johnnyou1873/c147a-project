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


def _make_aligner() -> PairwiseAligner:
    # Mirrors src/kaggle.ipynb scoring for sequence search/adaptation.
    aligner = PairwiseAligner()
    aligner.mode = "global"
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
        rows.sort(key=lambda r: (r[0], r[1]))  # ID suffix order, then file order
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


_WORKER_ALIGNER: PairwiseAligner | None = None
_WORKER_SEQUENCES: dict[str, str] | None = None
_WORKER_COORDS_BY_TARGET: dict[str, np.ndarray] | None = None
_WORKER_EXACT_IDX: dict[str, list[str]] | None = None
_WORKER_VALID_TARGET_IDS: list[str] | None = None
_WORKER_MIN_PERCENT_IDENTITY = 50.0
_WORKER_MIN_SIMILARITY = 0.1
_WORKER_MAX_TEMPLATES = 8
_WORKER_TOP_K_STORE = 5
_WORKER_LENGTH_RATIO_TOL = 0.3
_WORKER_EXCLUDE_SELF = True
_WORKER_ENFORCE_MIN_TOPK = True
_WORKER_ALLOW_SELF_FALLBACK = True


def _init_precompute_worker(
    sequences: dict[str, str],
    coords_by_target: dict[str, np.ndarray],
    exact_idx: dict[str, list[str]],
    valid_target_ids: list[str],
    min_percent_identity: float,
    min_similarity: float,
    max_templates: int,
    top_k_store: int,
    length_ratio_tolerance: float,
    exclude_self: bool,
    enforce_min_topk: bool,
    allow_self_fallback: bool,
) -> None:
    global _WORKER_ALIGNER
    global _WORKER_SEQUENCES
    global _WORKER_COORDS_BY_TARGET
    global _WORKER_EXACT_IDX
    global _WORKER_VALID_TARGET_IDS
    global _WORKER_MIN_PERCENT_IDENTITY
    global _WORKER_MIN_SIMILARITY
    global _WORKER_MAX_TEMPLATES
    global _WORKER_TOP_K_STORE
    global _WORKER_LENGTH_RATIO_TOL
    global _WORKER_EXCLUDE_SELF
    global _WORKER_ENFORCE_MIN_TOPK
    global _WORKER_ALLOW_SELF_FALLBACK

    _WORKER_ALIGNER = _make_aligner()
    _WORKER_SEQUENCES = sequences
    _WORKER_COORDS_BY_TARGET = coords_by_target
    _WORKER_EXACT_IDX = exact_idx
    _WORKER_VALID_TARGET_IDS = valid_target_ids
    _WORKER_MIN_PERCENT_IDENTITY = float(min_percent_identity)
    _WORKER_MIN_SIMILARITY = float(min_similarity)
    _WORKER_MAX_TEMPLATES = int(max_templates)
    _WORKER_TOP_K_STORE = int(top_k_store)
    _WORKER_LENGTH_RATIO_TOL = float(length_ratio_tolerance)
    _WORKER_EXCLUDE_SELF = bool(exclude_self)
    _WORKER_ENFORCE_MIN_TOPK = bool(enforce_min_topk)
    _WORKER_ALLOW_SELF_FALLBACK = bool(allow_self_fallback)


def _compute_single_query_worker(
    query_tid: str,
) -> tuple[
    str,
    np.ndarray,
    bool,
    list[str],
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    int,
    int,
    str,
]:
    if _WORKER_ALIGNER is None or _WORKER_SEQUENCES is None or _WORKER_COORDS_BY_TARGET is None:
        raise RuntimeError("Precompute worker was not initialized.")
    if _WORKER_EXACT_IDX is None or _WORKER_VALID_TARGET_IDS is None:
        raise RuntimeError("Precompute worker indices were not initialized.")

    aligner_local = _WORKER_ALIGNER
    sequences = _WORKER_SEQUENCES
    coords_by_target = _WORKER_COORDS_BY_TARGET
    exact_idx = _WORKER_EXACT_IDX
    valid_target_ids = _WORKER_VALID_TARGET_IDS

    query_seq = sequences[query_tid]
    q_len = len(query_seq)
    candidates: list[tuple[str, float, float, Any]] = []
    used_self_fallback = 0
    used_repeated_fill = 0

    for tid in exact_idx.get(query_seq, []):
        if _WORKER_EXCLUDE_SELF and tid == query_tid:
            continue
        aln = aligner_local.align(query_seq, sequences[tid])[0]
        candidates.append((tid, 1.0, 100.0, aln))

    for template_tid in valid_target_ids:
        if template_tid == query_tid and _WORKER_EXCLUDE_SELF:
            continue
        template_seq = sequences[template_tid]
        if template_seq == query_seq:
            continue

        len_ratio = abs(len(template_seq) - q_len) / float(max(len(template_seq), q_len, 1))
        if len_ratio > _WORKER_LENGTH_RATIO_TOL:
            continue

        aln = aligner_local.align(query_seq, template_seq)[0]
        norm_similarity = float(aln.score) / float(2.0 * min(q_len, len(template_seq), 1))
        pct_identity = _compute_identity_percent(query_seq, template_seq, aln)
        candidates.append((template_tid, norm_similarity, pct_identity, aln))

    dedup: dict[str, tuple[str, float, float, Any]] = {}
    for item in sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True):
        if item[0] not in dedup:
            dedup[item[0]] = item
    selected = list(dedup.values())[: int(_WORKER_TOP_K_STORE)]

    if _WORKER_ENFORCE_MIN_TOPK and len(selected) < _WORKER_TOP_K_STORE:
        fallback_item: tuple[str, float, float, Any] | None = None
        if selected:
            fallback_item = selected[0]
            used_repeated_fill = 1
        elif _WORKER_ALLOW_SELF_FALLBACK:
            self_aln = aligner_local.align(query_seq, query_seq)[0]
            fallback_item = (query_tid, 1.0, 100.0, self_aln)
            used_self_fallback = 1
        if fallback_item is not None:
            while len(selected) < _WORKER_TOP_K_STORE:
                selected.append(fallback_item)

    if not selected:
        return (
            query_tid,
            np.zeros((q_len, 3), dtype=np.float32),
            False,
            [],
            np.zeros((_WORKER_TOP_K_STORE, q_len, 3), dtype=np.float32),
            np.zeros((_WORKER_TOP_K_STORE,), dtype=np.bool_),
            [],
            np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32),
            np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32),
            used_self_fallback,
            used_repeated_fill,
            f"pid:{os.getpid()}",
        )

    adapted_coords = []
    weights = []
    used_ids = []
    used_identity = []
    used_similarity = []
    for template_tid, sim, pct_identity, aln in selected:
        adapted = _adapt_template_to_query(
            query_seq=query_seq,
            template_seq=sequences[template_tid],
            template_coords=coords_by_target[template_tid],
            alignment=aln,
        )
        adapted_coords.append(adapted)
        weights.append(max(0.0, sim))
        used_ids.append(template_tid)
        used_similarity.append(float(sim))
        used_identity.append(float(pct_identity))

    w = np.asarray(weights, dtype=np.float32)
    if np.all(w <= 0):
        w[:] = 1.0
    w = np.exp(w - np.max(w))
    w = w / np.sum(w)
    stacked = np.stack(adapted_coords, axis=0)  # (K, L, 3)
    consensus = np.tensordot(w, stacked, axes=(0, 0)).astype(np.float32)

    topk_coords = np.zeros((_WORKER_TOP_K_STORE, q_len, 3), dtype=np.float32)
    topk_valid = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.bool_)
    topk_id = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32)
    topk_sim = np.zeros((_WORKER_TOP_K_STORE,), dtype=np.float32)
    topk_ids_used: list[str] = []
    for k_idx, (coords_k, tid_k, id_k, sim_k) in enumerate(
        zip(
            adapted_coords[: _WORKER_TOP_K_STORE],
            used_ids[: _WORKER_TOP_K_STORE],
            used_identity[: _WORKER_TOP_K_STORE],
            used_similarity[: _WORKER_TOP_K_STORE],
        )
    ):
        topk_coords[k_idx] = coords_k.astype(np.float32, copy=False)
        topk_valid[k_idx] = True
        topk_id[k_idx] = float(id_k)
        topk_sim[k_idx] = float(sim_k)
        topk_ids_used.append(tid_k)

    if _WORKER_ENFORCE_MIN_TOPK:
        valid_count = int(np.sum(topk_valid))
        if valid_count < _WORKER_TOP_K_STORE:
            raise RuntimeError(
                f"Template coverage guarantee failed for target '{query_tid}': "
                f"valid={valid_count}, required_valid={_WORKER_TOP_K_STORE}"
            )

    return (
        query_tid,
        consensus,
        True,
        used_ids,
        topk_coords,
        topk_valid,
        topk_ids_used,
        topk_id,
        topk_sim,
        used_self_fallback,
        used_repeated_fill,
        f"pid:{os.getpid()}",
    )


def precompute_template_coords(
    labels_path: str | Path,
    output_path: str | Path,
    min_percent_identity: float = 50.0,
    min_similarity: float = 0.1,
    max_templates: int = 8,
    top_k_store: int = 5,
    max_residues_per_target: int = 5120,
    max_targets: int | None = None,
    length_ratio_tolerance: float = 0.3,
    exclude_self: bool = True,
    enforce_min_topk: bool = True,
    allow_self_fallback: bool = True,
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
    exact_idx: dict[str, list[str]] = {}
    for tid in valid_target_ids:
        exact_idx.setdefault(sequences[tid], []).append(tid)

    templates: dict[str, torch.Tensor] = {}
    available: dict[str, bool] = {}
    sources: dict[str, list[str]] = {}
    topk_templates: dict[str, torch.Tensor] = {}
    topk_mask: dict[str, torch.Tensor] = {}
    topk_sources: dict[str, list[str]] = {}
    topk_identity: dict[str, torch.Tensor] = {}
    topk_similarity: dict[str, torch.Tensor] = {}
    num_targets_with_self_fallback = 0
    num_targets_with_repeated_templates = 0

    total_targets = len(valid_target_ids)
    requested_threads = int(num_threads)
    if requested_threads <= 0:
        worker_threads = min(16, max(1, os.cpu_count() or 1))
    else:
        worker_threads = max(1, requested_threads)
    worker_threads = min(worker_threads, max(1, total_targets))
    worker_ids_seen: set[str] = set()
    progress_start = time.perf_counter()
    progress_every = max(1, min(250, total_targets // 100 if total_targets >= 100 else 10))
    last_logged_done = 0

    def _log_progress(done: int, force: bool = False) -> None:
        nonlocal last_logged_done
        if (not force) and (done % progress_every != 0):
            return
        if force and done == last_logged_done:
            return
        elapsed = time.perf_counter() - progress_start
        rate = done / max(elapsed, 1e-9)
        remaining = max(0, total_targets - done)
        eta = remaining / max(rate, 1e-9)
        pct = 100.0 * float(done) / float(max(1, total_targets))
        log.info(
            "Template precompute progress: %d/%d (%.1f%%) elapsed=%s eta=%s rate=%.2f targets/s",
            done,
            total_targets,
            pct,
            _format_hms(elapsed),
            _format_hms(eta),
            rate,
        )
        last_logged_done = done

    def _store_query_result(
        result: tuple[
            str,
            np.ndarray,
            bool,
            list[str],
            np.ndarray,
            np.ndarray,
            list[str],
            np.ndarray,
            np.ndarray,
            int,
            int,
            str,
        ]
    ) -> None:
        nonlocal num_targets_with_self_fallback, num_targets_with_repeated_templates
        (
            query_tid,
            template_tensor,
            is_available,
            source_ids,
            topk_tensor,
            topk_valid_tensor,
            topk_source_ids,
            topk_id_tensor,
            topk_sim_tensor,
            self_fallback_flag,
            repeated_fill_flag,
            worker_name,
        ) = result
        worker_ids_seen.add(worker_name)
        templates[query_tid] = torch.from_numpy(template_tensor.astype(np.float32, copy=False))
        available[query_tid] = bool(is_available)
        sources[query_tid] = source_ids
        topk_templates[query_tid] = torch.from_numpy(topk_tensor.astype(np.float32, copy=False))
        topk_mask[query_tid] = torch.from_numpy(topk_valid_tensor.astype(np.bool_, copy=False))
        topk_sources[query_tid] = topk_source_ids
        topk_identity[query_tid] = torch.from_numpy(topk_id_tensor.astype(np.float32, copy=False))
        topk_similarity[query_tid] = torch.from_numpy(topk_sim_tensor.astype(np.float32, copy=False))
        num_targets_with_self_fallback += int(self_fallback_flag)
        num_targets_with_repeated_templates += int(repeated_fill_flag)

    log.info(
        "Template precompute started for %d targets using %d process(es).",
        total_targets,
        worker_threads,
    )

    if worker_threads > 1 and total_targets > 1:
        with ProcessPoolExecutor(
            max_workers=worker_threads,
            initializer=_init_precompute_worker,
            initargs=(
                sequences,
                coords_by_target,
                exact_idx,
                valid_target_ids,
                float(min_percent_identity),
                float(min_similarity),
                int(max_templates),
                int(top_k_store),
                float(length_ratio_tolerance),
                bool(exclude_self),
                bool(enforce_min_topk),
                bool(allow_self_fallback),
            ),
        ) as executor:
            futures = [executor.submit(_compute_single_query_worker, query_tid) for query_tid in valid_target_ids]
            done = 0
            for future in as_completed(futures):
                _store_query_result(future.result())
                done += 1
                _log_progress(done)
            _log_progress(done, force=True)
    else:
        _init_precompute_worker(
            sequences,
            coords_by_target,
            exact_idx,
            valid_target_ids,
            float(min_percent_identity),
            float(min_similarity),
            int(max_templates),
            int(top_k_store),
            float(length_ratio_tolerance),
            bool(exclude_self),
            bool(enforce_min_topk),
            bool(allow_self_fallback),
        )
        for idx, query_tid in enumerate(valid_target_ids, start=1):
            _store_query_result(_compute_single_query_worker(query_tid))
            _log_progress(idx)
        _log_progress(total_targets, force=True)

    used_workers = len(worker_ids_seen)
    log.info(
        "Template precompute worker utilization: used %d/%d process(es).",
        used_workers,
        worker_threads,
    )
    if used_workers < worker_threads and total_targets >= worker_threads:
        log.warning(
            "Precompute used fewer processes than requested (used=%d, requested=%d). "
            "This can indicate startup overhead or workload imbalance.",
            used_workers,
            worker_threads,
        )

    payload = {
        "templates": templates,
        "available": available,
        "sources": sources,
        "topk_templates": topk_templates,
        "topk_mask": topk_mask,
        "topk_sources": topk_sources,
        "topk_identity": topk_identity,
        "topk_similarity": topk_similarity,
        "meta": {
            "labels_path": str(labels.resolve()),
            "min_percent_identity": float(min_percent_identity),
            "min_similarity": float(min_similarity),
            "selection_policy": "topk_non_self_by_identity_similarity_no_threshold",
            "max_templates": int(max_templates),
            "top_k_store": int(top_k_store),
            "max_residues_per_target": int(max_residues_per_target),
            "max_targets": None if max_targets is None else int(max_targets),
            "length_ratio_tolerance": float(length_ratio_tolerance),
            "exclude_self": bool(exclude_self),
            "enforce_min_topk": bool(enforce_min_topk),
            "allow_self_fallback": bool(allow_self_fallback),
            "num_targets": int(len(valid_target_ids)),
            "num_threads": int(worker_threads),
            "executor_type": "process",
            "num_targets_with_self_fallback": int(num_targets_with_self_fallback),
            "num_targets_with_repeated_templates": int(num_targets_with_repeated_templates),
        },
    }
    torch.save(payload, output)
    log.info(
        "Saved template coordinates to %s (targets=%d, self_fallback=%d, repeated_fill=%d)",
        output,
        len(valid_target_ids),
        num_targets_with_self_fallback,
        num_targets_with_repeated_templates,
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute template coordinates from train_labels.csv")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--labels-file", type=str, default="train_labels.csv")
    parser.add_argument("--output-file", type=str, default="template_coords.pt")
    parser.add_argument("--min-percent-identity", type=float, default=50.0)
    parser.add_argument("--min-similarity", type=float, default=0.1)
    parser.add_argument("--max-templates", type=int, default=8)
    parser.add_argument("--top-k-store", type=int, default=5)
    parser.add_argument("--max-residues-per-target", type=int, default=5120)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--length-ratio-tolerance", type=float, default=0.3)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--include-self", action="store_true", help="Allow self-target as template.")
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
        min_percent_identity=args.min_percent_identity,
        min_similarity=args.min_similarity,
        max_templates=args.max_templates,
        top_k_store=args.top_k_store,
        max_residues_per_target=args.max_residues_per_target,
        max_targets=max_targets,
        length_ratio_tolerance=args.length_ratio_tolerance,
        exclude_self=not args.include_self,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
