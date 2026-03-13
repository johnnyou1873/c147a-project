from __future__ import annotations

import argparse
import logging
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.precompute_templates import (
    _adapt_template_residue_idx_to_query,
    _adapt_template_to_query,
    _build_oracle_diverse_candidate,
    _compute_identity_percent,
    _format_hms,
    _load_sequences_and_coords,
    _make_aligner,
)
from src.data.kaggle_sequence_metadata import resolve_sequences_path


log = logging.getLogger(__name__)

FULL_LENGTH_SELECTION_POLICY = "full_length_topk_non_self_by_similarity_rank_identity_gate_with_oracle_diversity_fallback"

_FullTemplateWorkerResult = tuple[
    str,
    np.ndarray,
    bool,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    int,
]

_FullTemplateStores = tuple[
    dict[str, torch.Tensor],
    dict[str, bool],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, list[str]],
]

_WORKER_ALIGNER = None
_WORKER_SEQUENCES: dict[str, str] = {}
_WORKER_COORDS: dict[str, np.ndarray] = {}
_WORKER_VALID_TARGET_IDS: list[str] = []
_WORKER_TOP_K = 5
_WORKER_EXCLUDE_SELF = True
_WORKER_MIN_PERCENT_IDENTITY = 50.0
_WORKER_MIN_SIMILARITY = 0.0
_WORKER_LENGTH_RATIO_TOLERANCE = 0.3


def _new_full_template_stores() -> _FullTemplateStores:
    return {}, {}, {}, {}, {}, {}, {}, {}


def _store_full_template_result(
    stores: _FullTemplateStores,
    result: _FullTemplateWorkerResult,
) -> tuple[int, int, int]:
    (
        templates,
        available,
        template_topk_coords,
        template_topk_valid,
        template_topk_identity,
        template_topk_similarity,
        template_topk_residue_idx,
        template_topk_sources,
    ) = stores
    (
        query_tid,
        consensus,
        has_template,
        topk_coords_np,
        topk_valid_np,
        topk_identity_np,
        topk_similarity_np,
        topk_residue_idx_np,
        topk_sources_list,
        candidate_count,
        alignment_attempts,
        repeated_fill_used,
    ) = result
    templates[query_tid] = torch.from_numpy(consensus.astype(np.float32, copy=False))
    available[query_tid] = bool(has_template)
    template_topk_coords[query_tid] = torch.from_numpy(topk_coords_np.astype(np.float32, copy=False))
    template_topk_valid[query_tid] = torch.from_numpy(topk_valid_np.astype(np.bool_, copy=False))
    template_topk_identity[query_tid] = torch.from_numpy(topk_identity_np.astype(np.float32, copy=False))
    template_topk_similarity[query_tid] = torch.from_numpy(topk_similarity_np.astype(np.float32, copy=False))
    template_topk_residue_idx[query_tid] = torch.from_numpy(topk_residue_idx_np.astype(np.int64, copy=False))
    template_topk_sources[query_tid] = list(topk_sources_list)
    return int(candidate_count), int(alignment_attempts), int(repeated_fill_used)


def _build_full_template_payload(
    *,
    stores: _FullTemplateStores,
    labels: Path,
    sequences_path_resolved: Path | None,
    top_k_store: int,
    max_residues_per_target: int,
    max_targets: int | None,
    query_cap: int | None,
    search_cap: int | None,
    exclude_self: bool,
    min_percent_identity: float,
    min_similarity: float,
    length_ratio_tolerance: float,
    query_target_ids: list[str],
    search_sequences: dict[str, str],
    total_candidate_targets: int,
    total_alignment_attempts: int,
    total_repeated_fill_targets: int,
    worker_threads: int,
) -> dict[str, Any]:
    (
        templates,
        available,
        template_topk_coords,
        template_topk_valid,
        template_topk_identity,
        template_topk_similarity,
        template_topk_residue_idx,
        template_topk_sources,
    ) = stores
    return {
        "templates": templates,
        "available": available,
        "template_topk_coords": template_topk_coords,
        "template_topk_valid": template_topk_valid,
        "template_topk_identity": template_topk_identity,
        "template_topk_similarity": template_topk_similarity,
        "template_topk_residue_idx": template_topk_residue_idx,
        "template_topk_sources": template_topk_sources,
        "meta": {
            "labels_path": str(labels.resolve()),
            "sequences_path": None if sequences_path_resolved is None else str(sequences_path_resolved.resolve()),
            "selection_policy": FULL_LENGTH_SELECTION_POLICY,
            "alignment_mode": "global",
            "search_strategy": "full_exhaustive_alignment",
            "top_k_store": int(top_k_store),
            "max_residues_per_target": int(max_residues_per_target),
            "max_targets": None if max_targets is None else int(max_targets),
            "query_max_targets": None if query_cap is None else int(query_cap),
            "search_pool_max_targets": None if search_cap is None else int(search_cap),
            "exclude_self": bool(exclude_self),
            "min_percent_identity": float(min_percent_identity),
            "min_similarity": float(min_similarity),
            "length_ratio_tolerance": float(length_ratio_tolerance),
            "num_query_targets": int(len(query_target_ids)),
            "num_search_pool_targets": int(len(search_sequences)),
            "num_candidate_targets_total": int(total_candidate_targets),
            "num_alignment_attempts_total": int(total_alignment_attempts),
            "num_targets_with_repeated_templates": int(total_repeated_fill_targets),
            "num_threads": int(worker_threads),
        },
    }


def _full_length_seed(query_tid: str, slot_idx: int) -> int:
    raw = f"{query_tid}:{int(slot_idx)}".encode("utf-8")
    return int(zlib.adler32(raw) & 0xFFFFFFFF)


def _softmax(values: np.ndarray, temperature: float = 0.08) -> np.ndarray:
    if values.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    scaled = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    scaled = scaled - float(np.max(scaled))
    probs = np.exp(scaled)
    denom = float(np.sum(probs))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full(values.shape, 1.0 / float(max(1, values.size)), dtype=np.float64)
    return probs / denom


def _fill_topk_with_diverse_transforms(
    query_tid: str,
    query_len: int,
    topk_coords: np.ndarray,
    topk_valid: np.ndarray,
    topk_identity: np.ndarray,
    topk_similarity: np.ndarray,
    topk_residue_idx: np.ndarray,
    topk_sources: list[str],
    matches: list[tuple[float, float, str, np.ndarray, np.ndarray]],
) -> bool:
    valid_count = int(np.sum(topk_valid))
    if valid_count <= 0 or valid_count >= int(topk_valid.shape[0]):
        return False

    shortlist = matches[: min(15, len(matches))]
    if not shortlist:
        return False

    used_sources = {str(source) for source in topk_sources if str(source)}
    similarity_scores = np.asarray([float(item[0]) for item in shortlist], dtype=np.float64)
    source_probs = _softmax(similarity_scores, temperature=0.08)

    for slot_idx in range(valid_count, int(topk_valid.shape[0])):
        penalties = np.asarray(
            [0.10 if str(item[2]) in used_sources else 1.0 for item in shortlist],
            dtype=np.float64,
        )
        probs = source_probs * penalties
        prob_sum = float(np.sum(probs))
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            probs = np.full((len(shortlist),), 1.0 / float(max(1, len(shortlist))), dtype=np.float64)
        else:
            probs = probs / prob_sum

        rng = np.random.default_rng(np.uint64(_full_length_seed(query_tid, slot_idx)))
        picked = int(rng.choice(len(shortlist), p=probs))
        similarity, percent_identity, source_tid, adapted_coords, adapted_residue_idx = shortlist[picked]
        diverse_coords = _build_oracle_diverse_candidate(
            base_coords=np.asarray(adapted_coords, dtype=np.float32),
            chunk_len=int(query_len),
            variant_rank=slot_idx,
            seed_base=_full_length_seed(query_tid, slot_idx),
        )
        topk_coords[slot_idx] = diverse_coords.astype(np.float32, copy=False)
        topk_valid[slot_idx] = True
        topk_identity[slot_idx] = float(percent_identity)
        topk_similarity[slot_idx] = float(similarity)
        topk_residue_idx[slot_idx] = np.asarray(adapted_residue_idx, dtype=np.int64)
        topk_sources[slot_idx] = f"oracle:{source_tid}:{slot_idx}"
        used_sources.add(str(source_tid))
    return True


def _init_full_template_worker(
    sequences: dict[str, str],
    coords_by_target: dict[str, np.ndarray],
    valid_target_ids: list[str],
    top_k_store: int,
    exclude_self: bool,
    min_percent_identity: float,
    min_similarity: float,
    length_ratio_tolerance: float,
) -> None:
    global _WORKER_ALIGNER
    global _WORKER_SEQUENCES
    global _WORKER_COORDS
    global _WORKER_VALID_TARGET_IDS
    global _WORKER_TOP_K
    global _WORKER_EXCLUDE_SELF
    global _WORKER_MIN_PERCENT_IDENTITY
    global _WORKER_MIN_SIMILARITY
    global _WORKER_LENGTH_RATIO_TOLERANCE

    _WORKER_ALIGNER = _make_aligner(mode="global")
    _WORKER_SEQUENCES = sequences
    _WORKER_COORDS = coords_by_target
    _WORKER_VALID_TARGET_IDS = valid_target_ids
    _WORKER_TOP_K = max(1, int(top_k_store))
    _WORKER_EXCLUDE_SELF = bool(exclude_self)
    _WORKER_MIN_PERCENT_IDENTITY = float(min_percent_identity)
    _WORKER_MIN_SIMILARITY = float(min_similarity)
    _WORKER_LENGTH_RATIO_TOLERANCE = float(length_ratio_tolerance)


def _compute_single_full_template_worker(query_tid: str) -> _FullTemplateWorkerResult:
    if _WORKER_ALIGNER is None:
        raise RuntimeError("Full-template worker not initialized.")

    query_seq = _WORKER_SEQUENCES[query_tid]
    query_len = len(query_seq)
    query_coords = _WORKER_COORDS[query_tid]
    topk_coords = np.zeros((_WORKER_TOP_K, query_len, 3), dtype=np.float32)
    topk_valid = np.zeros((_WORKER_TOP_K,), dtype=np.bool_)
    topk_identity = np.zeros((_WORKER_TOP_K,), dtype=np.float32)
    topk_similarity = np.zeros((_WORKER_TOP_K,), dtype=np.float32)
    topk_residue_idx = np.full((_WORKER_TOP_K, query_len), 4, dtype=np.int64)
    topk_sources = [""] * _WORKER_TOP_K

    candidate_count = 0
    alignment_attempts = 0
    repeated_fill_used = 0
    matches: list[tuple[float, float, str, np.ndarray, np.ndarray]] = []

    for template_tid in _WORKER_VALID_TARGET_IDS:
        if _WORKER_EXCLUDE_SELF and template_tid == query_tid:
            continue
        template_seq = _WORKER_SEQUENCES[template_tid]
        max_len = max(len(query_seq), len(template_seq))
        if max_len <= 0:
            continue
        length_ratio_gap = abs(len(template_seq) - len(query_seq)) / float(max_len)
        if length_ratio_gap > _WORKER_LENGTH_RATIO_TOLERANCE:
            continue

        candidate_count += 1
        alignment = next(iter(_WORKER_ALIGNER.align(query_seq, template_seq)))
        alignment_attempts += 1
        similarity = float(alignment.score) / float(max(1, 2 * min(len(query_seq), len(template_seq))))
        if similarity < _WORKER_MIN_SIMILARITY:
            continue
        percent_identity = _compute_identity_percent(query_seq, template_seq, alignment)
        if percent_identity < _WORKER_MIN_PERCENT_IDENTITY:
            continue

        adapted_coords = _adapt_template_to_query(
            query_seq=query_seq,
            template_seq=template_seq,
            template_coords=_WORKER_COORDS[template_tid],
            alignment=alignment,
        )
        adapted_residue_idx = _adapt_template_residue_idx_to_query(
            query_seq=query_seq,
            template_seq=template_seq,
            alignment=alignment,
        )
        matches.append(
            (
                similarity,
                float(percent_identity),
                template_tid,
                adapted_coords.astype(np.float32, copy=False),
                adapted_residue_idx.astype(np.int64, copy=False),
            )
        )

    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = matches[:_WORKER_TOP_K]
    for rank, (similarity, percent_identity, source_tid, adapted_coords, adapted_residue_idx) in enumerate(selected):
        topk_coords[rank] = adapted_coords
        topk_valid[rank] = True
        topk_identity[rank] = percent_identity
        topk_similarity[rank] = similarity
        topk_residue_idx[rank] = adapted_residue_idx
        topk_sources[rank] = source_tid

    if _fill_topk_with_diverse_transforms(
        query_tid=query_tid,
        query_len=query_len,
        topk_coords=topk_coords,
        topk_valid=topk_valid,
        topk_identity=topk_identity,
        topk_similarity=topk_similarity,
        topk_residue_idx=topk_residue_idx,
        topk_sources=topk_sources,
        matches=matches,
    ):
        repeated_fill_used = 1

    if bool(topk_valid.any()):
        weights = np.maximum(topk_similarity[topk_valid], 1e-4).astype(np.float32, copy=False)
        weights = weights / np.maximum(weights.sum(), 1e-6)
        consensus = (topk_coords[topk_valid] * weights[:, None, None]).sum(axis=0).astype(np.float32, copy=False)
        available = True
    else:
        consensus = np.zeros_like(query_coords, dtype=np.float32)
        available = False

    return (
        query_tid,
        consensus,
        available,
        topk_coords,
        topk_valid,
        topk_identity,
        topk_similarity,
        topk_residue_idx,
        topk_sources,
        candidate_count,
        alignment_attempts,
        repeated_fill_used,
    )


def precompute_full_length_template_coords(
    labels_path: str | Path,
    output_path: str | Path,
    sequences_path: str | Path | None = None,
    top_k_store: int = 5,
    max_residues_per_target: int = 5120,
    max_targets: int | None = None,
    query_max_targets: int | None = None,
    search_pool_max_targets: int | None = None,
    exclude_self: bool = True,
    min_percent_identity: float = 50.0,
    min_similarity: float = 0.0,
    length_ratio_tolerance: float = 0.3,
    num_threads: int = 0,
) -> dict[str, Any]:
    labels = Path(labels_path)
    sequences_path_resolved = resolve_sequences_path(labels_path=labels, sequences_path=sequences_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    query_cap = query_max_targets if query_max_targets is not None else max_targets
    search_cap = search_pool_max_targets if search_pool_max_targets is not None else max_targets

    search_sequences, search_coords = _load_sequences_and_coords(
        labels_path=labels,
        max_residues_per_target=max_residues_per_target,
        max_targets=search_cap,
        sequences_path=sequences_path_resolved,
    )
    if not search_sequences:
        raise RuntimeError(f"No valid RNA targets found in labels file: {labels}")

    query_target_ids = list(search_sequences.keys())
    if query_cap is not None:
        query_target_ids = query_target_ids[: max(0, int(query_cap))]
    if not query_target_ids:
        raise RuntimeError("No query targets selected for full-length template precompute.")

    worker_threads = int(num_threads)
    if worker_threads <= 0:
        worker_threads = min(max(1, os.cpu_count() or 1), max(1, len(query_target_ids)))

    stores = _new_full_template_stores()

    total_candidate_targets = 0
    total_alignment_attempts = 0
    total_repeated_fill_targets = 0
    start_time = time.time()

    def _store(result: _FullTemplateWorkerResult) -> None:
        nonlocal total_candidate_targets
        nonlocal total_alignment_attempts
        nonlocal total_repeated_fill_targets
        candidate_count, alignment_attempts, repeated_fill_used = _store_full_template_result(stores, result)
        total_candidate_targets += candidate_count
        total_alignment_attempts += alignment_attempts
        total_repeated_fill_targets += repeated_fill_used

    def _log_progress(done: int, force: bool = False) -> None:
        if done <= 0:
            return
        if (not force) and (done % max(1, min(25, len(query_target_ids) // 20 or 1)) != 0):
            return
        elapsed = time.time() - start_time
        rate = float(done) / max(elapsed, 1e-6)
        remaining = max(0, len(query_target_ids) - done)
        eta = remaining / max(rate, 1e-6)
        log.info(
            "Full-length template precompute progress: %d/%d targets (%.1f%%), elapsed=%s, eta=%s",
            done,
            len(query_target_ids),
            100.0 * float(done) / float(max(1, len(query_target_ids))),
            _format_hms(elapsed),
            _format_hms(eta),
        )

    log.info(
        "Full-length template precompute started for %d query targets with %d search-pool targets using %d process(es) "
        "(alignment=global, search=full_exhaustive_alignment, identity_gate=%.1f, similarity_gate=%.3f, length_ratio_tolerance=%.3f).",
        len(query_target_ids),
        len(search_sequences),
        worker_threads,
        float(min_percent_identity),
        float(min_similarity),
        float(length_ratio_tolerance),
    )

    valid_target_ids = list(search_sequences.keys())
    if worker_threads > 1 and len(query_target_ids) > 1:
        with ProcessPoolExecutor(
            max_workers=worker_threads,
            initializer=_init_full_template_worker,
            initargs=(
                search_sequences,
                search_coords,
                valid_target_ids,
                int(top_k_store),
                bool(exclude_self),
                float(min_percent_identity),
                float(min_similarity),
                float(length_ratio_tolerance),
            ),
        ) as executor:
            futures = [executor.submit(_compute_single_full_template_worker, query_tid) for query_tid in query_target_ids]
            done = 0
            for future in as_completed(futures):
                _store(future.result())
                done += 1
                _log_progress(done)
            _log_progress(done, force=True)
    else:
        _init_full_template_worker(
            sequences=search_sequences,
            coords_by_target=search_coords,
            valid_target_ids=valid_target_ids,
            top_k_store=int(top_k_store),
            exclude_self=bool(exclude_self),
            min_percent_identity=float(min_percent_identity),
            min_similarity=float(min_similarity),
            length_ratio_tolerance=float(length_ratio_tolerance),
        )
        for idx, query_tid in enumerate(query_target_ids, start=1):
            _store(_compute_single_full_template_worker(query_tid))
            _log_progress(idx)
        _log_progress(len(query_target_ids), force=True)

    payload = _build_full_template_payload(
        stores=stores,
        labels=labels,
        sequences_path_resolved=sequences_path_resolved,
        top_k_store=top_k_store,
        max_residues_per_target=max_residues_per_target,
        max_targets=max_targets,
        query_cap=query_cap,
        search_cap=search_cap,
        exclude_self=exclude_self,
        min_percent_identity=min_percent_identity,
        min_similarity=min_similarity,
        length_ratio_tolerance=length_ratio_tolerance,
        query_target_ids=query_target_ids,
        search_sequences=search_sequences,
        total_candidate_targets=total_candidate_targets,
        total_alignment_attempts=total_alignment_attempts,
        total_repeated_fill_targets=total_repeated_fill_targets,
        worker_threads=worker_threads,
    )
    torch.save(payload, output)
    log.info(
        "Saved full-length template payload to %s (targets=%d, avg candidates %.2f/target, avg alignments %.2f/target, repeated_fill=%d).",
        output,
        len(query_target_ids),
        float(total_candidate_targets) / float(max(1, len(query_target_ids))),
        float(total_alignment_attempts) / float(max(1, len(query_target_ids))),
        total_repeated_fill_targets,
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute full-length AF3-style templates from train_labels.csv")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--labels-file", type=str, default="train_labels.csv")
    parser.add_argument("--sequences-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="full_length_template_coords.pt")
    parser.add_argument("--top-k-store", type=int, default=5)
    parser.add_argument("--max-residues-per-target", type=int, default=5120)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--query-max-targets", type=int, default=0)
    parser.add_argument("--search-pool-max-targets", type=int, default=0)
    parser.add_argument("--min-percent-identity", type=float, default=50.0)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--length-ratio-tolerance", type=float, default=0.3)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--include-self", action="store_true", help="Allow self-target as a template source.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    labels_path = data_dir / args.labels_file
    sequences_path = (data_dir / args.sequences_file) if str(args.sequences_file).strip() else None
    output_path = data_dir / args.output_file
    max_targets = None if int(args.max_targets) <= 0 else int(args.max_targets)
    query_max_targets = None if int(args.query_max_targets) <= 0 else int(args.query_max_targets)
    search_pool_max_targets = None if int(args.search_pool_max_targets) <= 0 else int(args.search_pool_max_targets)
    precompute_full_length_template_coords(
        labels_path=labels_path,
        sequences_path=sequences_path,
        output_path=output_path,
        top_k_store=args.top_k_store,
        max_residues_per_target=args.max_residues_per_target,
        max_targets=max_targets,
        query_max_targets=query_max_targets,
        search_pool_max_targets=search_pool_max_targets,
        exclude_self=not args.include_self,
        min_percent_identity=args.min_percent_identity,
        min_similarity=args.min_similarity,
        length_ratio_tolerance=args.length_ratio_tolerance,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
