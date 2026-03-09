from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.Align import PairwiseAligner


log = logging.getLogger(__name__)

RNA_RESIDUES = {"A", "C", "G", "U", "T"}
RESNAME_TO_BASE = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U"}


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


def precompute_template_coords(
    labels_path: str | Path,
    output_path: str | Path,
    min_percent_identity: float = 50.0,
    min_similarity: float = 0.1,
    max_templates: int = 8,
    max_residues_per_target: int = 5120,
    max_targets: int | None = None,
    length_ratio_tolerance: float = 0.3,
    exclude_self: bool = True,
) -> dict[str, Any]:
    labels = Path(labels_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sequences, coords_by_target = _load_sequences_and_coords(
        labels_path=labels,
        max_residues_per_target=max_residues_per_target,
        max_targets=max_targets,
    )
    aligner = _make_aligner()

    valid_target_ids = [tid for tid, c in coords_by_target.items() if _coords_are_valid(c)]
    exact_idx: dict[str, list[str]] = {}
    for tid in valid_target_ids:
        exact_idx.setdefault(sequences[tid], []).append(tid)

    templates: dict[str, torch.Tensor] = {}
    available: dict[str, bool] = {}
    sources: dict[str, list[str]] = {}

    for query_tid in valid_target_ids:
        query_seq = sequences[query_tid]
        q_len = len(query_seq)
        candidates: list[tuple[str, float, float, Any]] = []

        for tid in exact_idx.get(query_seq, []):
            if exclude_self and tid == query_tid:
                continue
            aln = aligner.align(query_seq, sequences[tid])[0]
            candidates.append((tid, 1.0, 100.0, aln))

        for template_tid in valid_target_ids:
            if template_tid == query_tid and exclude_self:
                continue
            template_seq = sequences[template_tid]
            if template_seq == query_seq:
                continue

            len_ratio = abs(len(template_seq) - q_len) / float(max(len(template_seq), q_len, 1))
            if len_ratio > length_ratio_tolerance:
                continue

            aln = aligner.align(query_seq, template_seq)[0]
            norm_similarity = float(aln.score) / float(2.0 * min(q_len, len(template_seq), 1))
            pct_identity = _compute_identity_percent(query_seq, template_seq, aln)
            if norm_similarity < min_similarity or pct_identity < min_percent_identity:
                continue
            candidates.append((template_tid, norm_similarity, pct_identity, aln))

        dedup: dict[str, tuple[str, float, float, Any]] = {}
        for item in sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True):
            if item[0] not in dedup:
                dedup[item[0]] = item
        selected = list(dedup.values())[:max_templates]

        if not selected:
            templates[query_tid] = torch.zeros((q_len, 3), dtype=torch.float32)
            available[query_tid] = False
            sources[query_tid] = []
            continue

        adapted_coords = []
        weights = []
        used_ids = []
        for template_tid, sim, _, aln in selected:
            adapted = _adapt_template_to_query(
                query_seq=query_seq,
                template_seq=sequences[template_tid],
                template_coords=coords_by_target[template_tid],
                alignment=aln,
            )
            adapted_coords.append(adapted)
            weights.append(max(0.0, sim))
            used_ids.append(template_tid)

        w = np.asarray(weights, dtype=np.float32)
        if np.all(w <= 0):
            w[:] = 1.0
        w = np.exp(w - np.max(w))
        w = w / np.sum(w)
        stacked = np.stack(adapted_coords, axis=0)  # (K, L, 3)
        consensus = np.tensordot(w, stacked, axes=(0, 0)).astype(np.float32)

        templates[query_tid] = torch.from_numpy(consensus)
        available[query_tid] = True
        sources[query_tid] = used_ids

    payload = {
        "templates": templates,
        "available": available,
        "sources": sources,
        "meta": {
            "labels_path": str(labels.resolve()),
            "min_percent_identity": float(min_percent_identity),
            "min_similarity": float(min_similarity),
            "max_templates": int(max_templates),
            "max_residues_per_target": int(max_residues_per_target),
            "max_targets": None if max_targets is None else int(max_targets),
            "length_ratio_tolerance": float(length_ratio_tolerance),
            "exclude_self": bool(exclude_self),
            "num_targets": int(len(valid_target_ids)),
        },
    }
    torch.save(payload, output)
    log.info("Saved template coordinates to %s (targets=%d)", output, len(valid_target_ids))
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute template coordinates from train_labels.csv")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--labels-file", type=str, default="train_labels.csv")
    parser.add_argument("--output-file", type=str, default="template_coords.pt")
    parser.add_argument("--min-percent-identity", type=float, default=50.0)
    parser.add_argument("--min-similarity", type=float, default=0.1)
    parser.add_argument("--max-templates", type=int, default=8)
    parser.add_argument("--max-residues-per-target", type=int, default=5120)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--length-ratio-tolerance", type=float, default=0.3)
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
        max_residues_per_target=args.max_residues_per_target,
        max_targets=max_targets,
        length_ratio_tolerance=args.length_ratio_tolerance,
        exclude_self=not args.include_self,
    )


if __name__ == "__main__":
    main()
