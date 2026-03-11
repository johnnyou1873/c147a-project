from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
DEFAULT_RESIDUE_IDX = 4


@dataclass
class TargetRows:
    target_id: str
    rows: List[tuple[int, int, str, int, float, float, float]]


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


def _load_target_rows(
    labels_path: Path,
    max_targets: int,
    max_residues_per_target: int,
) -> Dict[str, TargetRows]:
    targets: Dict[str, TargetRows] = {}
    selected: List[str] = []
    selected_set: set[str] = set()

    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_idx, row in enumerate(reader):
            full_id = str(row.get("ID", ""))
            if "_" not in full_id:
                continue
            target_id, pos_token = full_id.rsplit("_", 1)

            if target_id not in selected_set:
                if len(selected) >= int(max_targets):
                    continue
                selected.append(target_id)
                selected_set.add(target_id)
                targets[target_id] = TargetRows(target_id=target_id, rows=[])

            target_rows = targets[target_id].rows
            if len(target_rows) >= int(max_residues_per_target):
                continue

            try:
                pos_idx = int(float(pos_token))
                resid = int(float(row.get("resid", 0)))
                resname = str(row.get("resname", "N")).upper()
                x = float(row.get("x_1", 0.0))
                y = float(row.get("y_1", 0.0))
                z = float(row.get("z_1", 0.0))
            except ValueError:
                continue

            target_rows.append((pos_idx, line_idx, resname, resid, x, y, z))

    return targets


def _build_payload(
    targets: Dict[str, TargetRows],
    top_k: int,
    chunk_length: int,
    chunk_stride: int,
    chunk_max_windows: int,
    max_targets: int,
    max_residues_per_target: int,
) -> dict:
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

    total_residues = 0
    total_windows = 0

    for target_id, target_rows in targets.items():
        rows = sorted(target_rows.rows, key=lambda x: (x[0], x[1]))
        if not rows:
            continue

        seq_len = len(rows)
        total_residues += seq_len
        residue_idx = torch.tensor([RESNAME_TO_IDX.get(r[2], DEFAULT_RESIDUE_IDX) for r in rows], dtype=torch.long)
        coords = torch.tensor([[r[4], r[5], r[6]] for r in rows], dtype=torch.float32)

        c_coords = torch.zeros(
            (chunk_max_windows, top_k, chunk_length, 3),
            dtype=torch.float32,
        )
        c_mask = torch.zeros((chunk_max_windows, chunk_length), dtype=torch.bool)
        c_start = torch.zeros((chunk_max_windows,), dtype=torch.long)
        c_window_valid = torch.zeros((chunk_max_windows,), dtype=torch.bool)
        c_valid = torch.zeros((chunk_max_windows, top_k), dtype=torch.bool)
        c_identity = torch.zeros((chunk_max_windows, top_k), dtype=torch.float32)
        c_similarity = torch.zeros((chunk_max_windows, top_k), dtype=torch.float32)
        c_confidence = torch.zeros((chunk_max_windows, top_k), dtype=torch.float32)
        c_residue_idx = torch.full((chunk_max_windows, top_k, chunk_length), DEFAULT_RESIDUE_IDX, dtype=torch.long)
        c_source_type = torch.zeros((chunk_max_windows, top_k), dtype=torch.long)
        c_source_onehot = torch.zeros((chunk_max_windows, top_k, 2), dtype=torch.float32)
        source_rows = [[""] * top_k for _ in range(chunk_max_windows)]

        starts = _compute_chunk_starts(seq_len, chunk_length, chunk_stride, chunk_max_windows)
        total_windows += len(starts)
        for w_idx, start_idx in enumerate(starts):
            end_idx = min(start_idx + chunk_length, seq_len)
            seg_len = max(0, end_idx - start_idx)
            if seg_len <= 0:
                continue

            seg_coords = coords[start_idx:end_idx]
            seg_residue = residue_idx[start_idx:end_idx]
            c_start[w_idx] = int(start_idx)
            c_window_valid[w_idx] = True
            c_mask[w_idx, :seg_len] = True

            for k_idx in range(top_k):
                c_coords[w_idx, k_idx, :seg_len] = seg_coords
                c_valid[w_idx, k_idx] = True
                c_identity[w_idx, k_idx] = 100.0
                c_similarity[w_idx, k_idx] = 1.0
                c_confidence[w_idx, k_idx] = 0.0
                c_residue_idx[w_idx, k_idx, :seg_len] = seg_residue
                c_source_type[w_idx, k_idx] = 0
                c_source_onehot[w_idx, k_idx, 0] = 1.0
                c_source_onehot[w_idx, k_idx, 1] = 0.0
                source_rows[w_idx][k_idx] = f"oracle:{target_id}:{int(start_idx)}:{k_idx}"

        templates[target_id] = coords
        available[target_id] = True
        chunk_topk_templates[target_id] = c_coords
        chunk_mask[target_id] = c_mask
        chunk_start[target_id] = c_start
        chunk_window_valid[target_id] = c_window_valid
        chunk_topk_valid[target_id] = c_valid
        chunk_topk_identity[target_id] = c_identity
        chunk_topk_similarity[target_id] = c_similarity
        chunk_topk_confidence[target_id] = c_confidence
        chunk_topk_residue_idx[target_id] = c_residue_idx
        chunk_topk_source_type[target_id] = c_source_type
        chunk_topk_source_onehot[target_id] = c_source_onehot
        chunk_topk_sources[target_id] = source_rows

    num_targets = len(templates)
    avg_len = float(total_residues) / float(max(1, num_targets))
    avg_windows = float(total_windows) / float(max(1, num_targets))

    return {
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
        "protenix_chunks_used": {},
        "meta": {
            "version": 1,
            "toy_payload": True,
            "chunk_selection_policy": "oracle_only_toy_payload",
            "alignment_mode": "none",
            "search_strategy": "none",
            "top_k_store": int(top_k),
            "max_targets": int(max_targets),
            "max_residues_per_target": int(max_residues_per_target),
            "chunk_length": int(chunk_length),
            "chunk_stride": int(chunk_stride),
            "chunk_max_windows": int(chunk_max_windows),
            "allow_self_fallback": False,
            "exclude_self": True,
            "num_targets": int(num_targets),
            "avg_seq_len": avg_len,
            "avg_windows_per_target": avg_windows,
        },
    }


def build_toy_templates(
    labels_path: Path,
    output_path: Path,
    summary_path: Path,
    max_targets: int,
    max_residues_per_target: int,
    top_k: int,
    chunk_length: int,
    chunk_stride: int,
    chunk_max_windows: int,
) -> None:
    targets = _load_target_rows(
        labels_path=labels_path,
        max_targets=max_targets,
        max_residues_per_target=max_residues_per_target,
    )
    payload = _build_payload(
        targets=targets,
        top_k=top_k,
        chunk_length=chunk_length,
        chunk_stride=chunk_stride,
        chunk_max_windows=chunk_max_windows,
        max_targets=max_targets,
        max_residues_per_target=max_residues_per_target,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    summary = {
        "output_path": str(output_path),
        "labels_path": str(labels_path),
        "num_targets": int(payload["meta"]["num_targets"]),
        "max_targets": int(max_targets),
        "max_residues_per_target": int(max_residues_per_target),
        "top_k": int(top_k),
        "chunk_length": int(chunk_length),
        "chunk_stride": int(chunk_stride),
        "chunk_max_windows": int(chunk_max_windows),
        "avg_seq_len": float(payload["meta"]["avg_seq_len"]),
        "avg_windows_per_target": float(payload["meta"]["avg_windows_per_target"]),
        "toy_payload": True,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build toy oracle template payload for fast model smoke testing.")
    parser.add_argument("--labels-file", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--output-file", type=Path, default=Path("data/toy-templates.pt"))
    parser.add_argument("--summary-file", type=Path, default=Path("data/toy-templates.json"))
    parser.add_argument("--max-targets", type=int, default=512)
    parser.add_argument("--max-residues-per-target", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--chunk-stride", type=int, default=256)
    parser.add_argument("--chunk-max-windows", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_toy_templates(
        labels_path=args.labels_file,
        output_path=args.output_file,
        summary_path=args.summary_file,
        max_targets=max(1, int(args.max_targets)),
        max_residues_per_target=max(1, int(args.max_residues_per_target)),
        top_k=max(1, int(args.top_k)),
        chunk_length=max(1, int(args.chunk_length)),
        chunk_stride=max(1, int(args.chunk_stride)),
        chunk_max_windows=max(1, int(args.chunk_max_windows)),
    )
