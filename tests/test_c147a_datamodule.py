from __future__ import annotations

import csv
import shutil
import uuid
from pathlib import Path

import torch

from src.data.c147a_datamodule import C147ADataModule


def _write_labels_csv(path: Path) -> None:
    fieldnames = ["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"]
    residues = ["A", "C", "G"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target_idx in range(4):
            tid = f"T{target_idx}"
            for pos in range(1, 4):
                writer.writerow(
                    {
                        "ID": f"{tid}_{pos}",
                        "resname": residues[(pos - 1) % len(residues)],
                        "resid": pos,
                        "x_1": float(target_idx * 10 + pos),
                        "y_1": float(pos),
                        "z_1": float(-pos),
                        "chain": "A",
                        "copy": 1,
                    }
                )


def _coords_for_target(target_idx: int) -> torch.Tensor:
    rows = []
    for pos in range(1, 4):
        rows.append([float(target_idx * 10 + pos), float(pos), float(-pos)])
    return torch.tensor(rows, dtype=torch.float32)


def _write_template_payload(path: Path, identity_pair: tuple[float, float] = (90.0, 80.0)) -> None:
    # Deterministic train split for seed=42 with 4 targets -> train={T2,T3}, val={T0}, test={T1}.
    source_plan = {
        "T0": ["T2", "T0"],
        "T1": ["T3", "T1"],
        "T2": ["T3", "T2"],
        "T3": ["T2", "T3"],
    }

    templates: dict[str, torch.Tensor] = {}
    available: dict[str, bool] = {}
    topk_templates: dict[str, torch.Tensor] = {}
    topk_mask: dict[str, torch.Tensor] = {}
    topk_sources: dict[str, list[str]] = {}
    topk_identity: dict[str, torch.Tensor] = {}
    topk_similarity: dict[str, torch.Tensor] = {}
    chunk_topk_templates: dict[str, torch.Tensor] = {}
    chunk_mask: dict[str, torch.Tensor] = {}
    chunk_start: dict[str, torch.Tensor] = {}
    chunk_window_valid: dict[str, torch.Tensor] = {}
    chunk_topk_valid: dict[str, torch.Tensor] = {}
    chunk_topk_identity: dict[str, torch.Tensor] = {}
    chunk_topk_similarity: dict[str, torch.Tensor] = {}
    chunk_topk_sources: dict[str, list[list[str]]] = {}

    for target_idx in range(4):
        tid = f"T{target_idx}"
        src_a, src_b = source_plan[tid]
        src_a_idx = int(src_a[1:])
        src_b_idx = int(src_b[1:])

        cand_a = _coords_for_target(src_a_idx) + 5.0
        cand_b = _coords_for_target(src_b_idx) + 7.0
        topk = torch.stack([cand_a, cand_b], dim=0)

        templates[tid] = cand_a.clone()
        available[tid] = True
        topk_templates[tid] = topk
        topk_mask[tid] = torch.tensor([True, True], dtype=torch.bool)
        topk_sources[tid] = [src_a, src_b]
        topk_identity[tid] = torch.tensor([float(identity_pair[0]), float(identity_pair[1])], dtype=torch.float32)
        topk_similarity[tid] = torch.tensor([0.9, 0.8], dtype=torch.float32)
        chunk_topk_templates[tid] = topk.unsqueeze(0)  # (W=1,K=2,L=3,3)
        chunk_mask[tid] = torch.tensor([[True, True, True]], dtype=torch.bool)
        chunk_start[tid] = torch.tensor([0], dtype=torch.long)
        chunk_window_valid[tid] = torch.tensor([True], dtype=torch.bool)
        chunk_topk_valid[tid] = torch.tensor([[True, True]], dtype=torch.bool)
        chunk_topk_identity[tid] = torch.tensor(
            [[float(identity_pair[0]), float(identity_pair[1])]], dtype=torch.float32
        )
        chunk_topk_similarity[tid] = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
        chunk_topk_sources[tid] = [[src_a, src_b]]

    payload = {
        "templates": templates,
        "available": available,
        "sources": {k: list(v) for k, v in topk_sources.items()},
        "topk_templates": topk_templates,
        "topk_mask": topk_mask,
        "topk_sources": topk_sources,
        "topk_identity": topk_identity,
        "topk_similarity": topk_similarity,
        "chunk_topk_templates": chunk_topk_templates,
        "chunk_mask": chunk_mask,
        "chunk_start": chunk_start,
        "chunk_window_valid": chunk_window_valid,
        "chunk_topk_valid": chunk_topk_valid,
        "chunk_topk_identity": chunk_topk_identity,
        "chunk_topk_similarity": chunk_topk_similarity,
        "chunk_topk_sources": chunk_topk_sources,
        "meta": {
            "labels_path": "synthetic",
            "chunk_selection_policy": "chunked_topk_non_self_by_identity_similarity_no_threshold",
            "top_k_store": 2,
            "max_residues_per_target": 64,
            "max_targets": None,
            "exclude_self": True,
            "enforce_min_topk": True,
            "allow_self_fallback": False,
            "chunk_length": 512,
            "chunk_stride": 256,
            "chunk_max_windows": 20,
            "num_targets": 4,
            "num_threads": 1,
            "num_targets_with_self_fallback": 0,
            "num_targets_with_repeated_templates": 0,
        },
    }
    torch.save(payload, path)


def _split_target_ids(split) -> set[str]:
    base = split.dataset if hasattr(split, "dataset") else split
    indices = list(split.indices) if hasattr(split, "indices") else list(range(len(split)))
    return {base[idx].target_id for idx in indices}


def test_c147a_split_disjoint_and_template_only_inputs() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"c147a_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    payload_path = tmp_path / "template_coords.pt"
    try:
        _write_labels_csv(labels_path)
        _write_template_payload(payload_path)

        dm = C147ADataModule(
            data_dir=str(tmp_path),
            labels_file="train_labels.csv",
            template_file="template_coords.pt",
            batch_size=2,
            num_workers=0,
            train_fraction=0.5,
            val_fraction=0.25,
            max_residues_per_target=64,
            max_targets=None,
            temperature_k=0.0,
            apply_thermal_noise_train=False,
            apply_thermal_noise_eval=False,
            train_use_template_only_inputs=True,
            eval_use_template_only_inputs=True,
            use_template_coords=True,
            precompute_templates_if_missing=False,
            template_topk_count=2,
            template_min_percent_identity=50.0,
            seed=42,
        )

        dm.prepare_data()
        dm.setup()

        train_ids = _split_target_ids(dm.data_train)
        val_ids = _split_target_ids(dm.data_val)
        test_ids = _split_target_ids(dm.data_test)

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

        batch = next(iter(dm.train_dataloader()))
        assert torch.allclose(batch["coords"], batch["template_coords"])
        assert not torch.allclose(batch["coords"], batch["target_coords"])
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_c147a_topk_ignores_identity_threshold_and_keeps_features() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"c147a_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    payload_path = tmp_path / "template_coords.pt"
    try:
        _write_labels_csv(labels_path)
        # Deliberately low identity values to verify they are still retained/used.
        _write_template_payload(payload_path, identity_pair=(12.0, 8.0))

        dm = C147ADataModule(
            data_dir=str(tmp_path),
            labels_file="train_labels.csv",
            template_file="template_coords.pt",
            batch_size=2,
            num_workers=0,
            train_fraction=0.5,
            val_fraction=0.25,
            max_residues_per_target=64,
            max_targets=None,
            temperature_k=0.0,
            apply_thermal_noise_train=False,
            apply_thermal_noise_eval=False,
            train_use_template_only_inputs=True,
            eval_use_template_only_inputs=True,
            use_template_coords=True,
            precompute_templates_if_missing=False,
            template_topk_count=2,
            template_min_percent_identity=95.0,
            seed=42,
        )

        dm.prepare_data()
        dm.setup()
        batch = next(iter(dm.train_dataloader()))

        # All slots remain active in train after non-self pruning + fill, despite low identity.
        assert bool((batch["template_topk_valid"].all()).item())
        assert float(batch["template_topk_identity"].max().item()) < 95.0
        # Identity/similarity features are still present and non-zero for active candidates.
        assert float(batch["template_topk_identity"].sum().item()) > 0.0
        assert float(batch["template_topk_similarity"].sum().item()) > 0.0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
