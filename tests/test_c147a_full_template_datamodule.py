from __future__ import annotations

import csv
import logging
import os
import pickle
from pathlib import Path
import shutil
from threading import Lock
import time
import uuid

import pytest
import torch

import src.data.c147a_full_template_datamodule as full_template_module
from src.data.c147a_full_template_datamodule import (
    C147AFullTemplateDataModule,
    RNAFullTemplateDataset,
    _collate_full_template_batch,
)
from src.data.precompute_full_length_templates import (
    FULL_LENGTH_SELECTION_POLICY,
    precompute_full_length_template_coords,
)
from src.data.eternafold_bpp import prune_stale_eternafold_cache_locks
from src.data.eternafold_bpp import EternaFoldCacheWarmer, eternafold_cache_path


def _write_labels_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
        writer.writeheader()
        writer.writerow(
            {
                "ID": "T0_1",
                "resname": "A",
                "resid": 1,
                "x_1": 0.0,
                "y_1": 0.0,
                "z_1": 0.0,
                "chain": "A",
                "copy": 0,
            }
        )


def _write_sequences_csv(
    path: Path,
    sequence: str = "A",
    stoichiometry: str = "A:1",
    all_sequences: str | None = None,
) -> None:
    fasta = all_sequences or f">T0_1|Chain A[auth A]|RNA|\n{sequence}\n"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_id",
                "sequence",
                "temporal_cutoff",
                "description",
                "stoichiometry",
                "all_sequences",
                "ligand_ids",
                "ligand_SMILES",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "target_id": "T0",
                "sequence": sequence,
                "temporal_cutoff": "2025-01-01",
                "description": "test target",
                "stoichiometry": stoichiometry,
                "all_sequences": fasta,
                "ligand_ids": "",
                "ligand_SMILES": "",
            }
        )


def _write_partial_full_length_payload(path: Path, labels_path: Path, sequences_path: Path | None = None) -> None:
    payload = {
        "templates": {"T0": torch.zeros((1, 3), dtype=torch.float32)},
        "available": {"T0": True},
        "template_topk_coords": {"T0": torch.zeros((5, 1, 3), dtype=torch.float32)},
        "template_topk_valid": {"T0": torch.tensor([True, False, False, False, False], dtype=torch.bool)},
        "template_topk_identity": {"T0": torch.tensor([85.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)},
        "template_topk_similarity": {"T0": torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)},
        "template_topk_residue_idx": {"T0": torch.zeros((5, 1), dtype=torch.long)},
        "template_topk_sources": {"T0": ["S0", "", "", "", ""]},
        "meta": {
            "labels_path": str(labels_path.resolve()),
            "sequences_path": None if sequences_path is None else str(sequences_path.resolve()),
            "selection_policy": FULL_LENGTH_SELECTION_POLICY,
            "alignment_mode": "global",
            "search_strategy": "full_exhaustive_alignment",
            "top_k_store": 5,
            "max_residues_per_target": 5120,
            "query_max_targets": None,
            "search_pool_max_targets": None,
            "exclude_self": True,
            "min_percent_identity": 40.0,
            "min_similarity": 0.0,
            "length_ratio_tolerance": 0.3,
        },
    }
    torch.save(payload, path)


def test_full_length_payload_with_partial_topk_does_not_recompute(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_prepare_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    payload_path = tmp_path / "full_length_template_coords.pt"

    try:
        _write_labels_csv(labels_path)
        _write_partial_full_length_payload(payload_path, labels_path)

        def _fail_precompute(*_args, **_kwargs):
            raise AssertionError("prepare_data() should not recompute this full-length template payload.")

        monkeypatch.setattr(full_template_module, "precompute_full_length_template_coords", _fail_precompute)

        dm = C147AFullTemplateDataModule(
            data_dir=str(tmp_path),
            labels_file="train_labels.csv",
            template_file="full_length_template_coords.pt",
            precompute_templates_if_missing=True,
            use_template_coords=True,
            use_rna_msa_features=False,
            template_topk_count=5,
            template_min_percent_identity=40.0,
            template_min_similarity=0.0,
            template_length_ratio_tolerance=0.3,
            max_residues_per_target=5120,
            max_targets=None,
            template_generation_max_targets=None,
            use_rna_bpp_features=False,
        )

        dm.prepare_data()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _write_msa_fasta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ">query\n"
        "ACG\n"
        ">homolog_1|chain=A\n"
        "A-G\n"
        ">homolog_2|chain=A\n"
        "AC-\n",
        encoding="utf-8",
    )


def test_full_template_dataset_uses_precomputed_fasta_msa() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_msa_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    msa_dir = tmp_path / "MSA"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for pos, resname in enumerate(["A", "C", "G"], start=1):
                writer.writerow(
                    {
                        "ID": f"T0_{pos}",
                        "resname": resname,
                        "resid": pos,
                        "x_1": float(pos),
                        "y_1": 0.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )

        _write_sequences_csv(sequences_path, sequence="ACG")
        _write_msa_fasta(msa_dir / "T0.MSA.fasta")

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=16,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=False,
            use_rna_msa_features=True,
            rna_msa_max_rows=3,
            rna_msa_fasta_dir=msa_dir,
        )

        example = dataset[0]
        assert example.target_id == "T0"
        assert example.rna_msa_tokens is not None
        assert example.rna_msa_mask is not None
        assert example.rna_msa_row_valid is not None
        assert example.rna_msa_profile is not None
        assert example.rna_msa_tokens.shape == (3, 3)
        assert torch.equal(example.rna_msa_tokens[0], torch.tensor([0, 1, 2], dtype=torch.long))
        assert torch.equal(example.rna_msa_tokens[1], torch.tensor([0, 4, 2], dtype=torch.long))
        assert torch.equal(example.rna_msa_tokens[2], torch.tensor([0, 1, 4], dtype=torch.long))
        assert torch.equal(example.rna_msa_mask[1], torch.tensor([True, False, True]))
        assert torch.equal(example.rna_msa_row_valid, torch.tensor([True, True, True]))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_caches_msa_tensors_by_target(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_msa_cache_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    msa_dir = tmp_path / "MSA"
    calls = {"count": 0}

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for pos, resname in enumerate(["A", "C", "G"], start=1):
                writer.writerow(
                    {
                        "ID": f"T0_{pos}",
                        "resname": resname,
                        "resid": pos,
                        "x_1": float(pos),
                        "y_1": 0.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )

        _write_sequences_csv(sequences_path, sequence="ACG")
        _write_msa_fasta(msa_dir / "T0.MSA.fasta")

        original_builder = full_template_module.build_precomputed_rna_msa_tensors

        def _counting_builder(*args, **kwargs):
            calls["count"] += 1
            return original_builder(*args, **kwargs)

        monkeypatch.setattr(full_template_module, "build_precomputed_rna_msa_tensors", _counting_builder)

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=16,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=False,
            use_rna_msa_features=True,
            rna_msa_max_rows=3,
            rna_msa_fasta_dir=msa_dir,
        )

        _ = dataset[0]
        dataset._example_cache.clear()
        _ = dataset[0]

        assert calls["count"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_preserves_blank_coord_rows_for_fasta_msa() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_msa_missing_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    msa_dir = tmp_path / "MSA"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            rows = [
                {"ID": "T0_1", "resname": "A", "resid": 1, "x_1": "", "y_1": "", "z_1": "", "chain": "A", "copy": 0},
                {"ID": "T0_2", "resname": "C", "resid": 2, "x_1": 1.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 0},
                {"ID": "T0_3", "resname": "G", "resid": 3, "x_1": 2.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 0},
                {"ID": "T0_4", "resname": "U", "resid": 4, "x_1": 3.0, "y_1": 0.0, "z_1": 0.0, "chain": "A", "copy": 0},
                {"ID": "T0_5", "resname": "A", "resid": 5, "x_1": "", "y_1": "", "z_1": "", "chain": "A", "copy": 0},
            ]
            writer.writerows(rows)

        _write_sequences_csv(sequences_path, sequence="ACGUA")
        (msa_dir / "T0.MSA.fasta").parent.mkdir(parents=True, exist_ok=True)
        (msa_dir / "T0.MSA.fasta").write_text(
            ">query\n"
            "ACGUA\n"
            ">homolog_1|chain=A\n"
            "A-G-A\n"
            ">homolog_2|chain=A\n"
            "AC-U-\n",
            encoding="utf-8",
        )

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=16,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=False,
            use_rna_msa_features=True,
            rna_msa_max_rows=3,
            rna_msa_fasta_dir=msa_dir,
        )

        example = dataset[0]
        batch = _collate_full_template_batch([example], use_template_only_inputs=False, use_rna_msa_features=True, rna_msa_max_rows=3)

        assert example.residue_idx.shape[0] == 5
        assert torch.equal(example.target_mask, torch.tensor([False, True, True, True, False], dtype=torch.bool))
        assert batch["target_mask"].shape == (1, 5)
        assert torch.equal(batch["target_mask"][0], torch.tensor([False, True, True, True, False], dtype=torch.bool))
        assert batch["rna_msa_tokens"].shape == (1, 3, 5)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_collate_keeps_origin_template_residue_valid() -> None:
    example = full_template_module.RNAFullTemplateExample(
        target_id="T0",
        residue_idx=torch.tensor([0], dtype=torch.long),
        chain_idx=torch.tensor([0], dtype=torch.long),
        copy_idx=torch.tensor([0], dtype=torch.long),
        resid_norm=torch.tensor([1.0], dtype=torch.float32),
        coords=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        target_mask=torch.tensor([True], dtype=torch.bool),
        template_coords=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        template_topk_coords=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
        template_topk_valid=torch.tensor([True], dtype=torch.bool),
        template_topk_identity=torch.tensor([100.0], dtype=torch.float32),
        template_topk_similarity=torch.tensor([1.0], dtype=torch.float32),
        template_topk_residue_idx=torch.tensor([[0]], dtype=torch.long),
        template_topk_sources=["oracle:T0:0"],
        rna_msa_tokens=torch.tensor([[0]], dtype=torch.long),
        rna_msa_mask=torch.tensor([[True]], dtype=torch.bool),
        rna_msa_row_valid=torch.tensor([True], dtype=torch.bool),
        rna_msa_profile=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        rna_bpp_banded=torch.zeros((1, 1), dtype=torch.float32),
        has_template=True,
    )

    batch = _collate_full_template_batch([example], use_rna_msa_features=True, rna_msa_max_rows=1)

    assert bool(batch["template_mask"][0, 0].item()) is True


def test_full_template_dataset_rejects_sequence_metadata_mismatch() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_sequence_mismatch_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    msa_dir = tmp_path / "MSA"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for pos, resname in enumerate(["A", "G", "G"], start=1):
                writer.writerow(
                    {
                        "ID": f"T0_{pos}",
                        "resname": resname,
                        "resid": pos,
                        "x_1": float(pos),
                        "y_1": 0.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )

        _write_sequences_csv(sequences_path, sequence="ACG")
        _write_msa_fasta(msa_dir / "T0.MSA.fasta")

        with pytest.raises(ValueError, match="Sequence mismatch"):
            RNAFullTemplateDataset(
                labels_path=labels_path,
                sequences_path=sequences_path,
                max_residues_per_target=16,
                max_targets=None,
                template_topk_count=2,
                use_rna_bpp_features=False,
                use_rna_msa_features=True,
                rna_msa_max_rows=3,
                rna_msa_fasta_dir=msa_dir,
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_requires_fasta_dir_when_msa_enabled() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_missing_msa_dir_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"

    try:
        _write_labels_csv(labels_path)
        _write_sequences_csv(sequences_path, sequence="A")

        with pytest.raises(ValueError, match="rna_msa_fasta_dir"):
            RNAFullTemplateDataset(
                labels_path=labels_path,
                sequences_path=sequences_path,
                max_residues_per_target=16,
                max_targets=None,
                template_topk_count=2,
                use_rna_bpp_features=False,
                use_rna_msa_features=True,
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_requires_msa_files_for_all_targets() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_missing_msa_file_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    msa_dir = tmp_path / "MSA"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for target_id, sequence in (("T0", "A"), ("T1", "C")):
                writer.writerow(
                    {
                        "ID": f"{target_id}_1",
                        "resname": sequence,
                        "resid": 1,
                        "x_1": 0.0,
                        "y_1": 0.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )
        with sequences_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_id",
                    "sequence",
                    "temporal_cutoff",
                    "description",
                    "stoichiometry",
                    "all_sequences",
                    "ligand_ids",
                    "ligand_SMILES",
                ],
            )
            writer.writeheader()
            for target_id, sequence in (("T0", "A"), ("T1", "C")):
                writer.writerow(
                    {
                        "target_id": target_id,
                        "sequence": sequence,
                        "temporal_cutoff": "2025-01-01",
                        "description": target_id,
                        "stoichiometry": "A:1",
                        "all_sequences": f">{target_id}_1|Chain A[auth A]|RNA|\n{sequence}\n",
                        "ligand_ids": "",
                        "ligand_SMILES": "",
                    }
                )
        _write_msa_fasta(msa_dir / "T0.MSA.fasta")

        with pytest.raises(FileNotFoundError, match="Missing required RNA MSA FASTA files"):
            RNAFullTemplateDataset(
                labels_path=labels_path,
                sequences_path=sequences_path,
                max_residues_per_target=16,
                max_targets=None,
                template_topk_count=5,
                use_rna_bpp_features=False,
                use_rna_msa_features=True,
                rna_msa_max_rows=3,
                rna_msa_fasta_dir=msa_dir,
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_length_payload_does_not_recompute_when_labels_path_changes(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_labels_mismatch_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    payload_path = tmp_path / "full_length_template_coords.pt"

    try:
        _write_labels_csv(labels_path)
        wrong_labels_path = tmp_path / "other_train_labels.csv"
        _write_partial_full_length_payload(payload_path, wrong_labels_path)

        called = {"value": False}

        def _mark_precompute(*_args, **_kwargs):
            called["value"] = True

        monkeypatch.setattr(full_template_module, "precompute_full_length_template_coords", _mark_precompute)

        dm = C147AFullTemplateDataModule(
            data_dir=str(tmp_path),
            labels_file="train_labels.csv",
            template_file="full_length_template_coords.pt",
            precompute_templates_if_missing=True,
            use_template_coords=True,
            use_rna_msa_features=False,
            template_topk_count=5,
            template_min_percent_identity=40.0,
            template_min_similarity=0.0,
            template_length_ratio_tolerance=0.3,
            max_residues_per_target=5120,
            max_targets=None,
            template_generation_max_targets=None,
            use_rna_bpp_features=False,
        )

        dm.prepare_data()

        assert called["value"] is False
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_length_payload_does_not_recompute_when_sequences_path_changes(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_sequences_mismatch_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    payload_path = tmp_path / "full_length_template_coords.pt"

    try:
        _write_labels_csv(labels_path)
        _write_sequences_csv(sequences_path, sequence="A")
        wrong_sequences_path = tmp_path / "other_train_sequences.csv"
        _write_partial_full_length_payload(payload_path, labels_path, wrong_sequences_path)

        called = {"value": False}

        def _mark_precompute(*_args, **_kwargs):
            called["value"] = True

        monkeypatch.setattr(full_template_module, "precompute_full_length_template_coords", _mark_precompute)

        dm = C147AFullTemplateDataModule(
            data_dir=str(tmp_path),
            labels_file="train_labels.csv",
            sequences_file="train_sequences.csv",
            template_file="full_length_template_coords.pt",
            precompute_templates_if_missing=True,
            use_template_coords=True,
            use_rna_msa_features=False,
            template_topk_count=5,
            template_min_percent_identity=40.0,
            template_min_similarity=0.0,
            template_length_ratio_tolerance=0.3,
            max_residues_per_target=5120,
            max_targets=None,
            template_generation_max_targets=None,
            use_rna_bpp_features=False,
        )

        dm.prepare_data()

        assert called["value"] is False
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_skips_targets_exceeding_residue_cap() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_overlong_skip_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    msa_dir = tmp_path / "MSA"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for pos, resname in enumerate(["A", "C", "G"], start=1):
                writer.writerow(
                    {
                        "ID": f"T0_{pos}",
                        "resname": resname,
                        "resid": pos,
                        "x_1": float(pos),
                        "y_1": 0.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )
            for pos, resname in enumerate(["A", "C", "G", "U"], start=1):
                writer.writerow(
                    {
                        "ID": f"T1_{pos}",
                        "resname": resname,
                        "resid": pos,
                        "x_1": float(pos),
                        "y_1": 1.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )

        with sequences_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_id",
                    "sequence",
                    "temporal_cutoff",
                    "description",
                    "stoichiometry",
                    "all_sequences",
                    "ligand_ids",
                    "ligand_SMILES",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "target_id": "T0",
                    "sequence": "ACG",
                    "temporal_cutoff": "2025-01-01",
                    "description": "short target",
                    "stoichiometry": "A:1",
                    "all_sequences": ">T0_1|Chain A[auth A]|RNA|\nACG\n",
                    "ligand_ids": "",
                    "ligand_SMILES": "",
                }
            )
            writer.writerow(
                {
                    "target_id": "T1",
                    "sequence": "ACGU",
                    "temporal_cutoff": "2025-01-01",
                    "description": "overlong target",
                    "stoichiometry": "A:1",
                    "all_sequences": ">T1_1|Chain A[auth A]|RNA|\nACGU\n",
                    "ligand_ids": "",
                    "ligand_SMILES": "",
                }
            )

        (msa_dir / "T0.MSA.fasta").parent.mkdir(parents=True, exist_ok=True)
        (msa_dir / "T0.MSA.fasta").write_text(
            ">query\n"
            "ACG\n"
            ">homolog_1|chain=A\n"
            "A-G\n",
            encoding="utf-8",
        )

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=3,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=False,
            use_rna_msa_features=True,
            rna_msa_max_rows=2,
            rna_msa_fasta_dir=msa_dir,
        )

        assert len(dataset) == 1
        example = dataset[0]
        assert example.target_id == "T0"
        assert example.residue_idx.shape[0] == 3
        assert example.rna_msa_tokens is not None
        assert example.rna_msa_tokens.shape == (2, 3)
        assert torch.equal(example.rna_msa_tokens[0], torch.tensor([0, 1, 2], dtype=torch.long))
        assert torch.equal(example.rna_msa_tokens[1], torch.tensor([0, 4, 2], dtype=torch.long))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_warms_eternafold_cache_with_16_threads(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_eternafold_warm_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    binary_path = tmp_path / "contrafold.exe"
    parameters_path = tmp_path / "EternaFoldParams.v1"
    cache_dir = tmp_path / "eternafold_cache"
    seen: dict[str, object] = {}

    class FakeWarmer:
        def __init__(
            self,
            max_workers: int,
            max_span: int,
            cutoff: float,
            binary_path: str | Path | None,
            parameters_path: str | Path | None,
            cache_dir: str | Path,
        ) -> None:
            seen["max_workers"] = max_workers
            seen["max_span"] = max_span
            seen["cutoff"] = cutoff
            seen["binary_path"] = Path(binary_path) if binary_path is not None else None
            seen["parameters_path"] = Path(parameters_path) if parameters_path is not None else None
            seen["cache_dir"] = Path(cache_dir)

        def submit_many(self, items) -> None:
            seen["submitted"] = [(target_id, residue_idx.clone()) for target_id, residue_idx in items]

        def shutdown(self, wait: bool = False) -> None:
            seen["shutdown_wait"] = wait

    try:
        _write_labels_csv(labels_path)
        _write_sequences_csv(sequences_path, sequence="A")
        binary_path.write_text("", encoding="utf-8")
        parameters_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(full_template_module, "EternaFoldCacheWarmer", FakeWarmer)

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=16,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=True,
            rna_bpp_max_span=256,
            rna_bpp_cutoff=1e-4,
            rna_bpp_binary_path=binary_path,
            rna_bpp_parameters_path=parameters_path,
            rna_bpp_cache_dir=cache_dir,
            rna_bpp_num_threads=16,
            use_rna_msa_features=False,
        )

        dataset.warm_rna_bpp_cache(indices=[0])

        assert seen["max_workers"] == 16
        assert seen["max_span"] == 256
        assert seen["cache_dir"] == cache_dir.resolve()
        submitted = seen["submitted"]
        assert isinstance(submitted, list)
        assert len(submitted) == 1
        assert submitted[0][0] == "T0"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_is_pickle_safe_after_bpp_warmup(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_pickle_safe_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    binary_path = tmp_path / "contrafold.exe"
    parameters_path = tmp_path / "EternaFoldParams.v1"
    cache_dir = tmp_path / "eternafold_cache"

    class FakeWarmer:
        def __init__(self, *args, **kwargs) -> None:
            self._lock = Lock()

        def submit_many(self, items) -> None:
            list(items)

        def shutdown(self, wait: bool = False) -> None:
            return None

    try:
        _write_labels_csv(labels_path)
        _write_sequences_csv(sequences_path, sequence="A")
        binary_path.write_text("", encoding="utf-8")
        parameters_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(full_template_module, "EternaFoldCacheWarmer", FakeWarmer)

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=16,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=True,
            rna_bpp_max_span=256,
            rna_bpp_cutoff=1e-4,
            rna_bpp_binary_path=binary_path,
            rna_bpp_parameters_path=parameters_path,
            rna_bpp_cache_dir=cache_dir,
            rna_bpp_num_threads=16,
            use_rna_msa_features=False,
        )

        dataset.warm_rna_bpp_cache(indices=[0])
        payload = pickle.dumps(dataset)
        restored = pickle.loads(payload)

        assert restored._rna_bpp_warmer is None
        assert len(restored) == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_prune_stale_eternafold_cache_locks_logs_summary(caplog) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"eternafold_stale_lock_cleanup_{uuid.uuid4().hex}"
    cache_dir = tmp_path / "eternafold_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        old_time = time.time() - 3600.0
        for name in ("A.pt.lock", "B.pt.lock", "C.pt.lock"):
            lock_path = cache_dir / name
            lock_path.write_text("stale", encoding="utf-8")
            os.utime(lock_path, (old_time, old_time))

        with caplog.at_level(logging.WARNING):
            removed = prune_stale_eternafold_cache_locks(cache_dir, stale_after_seconds=1800.0)

        assert removed == 3
        warnings = [record for record in caplog.records if "stale EternaFold cache lock" in record.getMessage()]
        assert len(warnings) == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_eternafold_cache_warmer_reports_progress_from_existing_cache(monkeypatch, caplog) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"eternafold_progress_{uuid.uuid4().hex}"
    cache_dir = tmp_path / "eternafold_cache"
    binary_path = tmp_path / "contrafold.exe"
    parameters_path = tmp_path / "EternaFoldParams.v1"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        binary_path.write_text("", encoding="utf-8")
        parameters_path.write_text("", encoding="utf-8")

        cached_residue_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        cached_path = eternafold_cache_path(
            target_id="T0",
            sequence="ACG",
            max_span=256,
            cutoff=1e-4,
            binary_path=binary_path.resolve(),
            parameters_path=parameters_path.resolve(),
            cache_dir=cache_dir.resolve(),
        )
        torch.save({"rna_bpp_banded": torch.zeros((3, 256), dtype=torch.float32)}, cached_path)

        def _fake_load_or_compute(*_args, **_kwargs):
            return torch.zeros((3, 256), dtype=torch.float32)

        monkeypatch.setattr(full_template_module, "load_or_compute_eternafold_bpp_banded", _fake_load_or_compute)
        monkeypatch.setattr("src.data.eternafold_bpp.load_or_compute_eternafold_bpp_banded", _fake_load_or_compute)

        with caplog.at_level(logging.INFO):
            warmer = EternaFoldCacheWarmer(
                max_workers=1,
                max_span=256,
                cutoff=1e-4,
                binary_path=binary_path,
                parameters_path=parameters_path,
                cache_dir=cache_dir,
            )
            warmer.submit_many(
                [
                    ("T0", cached_residue_idx),
                    ("T1", torch.tensor([0, 1, 2], dtype=torch.long)),
                ]
            )
            warmer.shutdown(wait=True)

        progress_logs = [record.getMessage() for record in caplog.records if "EternaFold cache progress:" in record.getMessage()]
        assert any("ready=1/2" in message for message in progress_logs)
        assert any("ready=2/2" in message for message in progress_logs)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_datamodule_limits_setup_bpp_warmup(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_setup_warmup_limit_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    binary_path = tmp_path / "contrafold.exe"
    parameters_path = tmp_path / "EternaFoldParams.v1"
    seen: dict[str, object] = {}

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for target_idx in range(4):
                writer.writerow(
                    {
                        "ID": f"T{target_idx}_1",
                        "resname": "A",
                        "resid": 1,
                        "x_1": 0.0,
                        "y_1": 0.0,
                        "z_1": 0.0,
                        "chain": "A",
                        "copy": 0,
                    }
                )

        with sequences_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_id",
                    "sequence",
                    "temporal_cutoff",
                    "description",
                    "stoichiometry",
                    "all_sequences",
                    "ligand_ids",
                    "ligand_SMILES",
                ],
            )
            writer.writeheader()
            for target_idx in range(4):
                writer.writerow(
                    {
                        "target_id": f"T{target_idx}",
                        "sequence": "A",
                        "temporal_cutoff": "2025-01-01",
                        "description": f"T{target_idx}",
                        "stoichiometry": "A:1",
                        "all_sequences": f">T{target_idx}_1|Chain A[auth A]|RNA|\nA\n",
                        "ligand_ids": "",
                        "ligand_SMILES": "",
                    }
                )

        binary_path.write_text("", encoding="utf-8")
        parameters_path.write_text("", encoding="utf-8")

        def _record_warmup(self, indices=None):
            seen["indices"] = list(indices) if indices is not None else None

        monkeypatch.setattr(full_template_module.RNAFullTemplateDataset, "warm_rna_bpp_cache", _record_warmup)

        dm = C147AFullTemplateDataModule(
            data_dir=str(tmp_path),
            labels_file="train_labels.csv",
            sequences_file="train_sequences.csv",
            batch_size=1,
            num_workers=0,
            train_fraction=0.5,
            val_fraction=0.25,
            max_residues_per_target=16,
            max_targets=None,
            use_template_coords=False,
            use_rna_msa_features=False,
            use_rna_bpp_features=True,
            rna_bpp_binary_path=str(binary_path),
            rna_bpp_parameters_path=str(parameters_path),
            rna_bpp_cache_dir="eternafold_cache",
            seed=42,
        )

        dm.prepare_data()
        dm.setup()

        assert "indices" in seen
        assert isinstance(seen["indices"], list)
        assert len(seen["indices"]) == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_length_precompute_uses_diversity_fallback_to_fill_five_templates() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_diverse_fill_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"
    payload_path = tmp_path / "full_length_template_coords.pt"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for target_id, offset in (("T0", 0.0), ("T1", 1.0)):
                for pos, resname in enumerate("ACG", start=1):
                    writer.writerow(
                        {
                            "ID": f"{target_id}_{pos}",
                            "resname": resname,
                            "resid": pos,
                            "x_1": float(pos) + offset,
                            "y_1": 0.0,
                            "z_1": 0.0,
                            "chain": "A",
                            "copy": 0,
                        }
                    )
        with sequences_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_id",
                    "sequence",
                    "temporal_cutoff",
                    "description",
                    "stoichiometry",
                    "all_sequences",
                    "ligand_ids",
                    "ligand_SMILES",
                ],
            )
            writer.writeheader()
            for target_id in ("T0", "T1"):
                writer.writerow(
                    {
                        "target_id": target_id,
                        "sequence": "ACG",
                        "temporal_cutoff": "2025-01-01",
                        "description": target_id,
                        "stoichiometry": "A:1",
                        "all_sequences": f">{target_id}_1|Chain A[auth A]|RNA|\nACG\n",
                        "ligand_ids": "",
                        "ligand_SMILES": "",
                    }
                )

        payload = precompute_full_length_template_coords(
            labels_path=labels_path,
            sequences_path=sequences_path,
            output_path=payload_path,
            top_k_store=5,
            max_residues_per_target=16,
            num_threads=0,
            min_percent_identity=40.0,
            min_similarity=0.0,
            length_ratio_tolerance=0.3,
        )

        assert int(payload["template_topk_valid"]["T0"].sum().item()) == 5
        assert any(str(source).startswith("oracle:") for source in payload["template_topk_sources"]["T0"][1:])
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_full_template_dataset_materializes_examples_lazily() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"full_template_lazy_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "train_labels.csv"
    sequences_path = tmp_path / "train_sequences.csv"

    try:
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "resname", "resid", "x_1", "y_1", "z_1", "chain", "copy"])
            writer.writeheader()
            for target_id, sequence in (("T0", "ACG"), ("T1", "GU")):
                for pos, resname in enumerate(sequence, start=1):
                    writer.writerow(
                        {
                            "ID": f"{target_id}_{pos}",
                            "resname": resname,
                            "resid": pos,
                            "x_1": float(pos),
                            "y_1": 0.0,
                            "z_1": 0.0,
                            "chain": "A",
                            "copy": 0,
                        }
                    )

        with sequences_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "target_id",
                    "sequence",
                    "temporal_cutoff",
                    "description",
                    "stoichiometry",
                    "all_sequences",
                    "ligand_ids",
                    "ligand_SMILES",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "target_id": "T0",
                    "sequence": "ACG",
                    "temporal_cutoff": "2025-01-01",
                    "description": "target 0",
                    "stoichiometry": "A:1",
                    "all_sequences": ">T0_1|Chain A[auth A]|RNA|\nACG\n",
                    "ligand_ids": "",
                    "ligand_SMILES": "",
                }
            )
            writer.writerow(
                {
                    "target_id": "T1",
                    "sequence": "GU",
                    "temporal_cutoff": "2025-01-01",
                    "description": "target 1",
                    "stoichiometry": "A:1",
                    "all_sequences": ">T1_1|Chain A[auth A]|RNA|\nGU\n",
                    "ligand_ids": "",
                    "ligand_SMILES": "",
                }
            )

        dataset = RNAFullTemplateDataset(
            labels_path=labels_path,
            sequences_path=sequences_path,
            max_residues_per_target=16,
            max_targets=None,
            template_topk_count=2,
            use_rna_bpp_features=False,
            use_rna_msa_features=False,
        )

        assert len(dataset) == 2
        assert dataset._example_cache == {}

        example = dataset[1]

        assert example.target_id == "T1"
        assert len(dataset._example_cache) == 1
        assert 1 in dataset._example_cache
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
