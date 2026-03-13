from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import torch

import src.data.precomputed_rna_msa as msa_module


def test_build_precomputed_rna_msa_tensors_reuses_disk_cache(monkeypatch) -> None:
    tmp_path = Path(".pytest_local_tmp") / f"precomputed_rna_msa_cache_{uuid.uuid4().hex}"
    msa_path = tmp_path / "T0.MSA.fasta"
    cache_dir = tmp_path / "msa_tensor_cache"
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        msa_path.write_text(
            ">query\n"
            "ACG\n"
            ">homolog_1\n"
            "A-G\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("RNA_MSA_TENSOR_CACHE_DIR", str(cache_dir))

        query_tokens = torch.tensor([0, 1, 2], dtype=torch.long)
        first = msa_module.build_precomputed_rna_msa_tensors(msa_path=msa_path, query_tokens=query_tokens, max_rows=2)

        def _fail_parse(*_args, **_kwargs):
            raise AssertionError("Expected on-disk RNA MSA tensor cache to be reused.")

        monkeypatch.setattr(msa_module, "_parse_fasta_records", _fail_parse)
        second = msa_module.build_precomputed_rna_msa_tensors(msa_path=msa_path, query_tokens=query_tokens, max_rows=2)

        for first_tensor, second_tensor in zip(first, second):
            assert torch.equal(first_tensor, second_tensor)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
