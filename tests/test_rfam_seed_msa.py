from __future__ import annotations

import gzip
from pathlib import Path
import shutil
import uuid

import torch

from src.data.rfam_seed_msa import RfamSeedMSADatabase


def _write_seed_stockholm(path: Path) -> None:
    contents = "\n".join(
        [
            "# STOCKHOLM 1.0",
            "#=GF AC RF00001",
            "seqA ACGU",
            "seqB ACGC",
            "seqC AGGU",
            "//",
            "",
        ]
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(contents)


def test_rfam_seed_query_results_are_cached_across_runs() -> None:
    tmp_path = Path(".pytest_local_tmp") / f"rfam_cache_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    seed_path = tmp_path / "Rfam.seed.gz"
    seed_cache_path = tmp_path / "Rfam.seed.cache.pt"
    query_cache_path = tmp_path / "Rfam.seed.query_cache.pt"

    try:
        _write_seed_stockholm(seed_path)

        query_tokens = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        kwargs = {
            "max_sequences_per_family": 8,
            "kmer_size": 2,
            "prefilter_limit": 8,
            "min_similarity": 0.0,
            "min_percent_identity": 0.0,
            "length_ratio_tolerance": 1.0,
        }

        db = RfamSeedMSADatabase.from_files(
            seed_path=seed_path,
            cache_path=seed_cache_path,
            query_cache_path=query_cache_path,
            **kwargs,
        )
        db.begin_progress(total_queries=1)
        expected = db.build_msa_tensors(query_tokens=query_tokens, max_rows=3)
        db.finish_progress()

        seed_payload = torch.load(seed_cache_path, map_location="cpu", weights_only=False)
        query_payload = torch.load(query_cache_path, map_location="cpu", weights_only=False)
        assert "records" in seed_payload
        assert "query_cache" not in seed_payload
        assert "query_cache" in query_payload
        assert len(query_payload["query_cache"]) == 1

        db_cached = RfamSeedMSADatabase.from_files(
            seed_path=seed_path,
            cache_path=seed_cache_path,
            query_cache_path=query_cache_path,
            **kwargs,
        )

        class _FailingAligner:
            def align(self, *_args, **_kwargs):
                raise AssertionError("Alignment should not run for a cached MSA query.")

        db_cached.aligner = _FailingAligner()
        cached = db_cached.build_msa_tensors(query_tokens=query_tokens, max_rows=3)

        for actual, reference in zip(cached, expected):
            assert torch.equal(actual, reference)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
