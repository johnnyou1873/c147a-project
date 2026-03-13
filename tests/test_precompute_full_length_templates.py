from __future__ import annotations

import numpy as np

from src.data.precompute_full_length_templates import _repeat_fill_topk_templates


def test_repeat_fill_topk_templates_promotes_partial_payload_to_full_topk() -> None:
    topk_coords = np.zeros((5, 3, 3), dtype=np.float32)
    topk_coords[0] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    topk_coords[1] = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    topk_valid = np.array([True, True, False, False, False], dtype=np.bool_)
    topk_identity = np.array([88.0, 72.0, 0.0, 0.0, 0.0], dtype=np.float32)
    topk_similarity = np.array([0.9, 0.7, 0.0, 0.0, 0.0], dtype=np.float32)
    topk_residue_idx = np.full((5, 3), 4, dtype=np.int64)
    topk_residue_idx[0] = np.array([0, 1, 2], dtype=np.int64)
    topk_residue_idx[1] = np.array([0, 1, 2], dtype=np.int64)
    topk_sources = ["best", "other", "", "", ""]

    filled = _repeat_fill_topk_templates(
        topk_coords=topk_coords,
        topk_valid=topk_valid,
        topk_identity=topk_identity,
        topk_similarity=topk_similarity,
        topk_residue_idx=topk_residue_idx,
        topk_sources=topk_sources,
    )

    assert filled is True
    assert bool(topk_valid.all())
    for idx in range(2, 5):
        assert np.allclose(topk_coords[idx], topk_coords[0])
        assert float(topk_identity[idx]) == float(topk_identity[0])
        assert float(topk_similarity[idx]) == float(topk_similarity[0])
        assert np.array_equal(topk_residue_idx[idx], topk_residue_idx[0])
        assert topk_sources[idx] == topk_sources[0]
