from __future__ import annotations

import pytest

from src.train import _call_trainer_method


def test_call_trainer_method_unmasks_combined_loader_teardown_error() -> None:
    root = ValueError("root cause")

    def _boom(**_kwargs):
        exc = RuntimeError("Please call `iter(combined_loader)` first.")
        raise exc from root

    with pytest.raises(ValueError, match="root cause"):
        _call_trainer_method(_boom)
