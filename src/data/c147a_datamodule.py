"""
Archived legacy chunk-template datamodule.

The active training path now uses `src.data.c147a_full_template_datamodule`
exclusively. The previous `C147ADataModule` / `RNAIdentityDataset`
implementation has been intentionally unhooked and is preserved in git
history for reference.
"""

# Archived on 2026-03-12.
# Active replacement:
#   src.data.c147a_full_template_datamodule.C147AFullTemplateDataModule


class C147ADataModule:  # pragma: no cover - archive stub
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "src.data.c147a_datamodule.C147ADataModule is archived and no longer supported. "
            "Use src.data.c147a_full_template_datamodule.C147AFullTemplateDataModule instead."
        )


class RNAIdentityDataset:  # pragma: no cover - archive stub
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "src.data.c147a_datamodule.RNAIdentityDataset is archived and no longer supported."
        )
