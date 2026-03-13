"""
Archived legacy Lightning module entrypoint.

The active training path now uses `src.models.protenix_style_module`
with `ProtenixStyleLitModule`. The previous `C147ALitModule`
integration has been intentionally unhooked and is preserved in git
history for reference.
"""

# Archived on 2026-03-12.
# Active replacement:
#   src.models.protenix_style_module.ProtenixStyleLitModule


class C147ALitModule:  # pragma: no cover - archive stub
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "src.models.c147a_module.C147ALitModule is archived and no longer supported. "
            "Use src.models.protenix_style_module.ProtenixStyleLitModule instead."
        )
