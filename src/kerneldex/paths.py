"""Shared layout conventions for a kerneldex "dex" directory.

A dex directory always looks like::

    <dex>/
        kernels/
            <symbol>_<hash>.hsaco   # from `capture`
            <flat>_<hash>.co        # or from `import`, for AMDGPU CK .co files
            manifest.jsonl
        reports/                    # created lazily by histogram / coverage / report

This module centralizes the resolution so every subcommand agrees on the
shape. Subcommands must always receive the top-level ``<dex>`` path, not the
``kernels/`` subdirectory.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_dex", "find_kernels", "KERNEL_EXTS"]

# Accepted code-object extensions. ``.hsaco`` is the Triton / ROCm default;
# ``.co`` is what AMDGPU CK kernels (e.g. AITER) ship as. Both are HSA code
# objects and llvm-objdump handles them identically.
KERNEL_EXTS: tuple[str, ...] = (".hsaco", ".co")


def resolve_dex(dex_dir: Path) -> tuple[Path, Path]:
    """Return ``(kernels_dir, reports_dir)`` for a top-level dex directory.

    :raises FileNotFoundError: if ``dex_dir/kernels/`` does not exist.
    """
    dex_dir = Path(dex_dir)
    kernels_dir = dex_dir / "kernels"
    if not kernels_dir.is_dir():
        raise FileNotFoundError(
            f"{kernels_dir} does not exist; pass the top-level dex directory "
            f"(the one that contains kernels/), not the kernels subdirectory."
        )
    return kernels_dir, dex_dir / "reports"


def find_kernels(kernels_dir: Path) -> list[Path]:
    """Return sorted list of code-object files in ``kernels_dir``.

    Matches any of the extensions in :data:`KERNEL_EXTS`. The return order is
    deterministic so that downstream aggregate outputs are diffable.
    """
    files: list[Path] = []
    for ext in KERNEL_EXTS:
        files.extend(kernels_dir.glob(f"*{ext}"))
    return sorted(files)
