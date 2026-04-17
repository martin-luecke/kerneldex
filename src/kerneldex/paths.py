"""Shared layout conventions for a kerneldex "dex" directory.

A dex directory always looks like::

    <dex>/
        kernels/
            <symbol>_<hash>.hsaco
            manifest.jsonl
        reports/              # created lazily by histogram / coverage / report

This module centralizes the resolution so every subcommand agrees on the
shape. Subcommands must always receive the top-level ``<dex>`` path, not the
``kernels/`` subdirectory.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_dex"]


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
