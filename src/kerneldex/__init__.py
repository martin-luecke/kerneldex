"""kerneldex - catalog the GPU kernels a Triton workload emits for a target ISA.

Public API is deliberately narrow. Most users go through the ``kerneldex``
CLI; the ``hook`` module is exported for programmatic use (e.g. in notebooks
or test harnesses where launching a subprocess is undesirable).
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import hook  # re-export for programmatic use

__all__ = ["hook", "__version__"]
