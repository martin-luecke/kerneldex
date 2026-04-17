"""Wrapper around ``llvm-objdump`` for mnemonic extraction.

We deliberately shell out to ``llvm-objdump`` (rather than binding to LLVM
directly) so kerneldex stays a pure-Python package with no native build step.
The cost is one subprocess per kernel; the benefit is that the user chooses
which LLVM to use - importantly, some targets (e.g. gfx1250) require a
recent LLVM.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_OBJDUMP_CANDIDATES = (
    os.environ.get("KERNELDEX_OBJDUMP") or "",
    "llvm-objdump",
)

# Mnemonics follow optional "<offset>: <bytes>" prefix produced by
# `llvm-objdump -d`. The mnemonic is the first lowercase token on the line.
_MNEMONIC_RE = re.compile(
    r"^\s*(?:[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2}\s+)+\s*)?([a-z_][a-z0-9_]+)\b"
)


class ObjdumpError(RuntimeError):
    """Raised when ``llvm-objdump`` cannot disassemble a kernel."""


@dataclass(frozen=True)
class Disassembly:
    """Result of disassembling one HSA code object."""

    path: Path
    lines: list[str]

    def mnemonics(self) -> list[str]:
        out: list[str] = []
        for line in self.lines:
            m = _MNEMONIC_RE.match(line)
            if not m:
                continue
            token = m.group(1)
            # Skip objdump headers (e.g. "file format ...", section names).
            if token in {"file", "section", "disassembly", "of"}:
                continue
            out.append(token)
        return out


def resolve_objdump(explicit: str | None = None) -> str:
    """Return a usable ``llvm-objdump`` binary path.

    Resolution order:

    1. The ``explicit`` argument if provided.
    2. ``$KERNELDEX_OBJDUMP`` if set.
    3. The first ``llvm-objdump`` on ``$PATH``.

    :raises ObjdumpError: if nothing usable is found.
    """
    candidates = (
        explicit,
        os.environ.get("KERNELDEX_OBJDUMP") or None,
        "llvm-objdump",
    )
    for c in candidates:
        if not c:
            continue
        if os.path.isabs(c) and os.access(c, os.X_OK):
            return c
        found = shutil.which(c)
        if found:
            return found
    raise ObjdumpError(
        "llvm-objdump not found. Install it, put it on $PATH, or set "
        "$KERNELDEX_OBJDUMP to a specific binary. Note that some targets "
        "(e.g. gfx1250) require a recent LLVM; the one that ships with "
        "your Linux distro is likely too old."
    )


def disassemble(hsaco_path: Path, objdump: str | None = None) -> Disassembly:
    """Disassemble one ``.hsaco`` and return the raw text lines.

    :raises ObjdumpError: if objdump fails for any reason (e.g. unsupported
        target in an old LLVM build). We surface the stderr verbatim so the
        user can act on it.
    """
    binary = resolve_objdump(objdump)
    proc = subprocess.run(
        [binary, "-d", str(hsaco_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ObjdumpError(
            f"llvm-objdump exited {proc.returncode} on {hsaco_path}:\n"
            f"{proc.stderr.strip()}"
        )
    return Disassembly(path=hsaco_path, lines=proc.stdout.splitlines())
