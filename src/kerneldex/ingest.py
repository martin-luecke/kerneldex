"""Ingest an existing directory of code objects into a dex layout.

Use case: kerneldex was built around Triton-driven capture, but many
interesting corpora (AITER CK kernels, precompiled ROCm libraries, saved
JIT output from a previous run, ...) are already on disk as ``.hsaco`` or
``.co`` files with no live Triton pipeline to hook.

``ingest_corpus`` walks such a directory recursively, deduplicates by
content hash, and drops the survivors into a standard dex layout
(``<out>/kernels/<flat>_<hash>.<ext>``) with a ``manifest.jsonl`` header
row whose ``status == "import"`` marks the dex as an imported corpus
rather than a captured one.

After ingest, the rest of the kerneldex pipeline
(``histogram`` / ``coverage`` / ``report``) works unchanged.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import KERNEL_EXTS

__all__ = ["ingest_corpus", "IngestResult", "IngestError"]


class IngestError(RuntimeError):
    """Raised for precondition failures (destination non-empty, etc.)."""


@dataclass(frozen=True)
class IngestResult:
    dex_dir: Path
    kernels_dir: Path
    manifest_path: Path
    n_scanned: int
    n_imported: int
    n_duplicates: int
    total_bytes: int


def _flatten_relpath(relpath: Path) -> str:
    """Turn ``a/b/c.co`` into ``a__b__c``.

    Non-alnum characters other than ``.`` / ``-`` are collapsed to ``_`` so
    that the resulting filename is portable across filesystems.
    """
    stem_parts = list(relpath.parts[:-1]) + [relpath.stem]
    joined = "__".join(stem_parts)
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in joined)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_corpus(source: Path) -> list[Path]:
    """Return every file under ``source`` whose suffix is a kernel extension.

    Sorted by relative path so the ingest order is deterministic across runs.
    """
    out: list[Path] = []
    for p in source.rglob("*"):
        if p.is_file() and p.suffix in KERNEL_EXTS:
            out.append(p)
    out.sort(key=lambda p: p.relative_to(source).as_posix())
    return out


def ingest_corpus(
    source: Path,
    out: Path,
    *,
    mode: str = "symlink",
    target: str | None = None,
    force: bool = False,
) -> IngestResult:
    """Ingest every code object under ``source`` into a dex at ``out``.

    :param source: Directory containing ``.hsaco`` or ``.co`` files (may be
        deeply nested).
    :param out: Top-level dex directory to create / populate.
    :param mode: ``"symlink"`` (default, fast, space-free) or ``"copy"``
        (self-contained).
    :param target: Optional architecture string (e.g. ``"gfx950"``) recorded
        in the manifest header and surfaced by ``kerneldex report``.
    :param force: If ``True``, wipe any existing ``<out>/kernels/`` contents
        before ingesting. Default is to refuse if the dex is already
        populated, to avoid silent half-overwrites.
    :raises IngestError: on any precondition failure. Consistent with the
        rest of kerneldex, this function never silently degrades.
    """
    if mode not in {"symlink", "copy"}:
        raise IngestError(f"invalid mode {mode!r}; expected 'symlink' or 'copy'")

    source = Path(source).resolve()
    if not source.is_dir():
        raise IngestError(f"source not a directory: {source}")

    out = Path(out).resolve()
    kernels_dir = out / "kernels"
    manifest_path = kernels_dir / "manifest.jsonl"

    if kernels_dir.exists() and any(kernels_dir.iterdir()):
        if not force:
            raise IngestError(
                f"{kernels_dir} is not empty; pass force=True to overwrite."
            )
        # Wipe only the kernels/ subdir; leave reports/ intact so a prior
        # histogram/coverage run is still available until the user re-runs
        # them against the new corpus.
        for entry in kernels_dir.iterdir():
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                shutil.rmtree(entry)

    kernels_dir.mkdir(parents=True, exist_ok=True)

    files = _walk_corpus(source)
    if not files:
        raise IngestError(
            f"no {'/'.join(KERNEL_EXTS)} files found under {source}"
        )

    manifest_rows: list[dict] = [{
        "status": "import",
        "source": str(source),
        "target": target or "",
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }]

    seen_hashes: dict[str, str] = {}  # sha256 -> kernels/<name>
    n_imported = 0
    n_duplicates = 0
    total_bytes = 0

    for src_path in files:
        relpath = src_path.relative_to(source)
        size = src_path.stat().st_size
        digest = _sha256(src_path)

        if digest in seen_hashes:
            manifest_rows.append({
                "status": "duplicate",
                "original_relpath": relpath.as_posix(),
                "sha256": digest,
                "size": size,
                "canonical": seen_hashes[digest],
            })
            n_duplicates += 1
            continue

        flat = _flatten_relpath(relpath)
        target_name = f"{flat}_{digest[:12]}{src_path.suffix}"
        target_path = kernels_dir / target_name
        # Name collisions after flattening should be unique because of the
        # hash suffix; if one still collides with a distinct source we bail
        # loudly rather than silently overwrite.
        if target_path.exists():
            raise IngestError(
                f"ingest name collision: {target_path} already exists "
                f"(from {relpath})"
            )

        if mode == "symlink":
            target_path.symlink_to(src_path)
        else:
            shutil.copy2(src_path, target_path)

        seen_hashes[digest] = f"kernels/{target_name}"
        manifest_rows.append({
            "status": "imported",
            "hsaco_file": f"kernels/{target_name}",
            "original_relpath": relpath.as_posix(),
            "sha256": digest,
            "size": size,
        })
        n_imported += 1
        total_bytes += size

    with manifest_path.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    return IngestResult(
        dex_dir=out,
        kernels_dir=kernels_dir,
        manifest_path=manifest_path,
        n_scanned=len(files),
        n_imported=n_imported,
        n_duplicates=n_duplicates,
        total_bytes=total_bytes,
    )
