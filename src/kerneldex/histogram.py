"""Aggregate mnemonics across a captured kernel corpus.

Input:  a dex directory whose ``kernels/`` subdirectory contains ``.hsaco``
or ``.co`` files (as produced by ``kerneldex capture`` or
``kerneldex import``).

Output:
  - ``<out>/reports/mnemonic_histogram.csv``
  - ``<out>/reports/per_kernel_mnemonics.jsonl``
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .disasm import disassemble
from .paths import find_kernels, resolve_dex


@dataclass(frozen=True)
class HistogramResult:
    n_kernels: int
    unique_mnemonics: int
    total_instructions: int
    csv_path: Path
    jsonl_path: Path


def build_histogram(dex_dir: Path, objdump: str | None = None) -> HistogramResult:
    kernels_dir, reports_dir = resolve_dex(Path(dex_dir))
    reports_dir.mkdir(parents=True, exist_ok=True)

    hsaco_files = find_kernels(kernels_dir)
    if not hsaco_files:
        raise FileNotFoundError(
            f"no .hsaco or .co files in {kernels_dir}"
        )

    global_counts: Counter[str] = Counter()
    kernels_per_mnemonic: defaultdict[str, set[str]] = defaultdict(set)
    per_kernel: dict[str, Counter[str]] = {}

    for hf in hsaco_files:
        d = disassemble(hf, objdump=objdump)
        mnems = d.mnemonics()
        counts = Counter(mnems)
        per_kernel[hf.name] = counts
        for m, n in counts.items():
            global_counts[m] += n
            kernels_per_mnemonic[m].add(hf.name)

    csv_path = reports_dir / "mnemonic_histogram.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mnemonic", "total_count", "num_kernels", "kernels"])
        for m, n in sorted(global_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([
                m,
                n,
                len(kernels_per_mnemonic[m]),
                ";".join(sorted(kernels_per_mnemonic[m])),
            ])

    jsonl_path = reports_dir / "per_kernel_mnemonics.jsonl"
    with jsonl_path.open("w") as f:
        for name, counts in sorted(per_kernel.items()):
            f.write(json.dumps({
                "kernel": name,
                "unique_mnemonics": len(counts),
                "total_instructions": sum(counts.values()),
                "mnemonics": dict(counts),
            }) + "\n")

    return HistogramResult(
        n_kernels=len(hsaco_files),
        unique_mnemonics=len(global_counts),
        total_instructions=sum(global_counts.values()),
        csv_path=csv_path,
        jsonl_path=jsonl_path,
    )
