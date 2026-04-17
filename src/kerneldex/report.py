"""Render a human-readable markdown report from the captured artifacts.

The report is intentionally minimal and deterministic: it summarizes the
corpus, embeds the top-N mnemonics inline, and (if coverage.csv exists)
reports per-kernel translator-coverage results plus a prioritized missing-
handler worklist. It links to the underlying CSV / JSONL files for full
detail.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .paths import resolve_dex

_TOP_N = 25


def _read_histogram(csv_path: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((
                row["mnemonic"],
                int(row["total_count"]),
                int(row["num_kernels"]),
            ))
    return rows


def _read_coverage(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _inventory_table(manifest_rows: list[dict]) -> list[list[str]]:
    seen: dict[str, dict] = {}
    for r in manifest_rows:
        if r.get("status") != "ok":
            continue
        key = r.get("key")
        if key and key not in seen:
            seen[key] = r
    out = [["kernel symbol", "source module", "hash", "hsaco"]]
    for key, r in sorted(seen.items(), key=lambda kv: (kv[1].get("kernel", ""), kv[0])):
        hsaco = Path(r.get("hsaco_file") or "").name
        out.append([
            r.get("kernel", "?"),
            r.get("module", "?"),
            key,
            hsaco,
        ])
    return out


def _worklist(coverage_rows: list[dict[str, str]]) -> list[tuple[str, str, list[str]]]:
    by_key: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in coverage_rows:
        if row["outcome"] != "fail":
            continue
        if not row["mnemonic"]:
            continue
        by_key[(row["mnemonic"], row["format"])].add(row["kernel_name"])
    out = [
        (mnem, fmt, sorted(names))
        for (mnem, fmt), names in by_key.items()
    ]
    out.sort(key=lambda t: (-len(t[2]), t[0]))
    return out


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def render(dex_dir: Path) -> Path:
    dex_dir = Path(dex_dir)
    kernels_dir, reports_dir = resolve_dex(dex_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = dex_dir / "REPORT.md"

    manifest = _read_manifest(kernels_dir / "manifest.jsonl")

    installs = [r for r in manifest if r.get("status") == "install"]
    target = installs[-1]["target"] if installs else "(unknown)"
    backend = installs[-1].get("backend", "(unknown)") if installs else "(unknown)"

    n_ok = sum(1 for r in manifest if r.get("status") == "ok")
    n_unique_ok = len({r["key"] for r in manifest if r.get("status") == "ok" and r.get("key")})
    n_fail = sum(1 for r in manifest if r.get("status") == "fail")
    n_ok_no_hsaco = sum(1 for r in manifest if r.get("status") == "ok_no_hsaco")

    parts: list[str] = []
    parts.append(f"# kerneldex report: {target}\n")
    parts.append(
        f"- target: `{target}` (backend `{backend}`)\n"
        f"- compile attempts logged: {n_ok + n_fail + n_ok_no_hsaco}\n"
        f"- unique kernels captured: {n_unique_ok}\n"
        f"- compile failures recorded: {n_fail}\n"
    )
    if n_ok_no_hsaco:
        parts.append(
            f"- compiles missing an hsaco blob (could not be cataloged): "
            f"{n_ok_no_hsaco}\n"
        )

    parts.append("\n## Kernel inventory\n\n")
    parts.append(_md_table(_inventory_table(manifest)))
    parts.append(
        "\n\nFull manifest: [kernels/manifest.jsonl](kernels/manifest.jsonl).\n"
    )

    hist_csv = reports_dir / "mnemonic_histogram.csv"
    if hist_csv.exists():
        hist = _read_histogram(hist_csv)
        top = hist[:_TOP_N]
        parts.append(f"\n## Mnemonic histogram (top {_TOP_N})\n\n")
        parts.append(_md_table(
            [["mnemonic", "total", "num_kernels"]]
            + [[m, str(n), str(k)] for (m, n, k) in top]
        ))
        parts.append(
            f"\n\n- {len(hist)} unique mnemonics total\n"
            f"- full CSV: [reports/mnemonic_histogram.csv](reports/mnemonic_histogram.csv)\n"
            f"- per-kernel counts: [reports/per_kernel_mnemonics.jsonl](reports/per_kernel_mnemonics.jsonl)\n"
        )

    cov_csv = reports_dir / "coverage.csv"
    if cov_csv.exists():
        cov_rows = _read_coverage(cov_csv)
        parts.append("\n## Translator coverage\n\n")
        parts.append(_md_table(
            [["file", "kernel", "outcome", "blocker"]]
            + [
                [
                    r["file"],
                    r["kernel_name"],
                    r["outcome"],
                    (
                        f"{r['mnemonic']} [{r['format']}]"
                        if r["mnemonic"]
                        else r["note"][:80]
                    ),
                ]
                for r in cov_rows
            ]
        ))
        parts.append(
            "\n\nFull CSV: [reports/coverage.csv](reports/coverage.csv).\n"
        )

        wl = _worklist(cov_rows)
        if wl:
            parts.append(
                "\n## Prioritized missing-handler worklist\n\n"
                "Ordered by the number of captured kernels a single handler "
                "would unblock (greedy; does not account for second-order "
                "blockers surfaced after each fix).\n\n"
            )
            parts.append(_md_table(
                [["mnemonic", "format", "kernels unblocked", "kernels"]]
                + [
                    [mnem, fmt, str(len(names)), ", ".join(names)]
                    for (mnem, fmt, names) in wl
                ]
            ))
            parts.append(
                "\n\nNote: the external coverage tool typically stops at the "
                "first unsupported instruction per kernel, so deeper blockers "
                "will surface only after each handler lands.\n"
            )

    out_path.write_text("\n".join(parts) + "\n")
    return out_path
