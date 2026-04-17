"""Pluggable translator-coverage runner.

kerneldex itself has no opinion on what "coverage" means - it just invokes an
external binary (a "raiser", or in general anything that accepts an ``.hsaco``
and tells you whether it can process it) once per captured kernel and parses
the result.

Protocol the external tool must satisfy:

* Invoked as ``<binary> <hsaco_path> [extra args...]`` (extras are forwarded
  through the CLI with ``--raiser-arg``).
* Exit 0 means every kernel in the code object was handled successfully.
* Non-zero exit means at least one kernel failed. kerneldex captures the
  return code and the last line of stderr (a human-readable reason).

For richer per-kernel information, the tool may optionally emit lines in
one of these shapes on stdout:

  - ``OK <kernel-name> [(lifted/total)]``
  - ``FAIL <kernel-name> ... -> <mnemonic> [<format>]``

If the tool prints a line of the shape
``... Unsupported instruction: <mnemonic> ... [format=<format>]`` on stderr
before exiting non-zero, kerneldex will extract the ``<mnemonic>`` /
``<format>`` pair into the crash row so the blocker is still visible even
when no per-kernel OK/FAIL line was emitted.

Any tool that does not emit these formats will still work - kerneldex will
just record "unknown" per-kernel outcomes in that case. Subprocess isolation
means the runner survives SIGABRT / fatal errors in the external tool.
"""
from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .paths import resolve_dex

# A kernel-name token: must start with a letter or underscore and contain only
# identifier-like characters. This keeps us from matching summary prose such
# as "OK and 3 Code Objects Had No Match" as a kernel-name row.
_NAME = r"[A-Za-z_][\w.]*"

# OK lines: mnemonically
#     OK <name> [(lifted/total)]
# and *only* that, to end-of-line. The strict anchor is what makes the protocol
# robust against arbitrary logging from the external tool.
_OK_LINE = re.compile(
    r"^\s*OK\s+(?P<name>" + _NAME + r")"
    r"(?:\s+\((?P<lifted>\d+)/(?P<total>\d+)\))?"
    r"\s*$"
)

# FAIL lines carry a trailing ``-> <mnemonic> [<format>]`` and may have
# arbitrary prose in between; anchor on the explicit arrow + bracket.
_FAIL_LINE = re.compile(
    r"^\s*FAIL\s+(?P<name>" + _NAME + r").*?->\s+(?P<mnem>\S+)\s+\[(?P<fmt>[^\]]+)\]\s*$"
)

_UNSUPPORTED_LINE = re.compile(
    r"Unsupported instruction:\s+(?P<mnem>\S+).*?\[format=(?P<fmt>[^\]]+)\]"
)


@dataclass(frozen=True)
class CoverageRow:
    file: str
    kernel_name: str
    outcome: str  # "ok" | "fail" | "crash:<rc>" | "unknown"
    lifted: str
    total: str
    mnemonic: str
    format: str
    note: str


def _parse_stdout(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        m = _FAIL_LINE.match(line)
        if m:
            rows.append({
                "kernel_name": m.group("name"),
                "outcome": "fail",
                "lifted": "",
                "total": "",
                "mnemonic": m.group("mnem"),
                "format": m.group("fmt"),
            })
            continue
        m = _OK_LINE.match(line)
        if m:
            rows.append({
                "kernel_name": m.group("name"),
                "outcome": "ok",
                "lifted": m.group("lifted") or "",
                "total": m.group("total") or "",
                "mnemonic": "",
                "format": "",
            })
    return rows


def _first_unsupported(text: str) -> tuple[str, str] | None:
    for line in text.splitlines():
        u = _UNSUPPORTED_LINE.search(line)
        if u:
            return u.group("mnem"), u.group("fmt")
    return None


def run_coverage(
    dex_dir: Path,
    raiser: Path,
    extra_args: list[str] | None = None,
    timeout: float | None = 300.0,
) -> Path:
    """Run ``raiser`` on every captured ``.hsaco`` under ``dex_dir``.

    :param timeout: Per-kernel wall-clock limit in seconds. ``None`` disables
        the timeout. Timed-out kernels are recorded as ``timeout:<seconds>``
        rather than killing the whole run.

    Writes ``<dex_dir>/reports/coverage.csv`` and returns its path.
    """
    kernels_dir, reports_dir = resolve_dex(dex_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    extras = list(extra_args or [])
    hsaco_files = sorted(kernels_dir.glob("*.hsaco"))
    if not hsaco_files:
        raise FileNotFoundError(f"no .hsaco files in {kernels_dir}")

    rows: list[CoverageRow] = []
    for hf in hsaco_files:
        try:
            proc = subprocess.run(
                [str(raiser), str(hf), *extras],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            rows.append(CoverageRow(
                file=hf.name,
                kernel_name="<timeout>",
                outcome=f"timeout:{timeout}",
                lifted="",
                total="",
                mnemonic="",
                format="",
                note=(exc.stderr or "")[-400:] if exc.stderr else "",
            ))
            continue

        parsed = _parse_stdout(proc.stdout)
        if proc.returncode != 0 and not parsed:
            fallback = _first_unsupported(proc.stderr) or ("", "")
            note = (proc.stderr.strip().splitlines() or ["unknown"])[-1]
            rows.append(CoverageRow(
                file=hf.name,
                kernel_name="<process_crashed>",
                outcome=f"crash:{proc.returncode}",
                lifted="",
                total="",
                mnemonic=fallback[0],
                format=fallback[1],
                note=note[:400],
            ))
        elif not parsed:
            rows.append(CoverageRow(
                file=hf.name,
                kernel_name="<no_parseable_output>",
                outcome="unknown",
                lifted="",
                total="",
                mnemonic="",
                format="",
                note="tool exited 0 but emitted no OK/FAIL lines",
            ))
        else:
            for p in parsed:
                rows.append(CoverageRow(
                    file=hf.name,
                    kernel_name=p["kernel_name"],
                    outcome=p["outcome"],
                    lifted=p["lifted"],
                    total=p["total"],
                    mnemonic=p["mnemonic"],
                    format=p["format"],
                    note="",
                ))

    csv_path = reports_dir / "coverage.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "kernel_name", "outcome", "lifted", "total",
            "mnemonic", "format", "note",
        ])
        for r in rows:
            w.writerow([
                r.file, r.kernel_name, r.outcome, r.lifted, r.total,
                r.mnemonic, r.format, r.note,
            ])
    return csv_path
