"""kerneldex command-line interface.

Four subcommands:

    kerneldex capture   - run a user script with the compile hook installed
    kerneldex histogram - disassemble captured kernels and aggregate mnemonics
    kerneldex coverage  - run an external tool over every captured kernel
    kerneldex report    - render a human-readable markdown report

Each command operates on a "dex directory" that contains a ``kernels/``
subdirectory (produced by capture) and optionally a ``reports/`` subdirectory
(produced by histogram / coverage).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import hook as hook_mod
from . import coverage as coverage_mod
from . import histogram as histogram_mod
from . import report as report_mod


def _cmd_capture(args: argparse.Namespace) -> int:
    user_argv = list(args.user_argv)
    if user_argv and user_argv[0] == "--":
        user_argv = user_argv[1:]
    if not user_argv:
        print(
            "kerneldex capture: error: pass the user program after --, e.g.:\n"
            "    kerneldex capture --target gfx1250 --out ./dex -- python my.py",
            file=sys.stderr,
        )
        return 2
    args.user_argv = user_argv

    out = Path(args.out).resolve()
    kernels_dir = out / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env[hook_mod.KERNELDEX_TARGET] = args.target
    env[hook_mod.KERNELDEX_BACKEND] = args.backend
    if args.warp_size:
        env[hook_mod.KERNELDEX_WARP_SIZE] = str(args.warp_size)
    env[hook_mod.KERNELDEX_OUT] = str(kernels_dir)
    if args.trace:
        env[hook_mod.KERNELDEX_TRACE] = "1"

    # We inject ``-m kerneldex._preload`` between the interpreter and the
    # user's script. We deliberately do NOT validate the interpreter's name
    # (common variants like ``python3.12`` or a venv's wrapper script would
    # otherwise be rejected). If kerneldex is not importable under the chosen
    # interpreter, ``-m kerneldex._preload`` will fail with a clear Python
    # error, which is a better signal than a heuristic reject here.
    interpreter = user_argv[0]
    user_rest = user_argv[1:]
    wrapped = [interpreter, "-m", "kerneldex._preload", *user_rest]

    print(
        f"[kerneldex] capture: target={args.target} out={out} "
        f"cmd={' '.join(wrapped)}",
        file=sys.stderr,
    )
    proc = subprocess.run(wrapped, env=env)
    return proc.returncode


def _cmd_histogram(args: argparse.Namespace) -> int:
    result = histogram_mod.build_histogram(Path(args.dex_dir), objdump=args.objdump)
    print(
        f"[kerneldex] histogram: {result.n_kernels} kernels, "
        f"{result.unique_mnemonics} unique mnemonics, "
        f"{result.total_instructions} total instructions",
    )
    print(f"  {result.csv_path}")
    print(f"  {result.jsonl_path}")
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    tool = Path(args.tool)
    if not tool.exists() or not os.access(tool, os.X_OK):
        print(
            f"kerneldex coverage: error: tool not executable: {tool}",
            file=sys.stderr,
        )
        return 2
    csv_path = coverage_mod.run_coverage(
        Path(args.dex_dir),
        tool,
        extra_args=args.tool_arg or [],
        timeout=args.timeout,
    )
    print(f"[kerneldex] coverage written to {csv_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    path = report_mod.render(Path(args.dex_dir))
    print(f"[kerneldex] report written to {path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kerneldex",
        description="Catalog the GPU kernels a Triton workload emits for a "
                    "target ISA, with instruction histograms and optional "
                    "per-kernel coverage against an external tool.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # capture --------------------------------------------------------------
    pc = sub.add_parser(
        "capture",
        help="Run a user script with the compile hook installed.",
        description="Run a user script with the kerneldex compile hook "
                    "installed, persisting every captured kernel into <out>.",
    )
    pc.add_argument("--target", required=True,
                    help="Target GPU architecture, e.g. gfx1250.")
    pc.add_argument("--backend", default="hip",
                    help="Triton backend (default: hip).")
    pc.add_argument("--warp-size", type=int, default=None,
                    help="Override warp size (default: 32 for gfx10+, 64 "
                         "otherwise).")
    pc.add_argument("--out", required=True,
                    help="Output directory for the dex (kernels + reports).")
    pc.add_argument("--trace", action="store_true",
                    help="Enable verbose hook tracing (sets KERNELDEX_TRACE).")
    pc.add_argument("user_argv", nargs=argparse.REMAINDER,
                    help="The user program, preceded by ``--``. The first "
                         "token must be a python interpreter.")
    pc.set_defaults(func=_cmd_capture)

    # histogram ------------------------------------------------------------
    ph = sub.add_parser(
        "histogram",
        help="Disassemble captured kernels and aggregate mnemonic histograms.",
    )
    ph.add_argument("dex_dir",
                    help="dex directory previously populated by ``capture``.")
    ph.add_argument("--objdump",
                    help="Path to a specific llvm-objdump binary. Defaults to "
                         "$KERNELDEX_OBJDUMP, then PATH lookup.")
    ph.set_defaults(func=_cmd_histogram)

    # coverage -------------------------------------------------------------
    pco = sub.add_parser(
        "coverage",
        help="Run an external tool over every captured kernel.",
    )
    pco.add_argument("dex_dir",
                     help="dex directory previously populated by ``capture``.")
    pco.add_argument("--tool", required=True,
                     help="Path to an external executable that accepts a "
                          "``.hsaco`` as its first positional argument.")
    pco.add_argument("--tool-arg", action="append",
                     help="Additional argument to forward to the tool. May "
                          "be repeated.")
    pco.add_argument("--timeout", type=float, default=300.0,
                     help="Per-kernel timeout for the tool subprocess, in "
                          "seconds. Timed-out kernels are recorded as "
                          "'timeout:<seconds>' in coverage.csv. Default 300.")
    pco.set_defaults(func=_cmd_coverage)

    # report ---------------------------------------------------------------
    pr = sub.add_parser(
        "report",
        help="Render a markdown REPORT.md combining manifest + histogram + coverage.",
    )
    pr.add_argument("dex_dir",
                    help="dex directory previously populated by ``capture``.")
    pr.set_defaults(func=_cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
