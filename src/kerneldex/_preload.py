"""Bootstrap that installs :mod:`kerneldex.hook` and then hands control to the
user's Python program.

This module is only meant to be invoked by ``kerneldex capture`` via

    python -m kerneldex._preload <user_script.py> [args...]

It installs the hook (which reads its configuration from the environment
variables set by the CLI) and then uses :mod:`runpy` to execute the user's
script in-process. The user script sees exactly the ``sys.argv`` / ``__name__``
it would have gotten from ``python user_script.py``.
"""
from __future__ import annotations

import os
import runpy
import sys

from . import hook


def _usage() -> int:
    print(
        "usage: python -m kerneldex._preload <script.py> [args...]\n"
        "       python -m kerneldex._preload -m <module> [args...]",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    # argv[0] is this module's name; argv[1:] is what the user asked to run.
    if len(sys.argv) < 2:
        return _usage()

    hook.install()

    rest = sys.argv[1:]
    if rest[0] == "-m":
        if len(rest) < 2:
            return _usage()
        module = rest[1]
        sys.argv = [module] + rest[2:]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0

    script = rest[0]
    if not os.path.isfile(script):
        print(f"[kerneldex] script not found: {script}", file=sys.stderr)
        return 2
    sys.argv = [script] + rest[1:]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
