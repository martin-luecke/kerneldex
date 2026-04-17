"""Compile hook that intercepts every ``triton.compile`` call and forces the
configured target ISA, persisting the resulting HSA code object (``.hsaco``)
plus metadata.

Design:

* The hook is installed by calling :func:`install`. It monkey-patches
  ``triton.compiler.compiler.compile`` and a couple of re-exports so that it
  catches every caller path Triton exposes.
* Each intercepted call runs the compile **twice**:

  1. Once against the configured ``target`` (the one the user asked us to
     catalog). If that succeeds, the ``.hsaco`` is written to ``out_dir``
     and a ``status=ok`` row is appended to ``manifest.jsonl``. If it fails,
     a ``status=fail`` row is appended, with the full traceback.
  2. Once against the **original** target the caller requested. That second
     result is what the hook actually returns, so the user's workload keeps
     running on the host device normally.

  This means kerneldex never alters what the user's program computes or
  which device it runs on - we just observe what Triton would have
  produced for ``target``.

* Configuration is taken from environment variables so a subprocess driver
  can set them before executing the user's script:

  - ``KERNELDEX_TARGET``     - required, e.g. ``"gfx1250"``
  - ``KERNELDEX_BACKEND``    - optional, default ``"hip"``
  - ``KERNELDEX_WARP_SIZE``  - optional; auto-detected if omitted
  - ``KERNELDEX_OUT``        - required; directory to populate
  - ``KERNELDEX_TRACE``      - optional; if truthy, emit verbose hook logs

This module deliberately has no fallback behaviour: if the target compile
fails, the failure is recorded verbatim; we do not silently skip the kernel
or rewrite it. See docs/design.md for the rationale.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import threading
import traceback
from typing import Any

__all__ = [
    "install",
    "restore",
    "is_installed",
    "compile_hook",
    "HookConfigError",
    "KERNELDEX_TARGET",
    "KERNELDEX_BACKEND",
    "KERNELDEX_OUT",
]


class HookConfigError(RuntimeError):
    """Raised when the hook is installed without the required configuration."""


# --- env-driven config -----------------------------------------------------

KERNELDEX_TARGET = "KERNELDEX_TARGET"
KERNELDEX_BACKEND = "KERNELDEX_BACKEND"
KERNELDEX_WARP_SIZE = "KERNELDEX_WARP_SIZE"
KERNELDEX_OUT = "KERNELDEX_OUT"
KERNELDEX_TRACE = "KERNELDEX_TRACE"


def _trace(msg: str) -> None:
    if os.environ.get(KERNELDEX_TRACE):
        print(f"[kerneldex] {msg}", file=sys.stderr, flush=True)


def _default_warp_size(arch: str) -> int:
    """Match what Triton's AMD backend does: gfx10+ is wave32, older is wave64.

    The suffix may contain a trailing letter (e.g. ``gfx90a``); we derive the
    major generation from all but the last two characters so that ``gfx90a``
    is treated as generation 9 (wave64) and ``gfx1250`` as generation 12
    (wave32).
    """
    if not arch.startswith("gfx"):
        return 32
    digits = arch[3:]
    try:
        major = int(digits[:-2]) if len(digits) >= 3 else int(digits)
    except ValueError:
        return 32
    return 32 if major >= 10 else 64


def _read_target_from_env():
    from triton.backends.compiler import GPUTarget

    arch = os.environ.get(KERNELDEX_TARGET)
    if not arch:
        raise HookConfigError(
            f"{KERNELDEX_TARGET} is not set; cannot install the hook. "
            "Run via the kerneldex CLI or set the env var manually."
        )
    backend = os.environ.get(KERNELDEX_BACKEND, "hip")
    warp_size_env = os.environ.get(KERNELDEX_WARP_SIZE)
    warp_size = int(warp_size_env) if warp_size_env else _default_warp_size(arch)
    return GPUTarget(backend=backend, arch=arch, warp_size=warp_size)


def _read_out_from_env() -> str:
    out = os.environ.get(KERNELDEX_OUT)
    if not out:
        raise HookConfigError(
            f"{KERNELDEX_OUT} is not set; cannot install the hook. "
            "Run via the kerneldex CLI or set the env var manually."
        )
    os.makedirs(out, exist_ok=True)
    return out


# --- hook state ------------------------------------------------------------
#
# ``_LOCK`` guards ``_ORIG_COMPILE``, ``_INSTALLED``, ``_SIG``, and
# ``_SEEN_KEYS``. The lock is held for the brief bookkeeping in
# :func:`install` / :func:`restore` / :func:`_record_success`; it is
# intentionally *not* held across a call to the real Triton compile, since
# that would serialize all Triton compilation.

_ORIG_COMPILE = None  # set in install()
_SIG: inspect.Signature | None = None  # cached for target-override
_LOCK = threading.Lock()
_SEEN_KEYS: set[str] = set()
_INSTALLED = False


# --- helpers ---------------------------------------------------------------

def _safe_repr(x: Any) -> str:
    try:
        return repr(x)
    except Exception:
        return f"<unrepr {type(x).__name__}>"


def _stringify_dict(d: Any) -> dict:
    if d is None:
        return {}
    if isinstance(d, dict):
        return {str(k): _safe_repr(v) for k, v in d.items()}
    out = {}
    for attr in dir(d):
        if attr.startswith("_"):
            continue
        try:
            out[attr] = _safe_repr(getattr(d, attr))
        except Exception:
            pass
    return out


def _src_identity(src: Any) -> dict:
    fn = getattr(src, "fn", None)
    return {
        "kernel": getattr(fn, "__name__", "?"),
        "module": getattr(fn, "__module__", "?"),
        "signature": _stringify_dict(getattr(src, "signature", None)),
        "constexprs": _stringify_dict(getattr(src, "constants", None)),
    }


def _hash_key(ident: dict) -> str:
    raw = json.dumps(ident, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.jsonl")


def _append_manifest(out_dir: str, row: dict) -> None:
    with _LOCK:
        with open(_manifest_path(out_dir), "a") as f:
            f.write(json.dumps(row) + "\n")


def _extract_hsaco(kernel_obj: Any) -> bytes | None:
    for attr in ("asm", "kernel"):
        asm = getattr(kernel_obj, attr, None)
        if isinstance(asm, dict) and "hsaco" in asm:
            blob = asm["hsaco"]
            if isinstance(blob, (bytes, bytearray)):
                return bytes(blob)
    return None


def _record_success(out_dir: str, src: Any, kernel_obj: Any) -> None:
    """Dedupe per kernel identity, write the blob, append an ``ok`` row.

    If the CompiledKernel has no ``hsaco`` blob (future API change, or an
    unexpected build where the binary is not attached), we emit a distinct
    ``ok_no_hsaco`` row rather than a silently broken ``ok`` row pointing at
    a file that doesn't exist. This preserves the "no fallback" principle.
    """
    ident = _src_identity(src)
    key = _hash_key(ident)
    with _LOCK:
        if key in _SEEN_KEYS:
            return
        _SEEN_KEYS.add(key)
    blob = _extract_hsaco(kernel_obj)
    if blob is None:
        _append_manifest(out_dir, {
            "status": "ok_no_hsaco",
            "key": key,
            **ident,
            "note": "compile succeeded but no 'hsaco' blob found on the "
                    "CompiledKernel; kerneldex cannot catalog this kernel.",
        })
        return
    hsaco_file = os.path.join(out_dir, f"{ident['kernel']}_{key}.hsaco")
    with open(hsaco_file, "wb") as f:
        f.write(blob)
    _append_manifest(out_dir, {
        "status": "ok",
        "key": key,
        **ident,
        "hsaco_file": hsaco_file,
        "hsaco_bytes": len(blob),
    })


def _record_failure(out_dir: str, src: Any, exc: BaseException) -> None:
    """Append a ``fail`` row. Intentionally NOT deduplicated - a caller that
    retries a failing kernel wants every attempt in the manifest."""
    ident = _src_identity(src)
    key = _hash_key(ident)
    _append_manifest(out_dir, {
        "status": "fail",
        "key": key,
        **ident,
        "exc_type": type(exc).__name__,
        "exc_msg": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    })


# --- the hook itself -------------------------------------------------------

def _override_target(args: tuple, kwargs: dict, new_target: Any) -> tuple[tuple, dict]:
    """Return an (args, kwargs) pair that re-invokes the underlying compile
    against ``new_target`` without losing any of the caller's other arguments.

    We use :func:`inspect.Signature.bind_partial` so that future additions to
    Triton's compile signature pass through untouched - kerneldex only
    overrides ``target`` (and clears ``options`` so the backend picks
    target-appropriate defaults).
    """
    assert _SIG is not None
    bound = _SIG.bind_partial(*args, **kwargs)
    bound.arguments["target"] = new_target
    if "options" in _SIG.parameters:
        bound.arguments["options"] = None
    return bound.args, bound.kwargs


def compile_hook(*args, **kwargs):
    """Replacement for ``triton.compiler.compiler.compile``.

    Compiles the kernel twice:

    1. Against the catalog target (``$KERNELDEX_TARGET``); records success or
       failure in ``manifest.jsonl``. Exceptions here are swallowed *except*
       for control-flow signals (:class:`KeyboardInterrupt`,
       :class:`SystemExit`), which propagate so the user can always interrupt
       a long capture.

    2. Against the caller's original target; the result is returned to the
       caller so the user's program keeps running as though kerneldex were
       not installed.

    We accept ``*args, **kwargs`` and route the target-override via the
    cached signature, so adding kwargs to Triton's compile does not break us.
    """
    assert _ORIG_COMPILE is not None, "hook used without install()"
    try:
        requested_target = _read_target_from_env()
        out_dir = _read_out_from_env()
    except HookConfigError:
        # Env was cleared after install(). Rare; behave as a transparent
        # passthrough so we don't break the user's program.
        return _ORIG_COMPILE(*args, **kwargs)

    src = args[0] if args else kwargs.get("src")
    fn_name = getattr(getattr(src, "fn", None), "__name__", "?")
    _trace(
        f"{fn_name} host_target={kwargs.get('target')} "
        f"-> catalog against {requested_target}"
    )

    try:
        t_args, t_kwargs = _override_target(args, kwargs, requested_target)
        kernel_obj = _ORIG_COMPILE(*t_args, **t_kwargs)
        _record_success(out_dir, src, kernel_obj)
    except (KeyboardInterrupt, SystemExit):
        # Never swallow user interrupts or explicit process exits.
        raise
    except BaseException as exc:  # noqa: BLE001 - we must see every failure
        _record_failure(out_dir, src, exc)
        print(
            f"[kerneldex] target compile FAILED for {fn_name}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    return _ORIG_COMPILE(*args, **kwargs)


# --- install / restore -----------------------------------------------------

def install() -> None:
    """Install the hook. Idempotent. Reads configuration from the environment.

    Raises :class:`HookConfigError` if required env vars are missing.

    Known limitation: a user module that executed
    ``from triton.compiler.compiler import compile`` *before* :func:`install`
    ran has already captured the original callable into its own namespace, and
    our monkey-patch cannot reach that binding. In practice the kerneldex CLI
    runs this before any user code and Triton's own internal call sites go
    through the module attribute we patch, so this is only observable with
    non-standard early imports.
    """
    global _ORIG_COMPILE, _SIG, _INSTALLED

    # Validate env up front so users get a clear error before any kernel
    # compiles.
    _read_target_from_env()
    out_dir = _read_out_from_env()

    with _LOCK:
        if _INSTALLED:
            _trace("install() called while already installed; no-op")
            return

        import triton
        import triton.compiler.compiler as _tcc

        _ORIG_COMPILE = _tcc.compile
        try:
            _SIG = inspect.signature(_ORIG_COMPILE)
        except (TypeError, ValueError) as exc:
            raise HookConfigError(
                f"cannot introspect triton.compile signature: {exc}"
            ) from exc

        _tcc.compile = compile_hook
        triton.compile = compile_hook  # type: ignore[attr-defined]
        if hasattr(triton, "compiler"):
            # older Tritons also expose triton.compiler.compile
            triton.compiler.compile = compile_hook  # type: ignore[attr-defined]

        _INSTALLED = True

    _append_manifest(out_dir, {
        "status": "install",
        "out_dir": out_dir,
        "target": os.environ.get(KERNELDEX_TARGET),
        "backend": os.environ.get(KERNELDEX_BACKEND, "hip"),
    })
    print(
        f"[kerneldex] hook installed; target={os.environ.get(KERNELDEX_TARGET)!r} "
        f"out={out_dir!r}",
        file=sys.stderr,
        flush=True,
    )


def restore() -> None:
    """Undo :func:`install`. Safe to call when not installed."""
    global _ORIG_COMPILE, _SIG, _INSTALLED
    with _LOCK:
        if not _INSTALLED or _ORIG_COMPILE is None:
            return
        import triton
        import triton.compiler.compiler as _tcc

        _tcc.compile = _ORIG_COMPILE
        triton.compile = _ORIG_COMPILE  # type: ignore[attr-defined]
        if hasattr(triton, "compiler"):
            triton.compiler.compile = _ORIG_COMPILE  # type: ignore[attr-defined]
        _ORIG_COMPILE = None
        _SIG = None
        _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED
