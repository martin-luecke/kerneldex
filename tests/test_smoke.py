"""Smoke tests that do not require a GPU or a functioning Triton install.

The tests exercise kerneldex's pure-Python machinery (argument parsing,
hook configuration validation, histogram/coverage parsers, report renderer)
by feeding them hand-built fixtures. The capture path is covered by a
GPU-dependent integration test elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from kerneldex import hook  # noqa: E402
from kerneldex.cli import _build_parser  # noqa: E402
from kerneldex.coverage import _parse_stdout  # noqa: E402
from kerneldex.paths import resolve_dex  # noqa: E402
from kerneldex.report import render  # noqa: E402


def test_default_warp_size() -> None:
    assert hook._default_warp_size("gfx942") == 64
    assert hook._default_warp_size("gfx1100") == 32
    assert hook._default_warp_size("gfx1250") == 32
    # gfx90a has a trailing letter; the major is still 9 -> wave64.
    assert hook._default_warp_size("gfx90a") == 64
    assert hook._default_warp_size("gfx908") == 64
    # Unknown / malformed inputs fall back to wave32 rather than raising.
    assert hook._default_warp_size("unknown") == 32
    assert hook._default_warp_size("gfx") == 32


def test_install_refuses_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hook.KERNELDEX_TARGET, raising=False)
    monkeypatch.delenv(hook.KERNELDEX_OUT, raising=False)
    with pytest.raises(hook.HookConfigError):
        hook.install()


def test_cli_parser_requires_subcommand() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_capture_parses(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "capture", "--target", "gfx1250", "--out", str(tmp_path),
        "--", "python", "script.py", "--flag",
    ])
    assert args.target == "gfx1250"
    # argparse.REMAINDER keeps the leading `--`; the CLI handler strips it.
    assert args.user_argv in (
        ["python", "script.py", "--flag"],
        ["--", "python", "script.py", "--flag"],
    )


def test_coverage_parser_fail() -> None:
    stdout = "  FAIL _reduce (0/1) -> v_add_lshl_u32 [VOP3]"
    parsed = _parse_stdout(stdout)
    assert parsed == [{
        "kernel_name": "_reduce",
        "outcome": "fail",
        "lifted": "",
        "total": "",
        "mnemonic": "v_add_lshl_u32",
        "format": "VOP3",
    }]


def test_coverage_parser_ok() -> None:
    stdout = "  OK _trivial_add (3/3)"
    parsed = _parse_stdout(stdout)
    assert parsed[0]["kernel_name"] == "_trivial_add"
    assert parsed[0]["outcome"] == "ok"
    assert parsed[0]["lifted"] == "3"
    assert parsed[0]["total"] == "3"


def test_coverage_parser_ignores_summary_prose() -> None:
    # A tight ``OK`` regex must not match summary sentences that happen to
    # start with 'OK'. This is the tooling-agnostic alternative to a token
    # deny-list.
    stdout = "\n".join([
        "OK and 0 Code Objects Had No Match",
        "OK _trivial_add (1/1)",
        "OK _trailing trailing words here",
    ])
    parsed = _parse_stdout(stdout)
    assert [p["kernel_name"] for p in parsed] == ["_trivial_add"]


def test_resolve_dex_requires_top_level(tmp_path: Path) -> None:
    (tmp_path / "kernels").mkdir()
    kernels, reports = resolve_dex(tmp_path)
    assert kernels == tmp_path / "kernels"
    assert reports == tmp_path / "reports"


def test_resolve_dex_rejects_kernels_subdir(tmp_path: Path) -> None:
    (tmp_path / "kernels").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_dex(tmp_path / "kernels")


def test_override_target_preserves_src_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_override_target must replace only target/options and leave every
    other argument (positional or kw) untouched. This is what protects us
    from future additions to Triton's compile signature."""
    import inspect

    def fake_compile(src, target=None, options=None, extra_future_kwarg=None):
        return None

    monkeypatch.setattr(hook, "_SIG", inspect.signature(fake_compile))

    args, kwargs = hook._override_target(
        ("src_obj",),
        {"target": "original", "options": {"o": 1}, "extra_future_kwarg": 42},
        "new_target",
    )
    merged = inspect.signature(fake_compile).bind_partial(*args, **kwargs).arguments
    assert merged["src"] == "src_obj"
    assert merged["target"] == "new_target"
    assert merged["options"] is None
    # Unknown kwarg must pass through unchanged.
    assert merged["extra_future_kwarg"] == 42


def test_report_renders_on_empty_dex(tmp_path: Path) -> None:
    """A dex with only an install row still produces a minimal report."""
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    (kernels / "manifest.jsonl").write_text(
        json.dumps({"status": "install", "target": "gfx1250", "backend": "hip",
                    "out_dir": str(kernels)}) + "\n"
    )
    out = render(tmp_path)
    text = out.read_text()
    assert "gfx1250" in text
    assert "Kernel inventory" in text
