"""Minimal Triton workload: element-wise add.

Useful as a quick end-to-end test for ``kerneldex capture``. The workload
runs on the host GPU so a host-supported Triton install is required, but
the captured ``.hsaco`` will be for whatever target you pass to
``kerneldex capture --target ...``.

Usage:

    kerneldex capture --target gfx1250 --out ./dex -- \
        python examples/trivial_add.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "no HIP/CUDA device visible - kerneldex's hook still forces the "
            "target compile, but the host-side fallback needs a device."
        )
    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, N, BLOCK=128)
    torch.cuda.synchronize()
    print(f"trivial_add: out[:4] = {out[:4].tolist()}")


if __name__ == "__main__":
    main()
