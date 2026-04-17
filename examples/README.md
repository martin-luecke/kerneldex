# kerneldex examples

These are deliberately tiny workloads to exercise the capture / histogram /
report path end to end.

## `trivial_add.py`

Single-kernel element-wise add. Good smoke test.

```
kerneldex capture --target gfx1250 --out ./dex -- python examples/trivial_add.py
kerneldex histogram ./dex
kerneldex report    ./dex
```

Expect to see one `.hsaco` under `dex/kernels/` and a one-row kernel
inventory in `dex/REPORT.md`.
