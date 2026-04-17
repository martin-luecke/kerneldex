# kerneldex tests

`test_smoke.py` exercises the pure-Python machinery (CLI parser, config
validation, stdout parsers, report renderer). It does not require a GPU,
Triton, or `llvm-objdump`, so it can run in minimal CI.

```
pip install -e '.[dev]'
pytest tests/
```

An end-to-end integration run (capture + histogram + report against
`examples/trivial_add.py`) is described in the top-level README's
"Quickstart" section. We intentionally do not wire it into pytest because
it requires an arch-supporting Triton and a recent `llvm-objdump`, neither
of which is a reasonable CI dependency.
