# kerneldex

*Catalog the GPU kernels a Triton workload emits for a target ISA, with
instruction histograms and optional translator-coverage.*

kerneldex answers one question as precisely as possible:

> **"What GPU instructions does *my* workload produce on target ISA *X*?"**

Point it at any Python program that uses Triton. kerneldex intercepts every
Triton compilation, redirects it at the target ISA you asked about, stores
the resulting HSA code objects, and then (offline) gives you:

- A per-kernel inventory with content-addressed hashes.
- A mnemonic histogram across the whole corpus plus per-kernel breakdowns.
- Optional translator-coverage against any external raiser/lifter you point
  it at, with a prioritized list of the missing handlers that block the
  most kernels.

The tool is deliberately small, principled, and Triton-only in v0.x.

---

## Requirements

kerneldex wraps your existing toolchain rather than shipping a compiler or
a disassembler. Please make sure the following are in place before running
it; kerneldex will fail loudly with a useful message when any of them is
missing at the point it's needed.

1. **Python 3.10+.**
2. **A Triton install that supports the target ISA you care about.** This
   is the most common pitfall. In particular, the stock `pytorch-triton-rocm`
   wheel does not support every AMDGPU target; if you need e.g. `gfx1250`
   coverage, build Triton from source and put its `python/` on
   `PYTHONPATH` before invoking kerneldex. kerneldex does **not** bundle
   Triton and does **not** paper over the failure: if the target compile
   raises, that row in `manifest.jsonl` will carry the full traceback, and
   the final report will show zero captured kernels.
3. **`llvm-objdump` that supports the target ISA.** Older LLVM builds
   (e.g. LLVM 14) crash on `gfx1250` code objects. Install a recent LLVM
   and either put it on `$PATH`, pass `--objdump /path/to/llvm-objdump`,
   or set `$KERNELDEX_OBJDUMP`.
4. **(Optional) An external raiser/translator binary** if you want the
   `coverage` subcommand. kerneldex invokes it as
   `<binary> <hsaco_path> [extra args...]` and parses a simple stdout
   protocol; see [`docs/design.md`](docs/design.md#architecture) for
   details.

kerneldex itself has no Python runtime dependencies beyond the standard
library.

---

## Install

```
git clone <remote>   # once you've attached a git remote
cd kerneldex
pip install -e .
```

This exposes the `kerneldex` console script and the `python -m kerneldex`
module.

---

## Quickstart

Capture, summarize, report - three one-liners:

```
# 1. Run *any* Python program; kerneldex silently catalogs every Triton
#    kernel it compiles, against the target ISA you pick.
kerneldex capture --target gfx1250 --out ./dex -- \
    python examples/trivial_add.py

# 2. Disassemble the captured corpus and aggregate mnemonics.
kerneldex histogram ./dex

# 3. Render a human-readable REPORT.md.
kerneldex report ./dex
```

Optional fourth step - translator coverage, if you have a raiser to point
at:

```
kerneldex coverage ./dex \
    --raiser /path/to/your/raiser \
    --raiser-arg --isa=gfx1250 \
    --raiser-arg --verbose
kerneldex report ./dex      # re-render; now includes the coverage table
```

What you'll find in `./dex` after this:

```
dex/
├── REPORT.md                    # human summary, inventory + histograms + coverage
├── kernels/
│   ├── <symbol>_<hash>.hsaco    # one per unique captured kernel
│   └── manifest.jsonl           # install + per-compile log (ok / fail rows)
└── reports/
    ├── mnemonic_histogram.csv       # global histogram, sorted by count
    ├── per_kernel_mnemonics.jsonl   # per-kernel mnemonic counts
    └── coverage.csv                 # present iff you ran `coverage`
```

---

## What kerneldex is (and is not)

**It is:** a static, offline catalog tool for Triton-on-AMDGPU workloads.
It observes; it does not transform the kernels the user's program runs,
and it does not fall back to a different target or different options if
the target compile fails.

**It is not:** a profiler, a performance counter tool, a lifter, a
translator, a replacement for Triton's own dumping options, or a solution
for non-Triton kernel paths (raw HIP, CUDA C++, Inductor-C++, Gluon, etc.).

See [`docs/design.md`](docs/design.md) for the full principles and
architecture.

---

## Known limitations

These are intentional tradeoffs, not bugs, but worth calling out up front.

- **Each Triton compile runs twice.** Once against the target you asked to
  catalog, once against the caller's original target so the user's program
  keeps running unchanged. This doubles compile cost but guarantees
  observe-don't-transform semantics. Triton's own on-disk cache absorbs
  most of the cost on repeated runs.
- **Early imports of `compile` bypass the hook.** A module that executed
  `from triton.compiler.compiler import compile` *before* `hook.install()`
  has already bound the original callable into its own namespace, and
  monkey-patching the module attribute cannot reach that binding. In
  practice the CLI runs `install()` before any user code and Triton's own
  internal call sites dispatch via the module attribute we patch, so this
  is only observable with non-standard early imports (e.g. something in
  `sitecustomize.py`).
- **Triton only.** Kernels compiled through torch.compile's C++ backend,
  raw HIP/CUDA, Gluon, or direct MLIR paths are not captured.

## CLI reference

```
kerneldex capture --target <arch> --out <dir> \
                  [--backend hip] [--warp-size N] [--trace] \
                  -- <python> <script.py> [args...]

kerneldex histogram <dir> [--objdump <path>]

kerneldex coverage  <dir> --raiser <path> [--raiser-arg <arg> ...]

kerneldex report    <dir>
```

The first token after `--` in `capture` must be a Python interpreter;
kerneldex injects itself via `-m kerneldex._preload` and hands the rest
of the arguments to `runpy`.

### Environment variables

kerneldex sets these in the capture subprocess; you can also set them by
hand if you need to install the hook from inside an existing Python
process (e.g. a notebook):

| var                       | purpose                                                   |
| ------------------------- | --------------------------------------------------------- |
| `KERNELDEX_TARGET`        | Target arch, e.g. `gfx1250`. **Required.**                |
| `KERNELDEX_BACKEND`       | Triton backend, default `hip`.                            |
| `KERNELDEX_WARP_SIZE`     | Override warp size (default: 32 for gfx10+, else 64).     |
| `KERNELDEX_OUT`           | Directory to write `.hsaco` and `manifest.jsonl` into.    |
| `KERNELDEX_OBJDUMP`       | Specific `llvm-objdump` binary for `histogram`.           |
| `KERNELDEX_TRACE`         | Truthy -> verbose hook logging on stderr.                 |

---

## Programmatic use

Prefer the CLI unless you have a reason not to. If you do need to install
the hook from inside an existing Python process:

```python
import os
from kerneldex import hook

os.environ["KERNELDEX_TARGET"] = "gfx1250"
os.environ["KERNELDEX_OUT"]    = "/tmp/my_dex/kernels"

hook.install()
# ... run your workload here ...
hook.restore()
```

The hook reads its configuration at `install()` time and fails fast if
the required env vars are missing.

---

## Development

```
pip install -e '.[dev]'
pytest tests/
```

The test suite is GPU-free; it covers the CLI parser, config validation,
stdout parsers, and the report renderer. GPU-dependent capture runs are
left to users' own integration workflows (there is no kerneldex CI
against a real `gfx*` device yet).

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
