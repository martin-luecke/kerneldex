# kerneldex

*Catalog the GPU kernels a Triton workload emits for a target ISA, with
instruction histograms and optional per-kernel coverage against an
external tool.*

kerneldex answers one question as precisely as possible:

> **"What GPU instructions does *my* workload produce on target ISA *X*?"**

Point it at any Python program that uses Triton, or at an existing
directory of pre-built `.hsaco` / `.co` code objects. kerneldex either
intercepts every Triton compilation (redirecting it at the target ISA
you asked about) or ingests the pre-built corpus, and then (offline)
gives you:

- A per-kernel inventory with content-addressed hashes.
- A mnemonic histogram across the whole corpus plus per-kernel breakdowns.
- Optional per-kernel coverage against any external tool you point it at,
  with a prioritized list of the missing handlers that block the most
  kernels.

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
4. **(Optional) An external tool** if you want the `coverage` subcommand.
   kerneldex invokes it as `<binary> <hsaco_path> [extra args...]` and
   parses a simple stdout protocol; see
   [`docs/design.md`](docs/design.md#architecture) for details.

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

### Triton workload (live capture)

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

### Pre-built corpus (import)

If you already have a directory of `.hsaco` / `.co` files (e.g.
precompiled CK kernels, JIT cache dumps, vendor-shipped code objects),
skip `capture` and use `import` instead:

```
kerneldex import /path/to/corpus --out ./dex --target gfx950
kerneldex histogram ./dex
kerneldex report ./dex
```

`import` walks the source recursively, deduplicates by content hash,
and drops every code object into `./dex/kernels/` under a flattened,
collision-free name. By default it creates symlinks (fast, no disk
cost); pass `--mode copy` for a self-contained dex.

### Coverage (optional)

Per-kernel coverage, if you have a tool to point at:

```
kerneldex coverage ./dex \
    --tool /path/to/your/tool \
    --tool-arg --isa=gfx1250 \
    --tool-arg --verbose
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

## Example output

Running the quickstart against `examples/trivial_add.py` for `gfx1250`
produces a `REPORT.md` that looks like this (abbreviated):

```markdown
# kerneldex report: gfx1250

- target: `gfx1250` (backend `hip`)
- compile attempts logged: 1
- unique kernels captured: 1
- compile failures recorded: 0

## Kernel inventory

| kernel symbol | source module | hash         | hsaco                            |
|---------------|---------------|--------------|----------------------------------|
| add_kernel    | __main__      | fe89a7d063d7 | add_kernel_fe89a7d063d7.hsaco    |

## Mnemonic histogram (top 25)

| mnemonic             | total | num_kernels |
|----------------------|-------|-------------|
| s_code_end           | 111   | 1           |
| s_mov_b32            | 4     | 1           |
| s_wait_xcnt          | 3     | 1           |
| buffer_load_b32      | 2     | 1           |
| s_lshl_b32           | 2     | 1           |
| ...                  | ...   | ...         |
| v_add_f32_e32        | 1     | 1           |
| v_cndmask_b32_e32    | 1     | 1           |

- 24 unique mnemonics total
```

`reports/mnemonic_histogram.csv` is the sorted long form of that table:

```csv
mnemonic,total_count,num_kernels,kernels
s_code_end,111,1,add_kernel_fe89a7d063d7.hsaco
s_mov_b32,4,1,add_kernel_fe89a7d063d7.hsaco
s_wait_xcnt,3,1,add_kernel_fe89a7d063d7.hsaco
buffer_load_b32,2,1,add_kernel_fe89a7d063d7.hsaco
...
```

`reports/per_kernel_mnemonics.jsonl` has one line per captured kernel,
suitable for piping to `jq` or further processing:

```json
{"kernel": "add_kernel_fe89a7d063d7.hsaco", "unique_mnemonics": 24, "total_instructions": 142, "mnemonics": {"s_code_end": 111, "s_mov_b32": 4, ...}}
```

If you also run `kerneldex coverage`, the report gains two more sections:
a per-kernel outcome table (`ok` / `fail` / `crash:<rc>` / `timeout:<s>` /
`unknown`) and a **prioritized missing-handler worklist** grouping failures
by `(mnemonic, format)` and sorting them by how many kernels a single
handler would unblock - the greedy worklist is the deliverable most people
actually care about:

```markdown
## Prioritized missing-handler worklist

| mnemonic        | format | kernels unblocked | kernels                       |
|-----------------|--------|-------------------|-------------------------------|
| v_add_lshl_u32  | VOP3   | 4                 | reduce_a, reduce_b, downcast, ... |
| s_bfe_i32       | SOP2   | 3                 | matmul_bf16, matmul_mxfp4_a, ... |
| v_mov_b64       | VOP1   | 1                 | bitmatrix_stage1              |
```

---

## How the coverage step works

`kerneldex coverage` has no opinion on what "coverage" means. It invokes
an external binary you supply once per captured `.hsaco`, in an isolated
subprocess:

```
<your-binary> <path/to/kernel.hsaco> [extra args via --tool-arg ...]
```

and infers the per-kernel outcome from the exit code and stdout:

- exit 0 → `ok`
- non-zero exit → `crash:<rc>` (with the last stderr line attached as a note)
- timeout → `timeout:<seconds>`
- your tool printed `OK <kernel-name>` on stdout → `ok` (optionally with a
  `(lifted/total)` count)
- your tool printed `FAIL <kernel-name> ... -> <mnem> [<fmt>]` → `fail`,
  and `<mnem>`/`<fmt>` feed the missing-handler worklist

Typical things to plug in:

- a lifter that raises AMDGPU machine code back to higher-level IR;
- a validator that asserts the kernel uses only instructions your
  simulator / emulator / interpreter supports;
- a static analyzer looking for specific opcodes or patterns;
- any custom `.hsaco`-consuming tool whose coverage against real
  Triton workloads you want to measure.

Exit codes alone are enough to get a usable report (`ok` / `crash:<rc>`);
printing `OK` / `FAIL` lines just gets you per-kernel granularity and the
mnemonic-level worklist.

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

kerneldex import  <source-dir> --out <dir> \
                  [--target <arch>] [--mode symlink|copy] [--force]

kerneldex histogram <dir> [--objdump <path>]

kerneldex coverage  <dir> --tool <path> [--tool-arg <arg> ...]

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
