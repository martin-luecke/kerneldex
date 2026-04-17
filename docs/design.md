# kerneldex design

## Purpose

Answer a single question, precisely: **"What GPU instructions does *my*
workload produce on target ISA *X*?"**

A typical use case is bringing up new codegen or analysis tooling for a
target ISA: before writing handlers, you want to know which mnemonics
actually show up in the kernels the workload emits, not the ones the ISA
manual lists. kerneldex automates that measurement.

## Scope

**In scope (v0.x):**
- Triton kernels compiled via `triton.compiler.compiler.compile`.
- AMDGPU targets (the code object format we persist is HSA `.hsaco`).
- Offline analysis: instruction histograms and pluggable per-kernel coverage.

**Explicitly out of scope (for now):**
- Gluon, direct-MLIR, or pure-LLVM kernel paths.
- Kernels compiled through `torch.compile` that do not go through Triton
  (e.g. Inductor's C++ backend).
- NVIDIA / CUDA-only targets (the code object extraction path is
  AMDGPU-specific today).
- Runtime profiling, performance counters, timing. kerneldex is a static
  corpus tool, not a profiler.
- Bundling our own coverage tool. Coverage is pluggable.

## Principles

### 1. Observe, don't transform

The capture path runs the user's program unmodified. We intercept Triton
compilation, compile twice (once against the target we care about, once
against the host target so execution continues), and never return a
modified kernel to the host runtime. Workloads behave exactly as they
would without kerneldex.

### 2. No fallbacks

If the target compile fails, we record the failure verbatim and make it
visible in the report. We do **not** silently try a different target,
different options, or the host's target as a proxy. The whole point of
kerneldex is to surface what happens when you ask the compiler for a
specific ISA; masking failures with fallbacks would make the results
meaningless.

(Related rule: `if exception: pass` is banned in this codebase.)

### 3. Subprocess isolation at stage boundaries

The external coverage tool is the process most likely to crash hard -
it is often under active development and may hit fatal asserts on
exotic inputs. We run it in a child process per kernel so one crash
never eats the whole corpus. The same isolation is available (and
encouraged) at the capture layer when driving many independent
workloads, but we leave that to the user to orchestrate because capture
is inherently Python-native and benefits from shared JIT caches within
a run.

### 4. Pluggable coverage, zero opinion

kerneldex ships no coverage tool of its own and no disassembler
analysis beyond mnemonic counting. It invokes what the user already has
and parses a deliberately simple stdout protocol. This keeps the
package boundary narrow and avoids tying kerneldex to any specific
tool's API.

### 5. Deterministic, diffable outputs

- Kernels are keyed by a content hash of their identity (function name,
  module, signature, constexprs). Two runs of the same workload produce
  identical file names and identical manifest rows.
- All aggregate outputs are plain CSV / JSONL / Markdown, sorted.
- The report renderer reads only on-disk artifacts; rerunning `kerneldex
  report` against a frozen `dex/` directory always yields the same file.

### 6. Transparent requirements

The user is responsible for:
- A Triton install that supports the target ISA.
- An `llvm-objdump` that supports the target ISA.
- An external tool for coverage (if desired).

kerneldex refuses to start (with a clear error) when any of these is
missing at the point it's needed, instead of pretending to work with
reduced fidelity.

## Architecture

```
   user script.py
         |
         v
+----------------------+     sets env:
|  kerneldex capture   | --- KERNELDEX_TARGET
|  (parent process)    |     KERNELDEX_OUT
+----------------------+     ...
         |
         |   subprocess: python -m kerneldex._preload <script.py> [argv...]
         v
+------------------------+
|  kerneldex._preload    |
|  - reads env config    |
|  - hook.install()      |
|  - runpy.run_path(...) |
+------------------------+
         |
         v
  user script imports triton,
  triton.compile()  ---->  monkey-patched compile_hook
                             |
                             |  1. compile for target; persist .hsaco +
                             |     append manifest.jsonl  (no fallback)
                             |  2. compile for host; return that to user
                             v
            dex/
            ├── kernels/
            │   ├── <symbol>_<hash>.hsaco
            │   └── manifest.jsonl
            └── reports/     (created by the next stages)
```

After capture, three read-only stages operate on `dex/`:

- **`histogram`**: walks `kernels/*.hsaco`, shells out to `llvm-objdump`,
  tokenizes mnemonics, writes `reports/mnemonic_histogram.csv` (global)
  and `reports/per_kernel_mnemonics.jsonl` (per-kernel).

- **`coverage`**: walks `kernels/*.hsaco`, spawns the user-supplied
  tool per kernel (isolated subprocess), parses a simple stdout
  protocol, writes `reports/coverage.csv`. Tool crashes are surfaced as
  `crash:<rc>` rows with the last stderr line attached.

- **`report`**: reads `manifest.jsonl` + the two CSV files above and
  renders `REPORT.md` with the inventory, top-N mnemonics, per-kernel
  coverage, and a prioritized missing-handler worklist (mnemonics
  ordered by the number of captured kernels each would unblock).

## Non-goals

- **Automatic remediation.** kerneldex reports what's missing; fixing
  handlers is a job for the codegen / tool developer. The worklist
  intentionally stops at "handler X unblocks these N kernels" - it does
  not suggest ISA semantics or handler code.
- **Comparison across runs.** Not in v0.x. The deterministic outputs are
  already diffable with `diff` / `git diff`; a first-class "what changed
  between two dex directories" command could come later if there's
  demand.
- **Packaging a bundled LLVM.** We resolve `llvm-objdump` at runtime and
  expect the user's toolchain to be sufficient. Supporting a bundled
  build would double the repository's maintenance surface for no clear
  benefit.
