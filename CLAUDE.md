# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

OmniDB is a research prototype of a parallel database co-processor that splits relational query work between the CPU and an integrated GPU via OpenCL. The original codebase targeted Windows + AMD/NVIDIA OpenCL SDKs (Visual Studio `.sln`/`.vcxproj` files are still checked in); the actively maintained build path is now Linux + Intel via PoCL (Portable Computing Language). The repo investigates three different *granularities* of CPU/GPU scheduling, each implemented as a separate, self-contained tree.

## The three scheduling levels (the central architectural axis)

The code is duplicated into three sibling directories under `src/`, one per scheduling granularity. They share most source filenames but diverge in scheduler logic:

| Level | Source root | Deployed to |
| --- | --- | --- |
| Kernel-level (finest) | `src/Kernel/` | `bin/release/KernelScheduler/` |
| Operator-level | `src/Operator/` | `bin/release/OperatorSheduler/` (note misspelling) |
| Query-level (coarsest) | `src/Query/` | `bin/release/QueryScheduler/` |

Each level is its own buildable unit with two halves:

- **`KernelEngine` / `KernelEngine_KernelSchedule` → `DLL_OpenCL/`** — the OpenCL primitive library (`Selection`, `Projection`, `GroupBy`, `BinaryJoin`, `IndexJoin`, `Sort`, `AggAfterGroupBy`, `Singular`, etc., plus `KernelScheduler`/`scheduler` for that level). Builds `libomnidb_opencl.so` (or `.a`) plus a standalone `exec` binary that runs the engine's own self-tests.
- **`CoProcessor` (or `CoProcessor-{Operator,Query}Scheduler`)** — the higher-level query engine. Parses query plans (`QueryPlanTree`/`QueryPlanNode`/`PredicateTree`), dispatches operators (`Co_Hj`, `Co_Smj`, `Co_Inlj`, `CO_Ninlj`, `Co_Sort`), runs the thread pool (`MyThreadPoolCop`), and drives the dynamic CPU/GPU burden balancer (`DynamicQueryProcessor`, `HandShaking`). Builds `CoProcessorApp` (Linux) / `CoProcessor.exe` (Windows). Links against the matching `libomnidb_opencl.so`.

When making changes, **the same fix usually has to be applied in 2–3 of these trees** — they were forked rather than parameterized. Pick the right tree based on which scheduling level you're working on; don't assume changes propagate.

`src/handshaking/DLL_OpenCL/` is a *fourth*, separate engine build whose only job is to run device-capability handshaking and emit `KernelTimeSpecification.list` (per-kernel timing baselines used by all three schedulers). It is not part of any scheduling level.

## Build commands

The Visual Studio `.sln`/`.vcxproj`/`.bat` files are legacy. Use the Linux Makefiles:

```bash
# 1. Build the engine for a given scheduling level (produces libomnidb_opencl.so/.a + exec)
cd src/Kernel/KernelEngine_KernelSchedule/DLL_OpenCL && make           # default: exec
cd src/Kernel/KernelEngine_KernelSchedule/DLL_OpenCL && make shared    # libomnidb_opencl.so
cd src/Kernel/KernelEngine_KernelSchedule/DLL_OpenCL && make static    # libomnidb_opencl.a
# Same Makefile pattern under src/Operator/KernelEngine/DLL_OpenCL and src/Query/KernelEngine/DLL_OpenCL

# 2. Build the CoProcessor for that level
cd src/Kernel/CoProcessor && make full-build && make deploy            # builds engine .a too, then deploys to bin/release/KernelScheduler/
cd src/Operator/CoProcessor-OperatorScheduler && make                  # expects libomnidb_opencl.so already in ./lib/
cd src/Query/CoProcessor-QueryScheduler && make                        # expects libomnidb_opencl.so already in ./lib/

# 3. Build the handshaking engine
cd src/handshaking/DLL_OpenCL && make                                  # produces exec used by bin/release/Handshaking/
```

Notes:

- `src/Kernel/CoProcessor/Makefile` is the most complete: it links `libomnidb_opencl_nomain.a` (the static lib with `mainProgram.o` stripped via `ar d`) so the result is a single self-contained executable with no `.so` runtime dep. The Operator and Query trees instead expect `libomnidb_opencl.so` to be pre-staged in `./lib/` and use `-Wl,-rpath,'$$ORIGIN'` so the binary loads it from its own directory at runtime. Build the engine *first*, copy/symlink `libomnidb_opencl.so` and `primitive.cl` into `./lib/`, then run `make`.
- `make debug` enables `-g -DDEBUG`. `make clean` removes `build/` and the binary.
- OpenCL is detected via `pkg-config OpenCL`, falling back to `-I/usr/include -lOpenCL`. Override with `OPENCL_INC=` / `OPENCL_LIB=` on the Operator/Query Makefiles.
- Required system packages: `ocl-icd-opencl-dev` and either `pocl` (CPU) or the Intel OpenCL runtime (GPU). `make check-opencl` runs `clinfo -l`.
- The `gen.bat` files and `*.tony`, `*.o`, `exec` artifacts are gitignored.

## Running benchmarks

The deployment layout under `bin/release/` is what `handshake.sh` expects:

```bash
cd bin/release
./handshake.sh                  # FIRST RUN ONLY on a new machine — generates KernelTimeSpecification.list
                                # by running ./Handshaking/DLL_OpenCL, then copies the list into each
                                # scheduler directory (KernelScheduler, OperatorSheduler, QueryScheduler).
cd KernelScheduler && ./test.sh # or OperatorSheduler / QueryScheduler — sweeps numQueries × numThreads
```

Direct invocation of any scheduler binary takes two args: `./CoProcessorApp <numQueries> <numThreads>` (defaults: 30 queries, 4 threads — see `CoProcessorApp.cpp`). The Kernel scheduler's binary is named `CoProcessor`; Operator and Query are `CoProcessorApp`.

`test.sh` sweeps `numThreads ∈ {2,4,6,8}` × `numQueries ∈ {15,20,25,30}`. **`bin/release/QueryScheduler/test.sh` carries an explicit warning**: with multiple threads `CoProcessorApp` can hang inside `pocl_level0_wait_event` (PoCL Intel GPU backend); reduce concurrency to 1 (`./CoProcessorApp 15 1`) if the hang reproduces.

Each scheduler's working dir must contain `RS.conf` (lists the data files: `R.a*`, `S.a*`), `dbmbench.conf`, `KernelTimeSpecification.list`, and `primitive.cl`. The TPC-H tables themselves live next to the binary or wherever `RS.conf` points.

## CPU-assisted prefetching (WAS) and the In-Cache co-processing port

### Source paper and goal

This line of work adapts **He, Zhang & He, "In-Cache Query Co-Processing on Coupled CPU-GPU Architectures," VLDB 2015** (`In_Cache_on_Coupled.pdf`) to OmniDB. Its prefetch structure is itself from **Zhou et al., VLDB 2005** (helper-thread Work-Ahead Set, "WAS"). The paper abstracts three functional modules — **P** (prefetching), **D** (decompression, optional), **E** (query execution) — and four CU-assignment configs: **PE** and **cPDE-c/b/g** (Figure 4). On AMD APUs (CPU and iGPU **share L2**) it reports +24% (selection) / +22% (hash join) for PE over the E baseline, and up to +36–40% for cPDE on TPC-H.

**Paper module → code mapping in this repo:**

| Paper module | Code artifact | Status |
| --- | --- | --- |
| **P** (prefetch) | `WAS_kernel` (kid=51) on a 1-CU prefetch sub-device | implemented (handshaking tree only) |
| **E** (execution) | operator kernels: `filterImpl_map_kernel` (kid=20, selection), `probe_kernel` (kid=50, hash-join probe) | implemented |
| **D** (decompression) | — | **not implemented** (deferred) |
| cost model (paper §4) | — | **not implemented** (deferred) |

### Where WAS lives today (handshaking tree ONLY)

All WAS code currently lives in `src/handshaking/DLL_OpenCL/` — the standalone self-test engine, **not** in any scheduler tree. The Kernel/Operator/Query `primitive.cl` files contain zero WAS code. Artifacts:

- **`primitive.cl`** — `WASEntry` struct (`__global uint *p1,*p2` + `long state1,state2`); `post()` inline helper (overwrites one slot, returns the displaced entry by value); `WAS_kernel` (kid=51, between `//>>>WASK_BEGIN..//<<<WASK_END` markers); and the WAS path inside `filterImpl_map_kernel` (kid=20) and `probe_kernel` (kid=50). `build_kernel` (kid=49) has **no** WAS (the paper does prefetch the build phase; we don't).
- **`common.cpp`** — device fission in `cl_init_prefetch`: splits the CPU into a 1-CU `PrefetchSubDevice` + 7-CU `MainCPUSubDevice` via `clCreateSubDevices(... CL_DEVICE_PARTITION_BY_COUNTS {1,7})`, plus a full 8-CU `FullCPUCommandQueue` on the parent device for noWAS baselines. All banners are prefixed `[Prefetch]`.
- **`Handshake.cpp`** — the WAS measurement sweep harness (`probe_kernel_handshake`, `filterImpl_map_kernel_handshake`).

**`WAS_kernel` `wasMode` bitfield (arg 4):** bit0=backward traversal, bit1=spin-on-unchanged, bit2=`CPU_PAUSE` in spin, bit3=64B-wide (2 entries/iter, MLP=2), bit4=no-deref ablation. Common combos: 0=F-nospin, 3=B-spin-nopause, 6=F-spin-pause, 7=B-spin-pause.

**Queue selection by `wassize` (a convention used throughout):** `wassize < 0` → noWAS on the 7-CU main sub-device (control); `wassize == 0` → noWAS on the full 8-CU `FullCPUCommandQueue`; `wassize > 0` → WAS: main on 7 CUs + `WAS_kernel` helper launched on `PrefetchCommandQueue`. `was_per_wg = wassize / num_wg` partitions WAS slots per work-group (race-free because PoCL runs a work-group's work-items sequentially on one CU).

### Measurement harness (env vars)

The handshaking driver is parameterized by env vars (no rebuild needed except to switch kernel). `mainProgram.cpp` runs a single-kid loop `for (kid = N; kid < N+1; kid++)` — the driver scripts rewrite that line to switch between kid=20 (filter) and kid=50 (probe).

| Env | Default | Effect |
| --- | --- | --- |
| `PROBE_MODE` / `FILTER_MODE` | 3 / 6 | `wasMode` helper variant |
| `PROBE_ROUNDS` | 1 | dataset-regeneration rounds (each: new R/S seeds 2r/2r+1, rebuild hash table, warm-up, randomized cfg order, Welford+paired stats) |
| `PROBE_REPS` / `FILTER_REPS` | 5 / 50 | timed iterations per cfg |
| `PROBE_ONLY_W` / `FILTER_ONLY_W` | unset | isolate one `wassize` (for perf measurement) |
| `PROBE_NO_HELPER` | unset | run the main WAS path but never launch the helper (ablation) |
| `PROBE_SKIP_WAS` | unset | skip all `wassize>0` cfgs (clean noWAS) |
| `PROBE_RLOG` | unset | per-round timing to stderr |

Results + driver scripts live under `src/handshaking/DLL_OpenCL/`: `results/` (CSVs + `README.md`), `filter_d1792_driver.py`, `probe_d1792_driver.py`, `percore_by_wassize.py`, `run_was_experiments.py`, `parse_*.py`. Per-core perf (`perf stat -A -C 0-7`) needs `perf_event_paranoid <= 0` (resets to default on reboot).

### WAS measurement results so far (CPU-only handshaking, i7-9700)

These are **CPU-only** results from the handshaking harness. They do **not** predict CPU-GPU co-processing behavior, which is the real target (P on CPU + E on GPU) and is still untested.

1. **Helper traversal variant doesn't matter on CPU.** F/B × spin/nospin × pause × 64B-wide were statistically indistinguishable. The helper is therefore **fixed to backward + spin-loop + no-pause** (`wasMode = 3`); don't re-sweep variants.
2. **WAS size has a U-curve; the CPU optimum was ≈ 2–5% of L3** (wassize ≈ 1792–3584 at d=1792) — too small loses latency hiding, too large evicts useful lines. This is only the CPU-side optimum; **WAS size will be re-tested under co-processing.**
3. **On CPU, WAS is at best break-even** (no speedup over noWAS in the handshaking engine). Porting it as the paper's **P** module is still worthwhile so PE-CPU/PE can be built and the in-cache paradigm reproduced; the expected payoff is in CPU-GPU co-processing (P on CPU + E on GPU), not yet measured.

Practical notes when re-running or porting: the `flush`/drain loop is **correctness**-critical (it processes the deferred tail tuples) — keep it when porting; CPU turbo/throttle can shift timings ~10% between light and heavy runs (use fresh per-process starts); per-core cache needs `perf -A -C 0-7` (paranoid ≤ 0).

### Porting WAS into the Kernel scheduler (the deployment path)

The handshaking results are a feasibility probe. To ship PE-CPU/PE in the real query engine, WAS must be **ported** into the Kernel scheduling tree and linked into its CoProcessor:

```
src/handshaking/DLL_OpenCL/                 →  src/Kernel/KernelEngine_KernelSchedule/DLL_OpenCL/
  primitive.cl (WASEntry, post, WAS_kernel,      (0 WAS code today — port kernel + WAS paths)
   WAS path in probe_kernel/filter)
  common.cpp   (cl_init_prefetch, 4 queues)      (no device-fission infra today — port it)
                                             →  make shared (libomnidb_opencl.so) / make static (.a)
                                             →  src/Kernel/CoProcessor links it; operators (Co_Hj probe,
                                                filter) wired to launch the P-helper + run E  = PE-CPU
                                             →  make deploy → bin/release/KernelScheduler/ (+ fresh primitive.cl)
```

**Port surface (3 places):** (1) `primitive.cl` — copy `WASEntry`/`post()`/`WAS_kernel` and the `wassize`-guarded WAS path into `probe_kernel`/`filter`; (2) `common.cpp` — copy `cl_init_prefetch` + the four-queue globals (`PrefetchSubDevice`, `MainCPUSubDevice`, `PrefetchCommandQueue`, `FullCPUCommandQueue`); (3) the CoProcessor operators (`Co_Hj` etc.) — allocate WAS buffers, launch the helper on `PrefetchCommandQueue`, run E on the 7-CU main queue. Remember the per-level fork rule: the same port may be needed for Operator/Query trees later.

**Build/link decision:** either link form is acceptable. Default to the **existing Kernel pattern** — static `libomnidb_opencl_nomain.a` (single self-contained binary, `mainProgram.o` stripped via `ar d`) — since it needs no Makefile surgery. The engine Makefile also has `make shared` (`libomnidb_opencl.so`) if a `.so` + `./lib/` + `-Wl,-rpath,'$$ORIGIN'` layout (like the Operator/Query trees) is ever preferred.

### PE-CPU / PE roadmap (next implementation; D and cPDE deferred)

- **PE-CPU** (paper's CPU-only baseline, Fig 5): P on 1 CPU CU + E on the remaining CPU CUs, no GPU. Closest to what the handshaking harness already runs; the work is packaging it as a named query-engine config in the Kernel CoProcessor.
- **PE** (paper Fig 4a): P on 1 CPU CU + E split across CPU **and** GPU. **GPU is required.** P+GPU was verified empirically in the handshaking engine via `PROBE_GPU_TEST={1,2,3}` (gated harness in `probe_kernel_handshake`):
  - **Stage A** (`=1`): GPU runs `probe_kernel` (E) correctly — GPU match count == CPU's. **PASS.**
  - **Stage B1** (`=2`): CPU helper (`WAS_kernel` mode 3) + GPU main run **concurrently** and the helper terminates cleanly via an `ALLOC_HOST_PTR` ctrl flag. **PASS** (the prior "infinite loop" does not reproduce here).
  - **Stage B2** (`=3`): minimal fine-grained **SVM** coherence probe (GPU producer → CPU consumer). **BLOCKED** — both PoCL devices advertise fine-grained SVM *buffer* sharing, but the CPU consumer never observes the GPU's writes *during concurrent execution* (GPU lacks SVM atomics); the value only appears after `clFinish`. **PoCL Level0 SVM is copy-at-sync, not live-coherent.**
  - **Verdict:** the paper's **concurrent posting-WAS** (GPU posts addresses → CPU helper reads them live) is **impossible on PoCL Level0** — do not re-attempt it. E-on-GPU and P/E concurrency themselves work, so PE is viable only with a **P mechanism that needs no live cross-device coupling**: decoupled/blind prefetch (helper independently warms the shared LLC by streaming the read-only S/hash-table — the iGPU shares L3) or step-pipelined prefetch (coherence used only at `clFinish` barriers, à la the paper's "steps"). This is the open PE design decision (brainstorm next).
  - Note: enabling the SVM test bumped `common.h` to `CL_TARGET_OPENCL_VERSION 200` (OpenCL 2.0); the engine still builds/runs (deprecation warnings only).
- **D / cPDE-c/b/g:** deferred (no decompression module yet).

## What you'll trip on

- **Per-level forks, not a shared library.** A bug fix in `src/Kernel/CoProcessor/CoProcessor/Co_Hj.cpp` almost certainly needs to be mirrored to the same file under `src/Operator/.../CoProcessor/` and `src/Query/.../CoProcessor/`. Same for `KernelEngine` primitives. Use `diff -r` between the trees before assuming they're in sync.
- **Hard-coded magic numbers.** `numQueries`, `numThread`, and `Query_rLen = 16 * 1024 * 1024` (≈16M tuples per relation) are top-of-file globals in `CoProcessorApp.cpp`. Several arrays are sized at `[12]` or `[60]` (query slots / per-kernel burden tracking) — check those if you change query counts.
- **Windows leftovers.** `.sln`, `.vcxproj`, `.suo`, `.bat`, `gen.bat`, `Debug/`, `Release/`, `ipch/`, `_UpgradeReport_Files/`, `TableDataGenerator.exe` etc. are not used on Linux. Don't try to build via them.
- **`OperatorSheduler` (one `c`)** is the on-disk directory name in `bin/release/`. `handshake.sh` and several scripts spell it that way intentionally.
- **`KernelTimeSpecification.list`** is regenerated by handshaking and consumed by every scheduler. If timing-based decisions look wrong on a new machine, re-run `./handshake.sh` before debugging the scheduler logic.
- **Output files.** Each scheduler writes to its local `Output/` (and `output/` for handshaking). Multiple `ExpOut_*.txt` / `ExpOut_*.tony` files are intermediate burden-tracking dumps.
- **WAS code is handshaking-only.** `WAS_kernel`, `post()`, the device-fission queues, and the `wassize`-guarded WAS paths exist only in `src/handshaking/DLL_OpenCL/`. The Kernel/Operator/Query trees have none of it — see "CPU-assisted prefetching (WAS)" for the port plan. Don't assume `diff -r` parity here; this is intentionally divergent until ported.
- **Duplicate kid numbers in `primitive.cl`.** There are two `kid=49` and two `kid=50` kernels: the comment-tagged `// kid=49`/`// kid=50` near the top are `Reorder_kernel`/`Histo3_kernel`; the load-bearing hash-join `build_kernel`/`probe_kernel` carry `// kid 49`/`// kid 50` (no `=`) further down and are what the handshaking driver actually invokes. Grep for the function name, not the kid comment.
- **`mainProgram.cpp` runs a 1-kid range.** The handshaking loop is `for (kid = N; kid < N+1; kid++)` (currently a single kernel). The WAS driver scripts rewrite this line to switch kernels (kid=20 filter ↔ kid=50 probe). If a sweep "runs the wrong kernel," check this line first.
- **`perf_event_paranoid` for per-core WAS measurement.** Per-core perf (`-A -C 0-7`) needs paranoid ≤ 0 (set via `sudo sysctl`), which resets to the distro default on reboot. Re-lower it before re-running `percore_by_wassize.py` etc.
