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

## What you'll trip on

- **Per-level forks, not a shared library.** A bug fix in `src/Kernel/CoProcessor/CoProcessor/Co_Hj.cpp` almost certainly needs to be mirrored to the same file under `src/Operator/.../CoProcessor/` and `src/Query/.../CoProcessor/`. Same for `KernelEngine` primitives. Use `diff -r` between the trees before assuming they're in sync.
- **Hard-coded magic numbers.** `numQueries`, `numThread`, and `Query_rLen = 16 * 1024 * 1024` (≈16M tuples per relation) are top-of-file globals in `CoProcessorApp.cpp`. Several arrays are sized at `[12]` or `[60]` (query slots / per-kernel burden tracking) — check those if you change query counts.
- **Windows leftovers.** `.sln`, `.vcxproj`, `.suo`, `.bat`, `gen.bat`, `Debug/`, `Release/`, `ipch/`, `_UpgradeReport_Files/`, `TableDataGenerator.exe` etc. are not used on Linux. Don't try to build via them.
- **`OperatorSheduler` (one `c`)** is the on-disk directory name in `bin/release/`. `handshake.sh` and several scripts spell it that way intentionally.
- **`KernelTimeSpecification.list`** is regenerated by handshaking and consumed by every scheduler. If timing-based decisions look wrong on a new machine, re-run `./handshake.sh` before debugging the scheduler logic.
- **Output files.** Each scheduler writes to its local `Output/` (and `output/` for handshaking). Multiple `ExpOut_*.txt` / `ExpOut_*.tony` files are intermediate burden-tracking dumps.
