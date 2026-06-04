# PE Co-Processing (CPU-E with WAS prefetch + GPU-E noWAS) for the Kernel-level Hash Join

**Status:** Design (approved in brainstorming; revised after adversarial spec review)
**Date:** 2026-06-04
**Target:** `src/Kernel/` (Kernel-level scheduler: `KernelEngine_KernelSchedule/DLL_OpenCL` engine + `CoProcessor`)
**Hardware of record:** Intel i7-9700 (8-core, no-SMT) + Intel UHD 630 iGPU, PoCL (CPU + Level0 GPU). The 1+7 device-fission split is hardcoded to this 8-CU host.

---

## 1. Goal and success criterion

Implement a **PE co-processing configuration** for the hash-join probe in the Kernel-level CoProcessor: the probe is co-processed across CPU and GPU, with the **CPU** executor accelerated by CPU-assisted prefetching (WAS). This adapts the **P** (prefetch) and **E** (execution) modules of He, Zhang & He, *In-Cache Query Co-Processing on Coupled CPU-GPU Architectures*, VLDB 2015 to the OmniDB Kernel tree.

**Success criterion (explicit):** the **gate is correctness** — the PE hash join must produce a join result identical to the baseline. A **speedup is NOT a success criterion on this hardware**: WAS is at best break-even on a no-SMT CPU (§2), and the paper's actual benefit mechanism (prefetch *for the GPU*) is closed here (§2). The deliverable is a correct, reproducible PE *structure* in the real engine, with measurement infrastructure to observe its cost/benefit — not a guaranteed win.

## 2. Background and the hardware constraint (verified this session)

The paper's PE prefetches **for the GPU** (the memory-stalled device): GPU-E work-items post their next addresses to a work-ahead set (WAS) and a CPU helper (P) reads them *live* and prefetches into the shared cache. That requires **live cross-device cache coherence**.

On this hardware that path is **closed**, established empirically via the `PROBE_GPU_TEST={1,2,3,4}` harness in `src/handshaking/DLL_OpenCL/Handshake.cpp`:

- **Stage A (`=1`)** — the PoCL Level0 GPU runs the probe kernel (E) correctly (GPU match count == CPU's). E-on-GPU works.
- **Stage B1 (`=2`)** — a CPU helper (`WAS_kernel` mode 3) and a GPU main kernel run concurrently and the helper terminates cleanly. Concurrency + control-signaling work.
- **Stage B2 (`=3`)** — fine-grained SVM coherence test: a CPU consumer kernel **never observes** a GPU producer's writes during concurrent execution; the value appears only after `clFinish`. The single root cause is **no live cross-device SVM coherence on PoCL Level0** — writes reconcile only at sync points ("copy-at-sync"); the GPU's lack of SVM atomics is the symptom. → the paper's concurrent posting-WAS is **impossible** here.
- **Stage C (`=4`, decoupled prefetch)** — a CPU pre-warming the shared LLC gives **no** speedup to iGPU random reads (warm/cold ≈ 0.96–1.01 across 4–64 MB, including L3-fitting sizes). The fallback "decoupled/blind LLC-warming for the GPU" is also futile on this hardware.

**Conclusion (drives this design):** P (WAS prefetch) can only help the **CPU** executor of E; the GPU executor must run **noWAS**. This is the inverse of the paper's PE, but it is the only viable PE on the i7-9700 + PoCL Level0 combination. (The paper's gains rely on a shared L2 + working live coherence present on AMD APUs, absent here.)

These results are recorded in `CLAUDE.md`.

## 3. Architecture

The hash-join probe is co-processed using the CoProcessor's **existing 2-partition, 2-thread design** (`Co_Hj.cpp` `CO_hj` → `tp_hj`). PE changes *what each executor thread runs*, not the dispatch.

**How the existing dispatch actually works (corrected):** `tp_hj` is a **work-stealing** loop — both worker threads pull partition indices from a single shared counter (`*curPartition` under `dispatchMutex`) and process whichever they grab. What is **fixed per thread is its `execMode`**: thread 0 = `EXEC_GPU`, thread 1 = `EXEC_CPU` (set in `CO_hj`). The partition *index* a thread processes is **dynamic**, not a static binding. With `numPartition == 2` and 2 threads it normally resolves to one partition each, but a fast thread can grab both.

**Therefore PE keys the WAS/noWAS decision off `execMode`, never off a partition index:**

```
            CO_hj: GPUCopy_Partition splits R,S into 2 partitions
                       (shared work-stealing queue, unchanged)
                                    |
        +---------------------------+---------------------------+
        |                                                       |
  EXEC_CPU worker thread                              EXEC_GPU worker thread
  (whatever partition(s) it steals)                  (whatever partition(s) it steals)
        |                                                       |
  build hash table (CPU)  ── identical build_kernel ──  build hash table (GPU)
  probe WITH WAS:                                        probe noWAS:
    1 CU = P helper (WAS_kernel mode 3)                    whole iGPU runs probe_kernel
    7 CU = CPU-E (probe_kernel, wassize = W)               (wassize <= 0)
        |                                                       |
        +------------------ host Record* results --------------+
                         MergeJoinResult (existing merge)
```

The build phase is **identical** on both executors (no WAS either side); only the probe differs.

Two layers change:

1. **Engine** (`src/Kernel/KernelEngine_KernelSchedule/DLL_OpenCL`): gains the WAS kernel + device-fission infrastructure, ported from the handshaking engine. The CoProcessor links this engine library, so the port is the prerequisite (Phase 1 — see §10).
2. **CoProcessor** (`Co_Hj.cpp`): the `EXEC_CPU` worker invokes the WAS-enabled probe; the `EXEC_GPU` worker invokes the noWAS probe (Phase 2).

## 4. Components

### 4.1 Engine port (`src/Kernel/KernelEngine_KernelSchedule/DLL_OpenCL`)

The Kernel-tree `primitive.cl` currently contains **zero** WAS code (confirmed). Port from `src/handshaking/DLL_OpenCL`:

- **`primitive.cl`**: the `WASEntry` struct (32 B), the `post()` inline helper, `WAS_kernel` (fixed to **mode 3 = backward + spin-loop + no-pause**; variants are statistically indistinguishable on CPU), and the `wassize`-guarded WAS path inside `probe_kernel` (deferred-scan + the **end-flush drain**, which is correctness-critical for the deferred tail). `build_kernel` is unchanged (no WAS).
- **`common.cpp` / `common.h`**: `cl_init_prefetch` (device fission: 1-CU `PrefetchSubDevice` + 7-CU `MainCPUSubDevice` via `clCreateSubDevices(BY_COUNTS {1,7})`, plus the full 8-CU `FullCPUCommandQueue`) and `cl_cleanup_prefetch`. Bump `CL_TARGET_OPENCL_VERSION` to `200` (builds with deprecation warnings only).
- **Global definitions placement (build trap — do NOT copy the handshaking layout verbatim):** in the handshaking tree the prefetch globals (`PrefetchSubDevice`, `MainCPUSubDevice`, `PrefetchCommandQueue`, `FullCPUCommandQueue`, `g_prefetchEnabled`) are *defined* in `mainProgram.cpp`. But the Kernel CoProcessor links `libomnidb_opencl_nomain.a`, built by stripping `mainProgram.o` via `ar d`. If the definitions live in `mainProgram.cpp`, every reference from `common.cpp`/`Co_Hj.cpp` becomes an undefined-symbol link error. **Relocate the global definitions into `common.cpp`** (the TU that already holds `cl_init_prefetch` and survives the strip); keep only `extern` declarations in the header.
- Build `libomnidb_opencl.a` (and the `nomain.a` strip variant) as the Kernel CoProcessor expects (`make full-build`).

(Per the per-level fork rule, the same port may later be mirrored to the Operator/Query trees; out of scope here.)

### 4.2 Engine API

Add one exported entry the CoProcessor calls (engine DLL header):

```
int CL_hj_PE(Record *R, int rLen, Record *S, int sLen,
             Record **Rout, int CPU_GPU, int wassize);
```

- **Returns host `Record**`** (match output), matching the existing `CL_hj` / `tp_hj` merge contract (`tempRout` is a host `Record*` fed to `MergeJoinResult`). CL_hj_PE performs any device→host readback internally before returning, so `Co_Hj.cpp`'s merge path is reused unchanged. Returns the match count.
- **Device selection is governed solely by `CPU_GPU`** (a new, explicit convention for this function — it intentionally does **not** reuse the handshaking *sign-of-wassize* device convention). `wassize` only toggles WAS on/off within the chosen device.
- **Scope — delegate, don't re-implement:** for the **noWAS** case (`wassize <= 0`, either device), CL_hj_PE **delegates to the existing `CL_hj(R, rLen, S, sLen, Rout, CPU_GPU)`** (which already does build + probe + host readback for CPU_GPU ∈ {0,1}). CL_hj_PE adds **only** the new code path for `wassize > 0 && CPU_GPU == 0`: WAS/dummy buffer setup, `WAS_kernel` helper launch on `PrefetchCommandQueue`, the main `probe_kernel` (WAS path) on the 7-CU `MainCPUSubDevice` queue, helper stop via the ctrl flag, and readback. This bounds the new code to one path instead of re-implementing the join.
- **Launch geometry & WAS sizing (must be pinned):** the CPU WAS probe launches with a fixed work-group count `num_wg` and local size 256 (start from the handshaking harness geometry so the measured optimum transfers). `was_per_wg = wassize / num_wg`. The WAS buffer is `wassize × 32 B`; its **L3 footprint ≈ wassize × 160 B** (32 B struct + 2 × 64 B preloaded cache lines — the `kCostPerEntry` model). Initial `wassize = 1792–3584` (≈ 2.3–4.8 % of the 12 MB L3 at the handshaking geometry). **This constant is geometry-coupled**: if the CoProcessor per-partition S length forces a different `num_wg`, re-validate the effective per-WG footprint and re-tune (§9).

### 4.3 CoProcessor integration (`src/Kernel/CoProcessor/.../Co_Hj.cpp`)

- **Device-fission init:** invoke `cl_init_prefetch()` **from inside the engine's `EngineStart()`** (co-located with the existing `CommandQueue[0..1]` setup, runs exactly once before any operator). Do **not** add an init call in `Co_Hj.cpp`. Note: `cl_init_prefetch` **repartitions the global `CommandQueue[0]` to the 7-CU sub-device**; any CPU-OpenCL noWAS control must therefore route through `FullCPUCommandQueue` (8 CU) to remain a fair 8-CU baseline (as the handshaking harness does). Confirm no other Kernel-tree operator depends on `CommandQueue[0]` being full-device.
- **In `tp_hj`, key the probe path off the worker's `execMode`** (not the partition index it dequeues):
  - `EXEC_CPU` worker: replace the current OpenMP `CPU_hj(...)` call with `CL_hj_PE(..., CPU_GPU = 0, wassize = W)` — CPU-E with WAS.
  - `EXEC_GPU` worker: replace the current `GPUCopy_hj(...)`→`CL_hj(...,1)` call with `CL_hj_PE(..., CPU_GPU = 1, wassize = 0)` — GPU-E noWAS (`wassize = 0` cleanly means "no WAS"; the handshaking `wassize<0` "7-CU control" semantics do not apply on the GPU branch).
- Keying off `execMode` makes WAS-vs-noWAS correct **regardless of which partition index each thread steals** (if one device grabs both partitions, it runs its own mode on both — still correct). For balanced co-processing the existing 2-partition/2-thread dispatch normally yields one partition per device.
- **Gate** the whole PE path behind a flag (env `PE_MODE`, with a `wassize` override) so the baseline (`CPU_hj` + `CL_hj`) is preserved for A/B/C comparison and fallback. The `PE_MODE`/`wassize` knobs select the three test configs in §7.

Switching the CPU executor from OpenMP `CPU_hj` to the OpenCL `probe_kernel`-on-PoCL-CPU is an intended part of PE (WAS is an OpenCL mechanism). The A/B/C test design (§7) isolates this OpenMP→OpenCL change from the prefetch effect so the two are not conflated.

## 5. Data flow

1. Host hands `R`, `S` to `CO_hj`; the existing `GPUCopy_Partition` splits each into 2 partitions; both worker threads pull partitions from the shared work-stealing counter.
2. Each executor builds its hash table (identical `build_kernel`, no WAS either side).
3. **`EXEC_CPU` worker — probe WITH WAS:** main work-items post the upcoming bucket address per S tuple to the WAS; the helper (1 CU) prefetches those buckets into the shared L3; the main work-items process the deferred (lookahead) entries against the now-warm buckets; the **end-flush** drains the in-flight tail.
4. **`EXEC_GPU` worker — probe noWAS:** the iGPU runs the plain probe over its partition.
5. `CL_hj_PE` reads each partition's matches back to host `Record*`; results are concatenated via the existing `MergeJoinResult`.
6. The WAS helper terminates via its ctrl flag once the CPU probe finishes (Stage-B1-verified clean termination).

## 6. Error handling and correctness

- **Fission unavailable:** `cl_init_prefetch` guards on actual CU count and `clCreateSubDevices` success (the `{1,7}` split is hardcoded to the 8-CU host of record); on failure it sets `g_prefetchEnabled = 0`. CL_hj_PE then forces the noWAS delegation path for the CPU executor too. Graceful, never a hard failure.
- **GPU unavailable / GPU launch error:** the `EXEC_CPU` worker still runs (its `CL_hj_PE(CPU_GPU=0)`); if the GPU worker fails, the CPU worker processes the remaining partition(s) in its own mode. ("`EXEC_CPU`" here is the `EXEC_MODE` enum value, **not** a revival of the dormant burden/`speedupGPUoverCPU` machinery in §8.)
- **Correctness (the gate):** the WAS probe must produce **identical** matches to the noWAS probe. The deferred-tail **flush** is correctness-critical and must be ported intact. Verification asserts match-count parity (§7.1, §7.2).
- **Buffers:** per the WAS path, the **`probe_kernel` arguments** are `was_buffer`, `wassize`, `dummy_buffer`; **`last_tag_buffer` belongs to the `WAS_kernel` helper launch** (host-managed), not to `probe_kernel`. The `dummy_buffer` uses `CL_MEM_ALLOC_HOST_PTR` so the host/CPU-kernel sentinel addresses coincide — this works on the CPU because main and helper share the address space (and is precisely why it fails cross-device, hence GPU = noWAS). All WAS/dummy/last_tag buffers are per-call and freed after the helper stops.

## 7. Testing / verification

Three configs, selectable via `PE_MODE`/`wassize`, to **isolate** the prefetch effect from the OpenMP→OpenCL-CPU change:

- **(A) Baseline:** OpenMP `CPU_hj` (CPU executor) + `CL_hj` GPU. The current behavior.
- **(B) OpenCL-CPU noWAS:** `CL_hj_PE(CPU_GPU=0, wassize<=0)` (CPU executor) + GPU noWAS.
- **(C) PE:** `CL_hj_PE(CPU_GPU=0, wassize=W)` (CPU executor WAS) + GPU noWAS.

Measurements:

1. **Engine self-test (`exec`):** in the Kernel engine, probe-WAS vs probe-noWAS over the same data must yield the **same match count** (correctness parity), mirroring the handshaking Stage-A parity check. **This must pass before Phase 2 (§10).**
2. **CoProcessor correctness:** a hash-join query under config (C) must produce a join result identical to (A). This is the success gate.
3. **CoProcessor timing:** `CoProcessorApp <numQueries> <numThreads>` across (A)/(B)/(C). **B-vs-C isolates the prefetch effect; A-vs-B captures the OpenMP→OpenCL-CPU change.** Per §1, a speedup is not required — this is to observe, not to certify a win.
4. Reuse the `bin/release/KernelScheduler/test.sh` sweep once deployed.

## 8. Out of scope (YAGNI)

- **GPU-side prefetch** (posting-WAS or decoupled LLC-warming) — both empirically closed on this hardware (§2). GPU-E stays noWAS.
- **Dynamic CPU/GPU split ratio / cost model** (paper §4): keep the existing static work-stealing 2-partition dispatch; the dormant `speedupGPUoverCPU`/burden machinery is not revived.
- **D (decompression) and cPDE-c/b/g** configs: no decompression module exists; deferred.
- **Operator-level and Query-level trees**: Kernel tree only for now.
- **Auto-sizing `wassize`**: fixed (geometry-coupled) constant initially.

## 9. Open questions / future

- `wassize` re-tuning in the CoProcessor: the optimum is tied to the CPU probe's launch geometry (`num_wg`); re-validate the effective per-WG L3 footprint once the CoProcessor partition sizes are known, before trusting the `1792–3584` constant.
- Whether to pin partitions to executors (deterministic one-each) or keep work-stealing — correctness holds either way (WAS keys off `execMode`); pinning only affects load balance. Decide if measurement noise warrants it.
- Later: mirror the engine port to the Operator/Query trees if PE is wanted at those granularities.

## 10. Phasing (ordering constraint)

A single implementation plan, two ordered phases:

- **Phase 1 — Engine port:** §4.1 (WAS kernel + fission + global relocation + OpenCL 2.0) and §4.2 (`CL_hj_PE`). **Must compile, link the `nomain.a`, and pass the §7.1 engine self-test (WAS/noWAS match parity) before Phase 2.**
- **Phase 2 — CoProcessor wiring:** §4.3 (`EngineStart` fission init, `tp_hj` execMode-keyed `CL_hj_PE` calls, `PE_MODE` gate) and the §7.2/§7.3 A/B/C verification.
