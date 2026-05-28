#!/usr/bin/env python3
"""WAS helper-kernel experiment driver.

Sweeps delta in {28672, 57344, 114688} x variant in {fs, fn, bs, bn}.
 - direction: f(orward, idx++) / b(ackward, idx--)
 - spin     : s(pin loop on unchanged slot) / n(o spin, just advance)
For each (delta, variant) it regenerates the WAS_kernel (primitive.cl, JIT'd by
PoCL at runtime) and the cfgs[] sweep (Handshake.cpp, needs make), rebuilds, and
runs ./exec N times, saving run_<variant>_d<delta>_<i>.log.

delta=28672 + fs is already done (run_high_fs_*.log) and is skipped.
"""
import os, re, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PRIM = os.path.join(HERE, "primitive.cl")
HAND = os.path.join(HERE, "Handshake.cpp")

# Absolute wassize columns (shared with the old delta=14336 experiment).
WASSIZE_SET = [56, 112, 224, 448, 1792, 3584, 7168, 14336, 28672, 57344]
WGSIZE = 256          # numThreadsPerBlock_x in the sweep
WASSIZE_MAX = 57344
# noWAS baselines (wassize=0 -> helper not launched, runs on full 8 CUs).
NOWAS_BASELINES = [(14336, 0), (131072, 0)]

# 11 remaining combos (28672-fs excluded — already done).
COMBOS = [
    (28672, 'b', True), (28672, 'b', False), (28672, 'f', False),
    (57344, 'f', True), (57344, 'f', False), (57344, 'b', True), (57344, 'b', False),
    (114688, 'f', True), (114688, 'f', False), (114688, 'b', True), (114688, 'b', False),
]

def variant_code(direction, spin):
    return ("f" if direction == 'f' else "b") + ("s" if spin else "n")

def adv(direction):
    return "(idx == wassize - 1) ? (0) : (idx + 1)" if direction == 'f' \
        else "(idx == 0) ? (wassize - 1) : (idx - 1)"

def init_idx(direction):
    return "0" if direction == 'f' else "wassize - 1"

def gen_kernel(direction, spin):
    a = adv(direction)
    if spin:
        unchanged = (
            "    if (cur == last_tag[idx]) {\n"
            "      do {\n"
            "        if (ctrl[1] != 0) {\n"
            "          ctrl[0] += count;\n"
            "          return;\n"
            "        }\n"
            "        CPU_PAUSE();\n"
            "        cur = (ulong)(uintptr_t)was_buffer[idx].p1;\n"
            "      } while (cur == last_tag[idx]);\n"
            "      if (cur == dummyVal) {\n"
            f"        idx = {a};\n"
            "        continue;\n"
            "      }\n"
            "    }\n"
        )
    else:
        unchanged = (
            "    if (cur == last_tag[idx]) {\n"
            f"      idx = {a};\n"
            "      continue;\n"
            "    }\n"
        )
    return (
        "//>>>WASK_BEGIN\n"
        "__kernel void WAS_kernel(volatile __global WASEntry *was_buffer,\n"
        "                         const int wassize, volatile __global uint *ctrl,\n"
        "                         __global ulong *last_tag) {\n"
        "  volatile uint temp = 0;\n"
        "  uint count = 0;\n"
        f"  int idx = {init_idx(direction)};\n"
        "  const ulong dummyVal = (ulong)(uintptr_t)ctrl;\n"
        "\n"
        "  while (ctrl[1] == 0) {\n"
        "    ulong cur = (ulong)(uintptr_t)was_buffer[idx].p1;\n"
        "\n"
        "    if (cur == dummyVal) {\n"
        f"      idx = {a};\n"
        "      continue;\n"
        "    }\n"
        "\n"
        f"{unchanged}"
        "\n"
        "    __global uint *p1 = was_buffer[idx].p1;\n"
        "    __global uint *p2 = was_buffer[idx].p2;\n"
        "    temp += *was_buffer[idx].p1;\n"
        "    temp += *was_buffer[idx].p2;\n"
        "    last_tag[idx] = cur;\n"
        "    count++;\n"
        f"    idx = {a};\n"
        "  }\n"
        "  ctrl[0] += count;\n"
        "}\n"
        "//<<<WASK_END"
    )

def gen_cfgs(delta, baselines=False):
    num_wg = delta // WGSIZE
    sizes = [w for w in WASSIZE_SET if w >= num_wg and w <= WASSIZE_MAX]
    body = ["      //>>>CFGS_BEGIN"]
    for w in sizes:
        body.append(f'      {{"M1 d={delta} w={w}", {delta}, {w}}},')
    if baselines:
        for bd, bw in NOWAS_BASELINES:
            body.append(f'      {{"noWAS d={bd} w={bw}", {bd}, {bw}}},')
    body.append("      //<<<CFGS_END")
    return "\n".join(body), sizes

def gen_cfgs_explicit(cfg_list):
    """cfg_list: [(label, delta, wassize), ...] -> cfgs block text."""
    body = ["      //>>>CFGS_BEGIN"]
    for label, d, w in cfg_list:
        body.append(f'      {{"{label}", {d}, {w}}},')
    body.append("      //<<<CFGS_END")
    return "\n".join(body)


def replace_block(path, begin, end, new_text):
    # Tolerant of marker indentation / a marker glued onto a code line by a
    # manual edit: consume optional leading whitespace and match markers anywhere.
    s = open(path).read()
    pat = re.compile(r"[ \t]*" + re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pat.search(s):
        sys.exit(f"ERROR: markers {begin}..{end} not found in {path}")
    s2 = pat.sub(lambda _: new_text, s, count=1)
    open(path, "w").write(s2)

def apply_combo(delta, direction, spin, baselines=False):
    replace_block(PRIM, "//>>>WASK_BEGIN", "//<<<WASK_END", gen_kernel(direction, spin))
    cfgs_text, sizes = gen_cfgs(delta, baselines)
    replace_block(HAND, "//>>>CFGS_BEGIN", "//<<<CFGS_END", cfgs_text)
    return sizes

def build():
    r = subprocess.run(["make"], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0 or "Build complete" not in r.stdout:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        sys.exit("ERROR: build failed")

def run_once(logpath):
    with open(logpath, "w") as f:
        subprocess.run(["./exec"], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        delta, direction, spin = 57344, 'b', True   # bs/57344: exercises kernel+cfgs regen
        vc = variant_code(direction, spin)
        sizes = apply_combo(delta, direction, spin)
        print(f"[smoke] combo d={delta} {vc} sizes={sizes}", flush=True)
        build()
        t0 = time.time()
        log = os.path.join(HERE, f"run_smoke_{vc}_d{delta}.log")
        run_once(log)
        print(f"[smoke] one run took {time.time()-t0:.1f}s -> {log}", flush=True)
        return
    if mode == "perf_wassize":
        # Cache-miss vs WAS size at d=114688 (fs kernel; variants are noise).
        # One wassize per build, wrapped in perf stat. init is constant across
        # wassizes -> the cross-wassize trend isolates the WAS/cache effect;
        # w=0 (noWAS) included as the Delta baseline.
        # REQUIRES: sudo sysctl -w kernel.perf_event_paranoid=1  (run by user).
        reps = int(os.environ.get("REPS", "3"))
        # -1 = noWAS 7-core control (no queue swap); 0 = noWAS 8-core; >0 = WAS
        wsizes = [-1, 0, 448, 1792, 3584, 7168, 14336, 28672, 57344]
        events = ("mem_load_retired.l3_miss,mem_load_retired.l3_hit,"
                  "mem_load_retired.l2_miss,cycle_activity.stalls_l3_miss,"
                  "instructions,cycles")
        replace_block(PRIM, "//>>>WASK_BEGIN", "//<<<WASK_END", gen_kernel('f', True))
        t_start = time.time()
        for w in wsizes:
            label = f"perf d=114688 w={w}"
            replace_block(HAND, "//>>>CFGS_BEGIN", "//<<<CFGS_END",
                          gen_cfgs_explicit([(label, 114688, w)]))
            build()
            for rep in range(1, reps + 1):
                out = os.path.join(HERE, f"perf_w{w}_{rep}.csv")
                runlog = os.path.join(HERE, f"perf_w{w}_{rep}.runlog")
                with open(runlog, "w") as f:
                    subprocess.run(["perf", "stat", "-e", events, "-x", ",",
                                    "-o", out, "./exec"],
                                   cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
                print(f"  w={w} rep{rep}: {time.time()-t_start:.0f}s elapsed", flush=True)
        apply_combo(28672, 'f', True)  # restore
        build()
        print(f"=== DONE perf_wassize in {(time.time()-t_start)/60:.1f}m ===", flush=True)
        return
    if mode == "nowas_baselines":
        # noWAS (wassize=0, 8-CU) at the same deltas as the WAS sweeps, so each
        # delta gets a same-delta WAS-vs-noWAS pair. Variants are noise -> fs only.
        # All 3 cfgs run in one sweep; 10 runs -> run_fs_d0_<i>.log (line-delta authoritative).
        runs = int(os.environ.get("RUNS", "10"))
        cfg_list = [("noWAS d=28672 w=0", 28672, 0),
                    ("noWAS d=57344 w=0", 57344, 0),
                    ("noWAS d=114688 w=0", 114688, 0)]
        replace_block(PRIM, "//>>>WASK_BEGIN", "//<<<WASK_END", gen_kernel('f', True))
        replace_block(HAND, "//>>>CFGS_BEGIN", "//<<<CFGS_END", gen_cfgs_explicit(cfg_list))
        build()
        t_start = time.time()
        for i in range(1, runs + 1):
            log = os.path.join(HERE, f"run_fs_d0_{i}.log")
            t0 = time.time()
            run_once(log)
            el = time.time() - t_start
            eta = el / i * (runs - i)
            print(f"  [{i}/{runs}] noWAS-baselines run {i}: {time.time()-t0:.1f}s "
                  f"(elapsed {el/60:.1f}m, eta {eta/60:.1f}m)", flush=True)
        apply_combo(28672, 'f', True)  # restore original state
        build()
        print(f"=== DONE nowas_baselines in {(time.time()-t_start)/60:.1f}m; restored fs/28672 ===", flush=True)
        return
    if mode == "rerun114688":
        # Re-run d=114688 for all 4 variants with 2 noWAS baselines appended
        # (d=14336 w=0, d=131072 w=0). Overwrites run_<vc>_d114688_<i>.log.
        runs = int(os.environ.get("RUNS", "10"))
        variants = [('f', True), ('f', False), ('b', True), ('b', False)]
        total = len(variants) * runs
        done = 0
        t_start = time.time()
        for direction, spin in variants:
            vc = variant_code(direction, spin)
            sizes = apply_combo(114688, direction, spin, baselines=True)
            print(f"=== d=114688 {vc} +noWAS{NOWAS_BASELINES} sizes={sizes} ===", flush=True)
            build()
            for i in range(1, runs + 1):
                log = os.path.join(HERE, f"run_{vc}_d114688_{i}.log")
                t0 = time.time()
                run_once(log)
                done += 1
                el = time.time() - t_start
                eta = el / done * (total - done)
                print(f"  [{done}/{total}] {vc} run {i}: {time.time()-t0:.1f}s "
                      f"(elapsed {el/60:.1f}m, eta {eta/60:.1f}m)", flush=True)
        apply_combo(28672, 'f', True)  # restore original state
        build()
        print(f"=== DONE rerun114688 in {(time.time()-t_start)/60:.1f}m; restored fs/28672 ===", flush=True)
        return
    # full
    runs = int(os.environ.get("RUNS", "10"))
    total = len(COMBOS) * runs
    done = 0
    t_start = time.time()
    for delta, direction, spin in COMBOS:
        vc = variant_code(direction, spin)
        sizes = apply_combo(delta, direction, spin)
        print(f"=== combo d={delta} {vc} sizes={sizes} ===", flush=True)
        build()
        for i in range(1, runs + 1):
            log = os.path.join(HERE, f"run_{vc}_d{delta}_{i}.log")
            t0 = time.time()
            run_once(log)
            done += 1
            el = time.time() - t_start
            eta = el / done * (total - done)
            print(f"  [{done}/{total}] {vc} d={delta} run {i}: "
                  f"{time.time()-t0:.1f}s  (elapsed {el/60:.1f}m, eta {eta/60:.1f}m)",
                  flush=True)
    # restore original state (fs, delta=28672)
    apply_combo(28672, 'f', True)
    build()
    print(f"=== DONE all combos in {(time.time()-t_start)/60:.1f}m; source restored to fs/28672 ===", flush=True)

if __name__ == "__main__":
    main()
