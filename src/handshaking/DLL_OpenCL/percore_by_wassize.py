#!/usr/bin/env python3
"""Per-(wassize, variant) per-core perf measurement, aggregated to a clean
per-wassize CSV with main vs helper L3 miss averaged over 6 variants.

For each (wassize, variant): fresh perf-wrapped process (throttle-free), parse
per-core L3 miss/hit, identify helper = core with min L3-miss-rate, main = avg
over the other 7. Average across the 6 variants per wassize.

noWAS rows (wassize -1, 0): no helper, run once (variant-independent), main =
avg over active cores.

Backs up existing *_percore.csv to *_percore_byvariant.csv before rewriting.
"""
import subprocess, os, re, csv, time, shutil

HERE = "/home/jy/study/omnidb-paralleldbonintelproc/src/handshaking/DLL_OpenCL"
RES = HERE + "/results"
MODES = [0, 1, 2, 3, 6, 7]
WASSIZES = [-1, 0, 28, 56, 112, 224, 448, 896, 1792, 3584, 7168, 14336, 28672, 57344]
REPS = "30"  # 30 reps per (variant,wassize) perf run; balance accuracy vs runtime
EV = "mem_load_retired.l3_miss,mem_load_retired.l3_hit"

def parse_percore(f):
    miss, hit = {}, {}
    for ln in open(f):
        m = re.search(r"CPU(\d+)\s+([\d,]+)\s+(\S+)", ln)
        if not m: continue
        c = "CPU" + m.group(1); v = float(m.group(2).replace(",", ""))
        if m.group(3) == "mem_load_retired.l3_miss": miss[c] = v
        elif m.group(3) == "mem_load_retired.l3_hit": hit[c] = v
    return miss, hit

def run_perf(env_extra, outf):
    env = dict(os.environ, **env_extra)
    subprocess.run(["perf", "stat", "-A", "-C", "0-7", "-e", EV, "-o", outf, "./exec"],
                   cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def edit_kid(target):
    f = HERE + "/mainProgram.cpp"
    txt = open(f).read()
    new = re.sub(r"for \(kid = \d+; kid < \d+; kid\+\+\)",
                 f"for (kid = {target}; kid < {target + 1}; kid++)", txt)
    open(f, "w").write(new)
    r = subprocess.run(["make"], cwd=HERE, capture_output=True, text=True)
    if "Build complete" not in r.stdout + r.stderr:
        print(f"[BUILD FAIL kid={target}]\n{r.stdout[-500:]}\n{r.stderr[-500:]}", flush=True)
        raise SystemExit(1)

def collect(kernel, only_env, mode_env, reps_env):
    """Return list of (wassize, main_l3_miss_avg, helper_l3_miss_avg)."""
    out = []
    for w in WASSIZES:
        var_main, var_helper = [], []
        modes_to_run = [3] if w <= 0 else MODES  # noWAS: variant irrelevant
        for m in modes_to_run:
            outf = f"/tmp/byw_{kernel}_w{w}_m{m}.txt"
            extra = {only_env: str(w), reps_env: REPS}
            if kernel == "probe":
                extra["PROBE_ROUNDS"] = "1"
            if w > 0:
                extra[mode_env] = str(m)
            run_perf(extra, outf)
            miss, hit = parse_percore(outf)
            if not miss: continue
            if w > 0 and len(miss) >= 2:
                # helper = lowest L3-miss-rate core
                rates = {c: miss[c] / (miss[c] + hit.get(c, 0))
                         if (miss[c] + hit.get(c, 0)) > 0 else 1.0 for c in miss}
                helper = min(rates, key=rates.get)
                var_helper.append(miss[helper])
                others = [miss[c] for c in miss if c != helper]
                var_main.append(sum(others) / len(others))
            else:
                # noWAS: no helper. main = avg over all active cores (>0 miss).
                active = [v for v in miss.values() if v > 0]
                var_main.append(sum(active) / len(active) if active else 0)
        main_avg = sum(var_main) / len(var_main) if var_main else 0
        helper_avg = sum(var_helper) / len(var_helper) if var_helper else 0
        out.append((w, main_avg, helper_avg))
        print(f"  [{kernel} w={w}] main_l3_miss={main_avg/1e6:.1f}M  "
              f"helper_l3_miss={helper_avg/1e6:.1f}M (n_var={len(var_main)})", flush=True)
    return out

def write_csv(kernel, rows):
    f = f"{RES}/{kernel}_percore.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wassize", "main_l3_miss_avg", "helper_l3_miss_avg", "notes"])
        for ws, m, h in rows:
            if ws == -1:
                note = "noWAS7c — base (main only, 7-CU sub-device, no helper)"
                h_s = ""
            elif ws == 0:
                note = "noWAS8c — base (main only, 8-CU parent device, no helper)"
                h_s = ""
            else:
                note = "avg over 6 helper variants (modes 0,1,2,3,6,7)"
                h_s = f"{h:.0f}"
            w.writerow([ws, f"{m:.0f}", h_s, note])
    print(f"  wrote {f}", flush=True)

t0 = time.time()
# Preserve existing per-variant CSV only if no byvariant backup exists yet.
for k in ("filter", "probe"):
    src = f"{RES}/{k}_percore.csv"
    dst = f"{RES}/{k}_percore_byvariant.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)
        print(f"[backup] {k}_percore.csv -> {k}_percore_byvariant.csv", flush=True)
    else:
        print(f"[backup] skip ({k}_percore_byvariant.csv already exists)", flush=True)

# Filter half
edit_kid(20)
print(f"[filter] kid=20 built @ {time.time()-t0:.0f}s", flush=True)
filter_rows = collect("filter", "FILTER_ONLY_W", "FILTER_MODE", "FILTER_REPS")
write_csv("filter", filter_rows)
print(f"[filter] done @ {time.time()-t0:.0f}s", flush=True)

# Probe half
edit_kid(50)
print(f"[probe] kid=50 built @ {time.time()-t0:.0f}s", flush=True)
probe_rows = collect("probe", "PROBE_ONLY_W", "PROBE_MODE", "PROBE_REPS")
write_csv("probe", probe_rows)
print(f"[probe] done @ {time.time()-t0:.0f}s", flush=True)
print(f"=== DONE @ {time.time()-t0:.0f}s ===", flush=True)
