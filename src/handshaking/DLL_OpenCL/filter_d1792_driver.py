#!/usr/bin/env python3
"""filter (kid=20) d=1792 throttle-free sweep: 6 helper variants, each in its
own fresh process (FILTER_MODE), each running noWAS7c/noWAS8c + WAS wassize
28..57344 at FILTER_REPS=100. Two passes:
  pass 1 -> elapsed-time CSV (/tmp/filter_timing.csv)
  pass 2 -> per-core cache-miss CSV via perf -A -C 0-7 (/tmp/filter_percore.csv)
Each cfg/variant runs in a separate cold process so CPU turbo/throttle does not
drift across the sweep.
"""
import subprocess, os, re, csv, time

HERE = "/home/jy/study/omnidb-paralleldbonintelproc/src/handshaking/DLL_OpenCL"
MODES = [0, 1, 2, 3, 6, 7]
MLABEL = {0: "F-nospin", 1: "B-nospin", 2: "F-spin", 3: "B-spin",
          6: "F-spin-pz", 7: "B-spin-pz"}
REPS = "100"
EVENTS = "mem_load_retired.l3_miss,mem_load_retired.l3_hit,cycles"
t0 = time.time()

# ---- Pass 1: elapsed time (clean, no perf) ----
with open("/tmp/filter_timing.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "variant", "wassize", "avg_ms", "hits"])
    for m in MODES:
        env = dict(os.environ, FILTER_MODE=str(m), FILTER_REPS=REPS)
        r = subprocess.run(["./exec"], cwd=HERE, env=env,
                           capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            mt = re.search(r"\]\s+(.*?)\s+d=\s*\d+\s+w=\s*(-?\d+).*?"
                           r"avg=\s*([\d.]+)ms\s+hits=(\d+)", ln)
            if mt:
                w.writerow([m, MLABEL[m], mt.group(2), mt.group(3), mt.group(4)])
        f.flush()
        print(f"[timing] mode {m} ({MLABEL[m]}) done @ {time.time()-t0:.0f}s",
              flush=True)

# ---- Pass 2: per-core cache miss via perf (same conditions) ----
with open("/tmp/filter_percore.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "variant", "cpu", "event", "count"])
    for m in MODES:
        env = dict(os.environ, FILTER_MODE=str(m), FILTER_REPS=REPS)
        outf = f"/tmp/fpc_m{m}.txt"
        subprocess.run(["perf", "stat", "-A", "-C", "0-7", "-e", EVENTS,
                        "-o", outf, "./exec"], cwd=HERE, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for ln in open(outf):
            mt = re.search(r"CPU(\d+)\s+([\d,]+)\s+(\S+)", ln)
            if mt:
                w.writerow([m, MLABEL[m], "CPU" + mt.group(1), mt.group(3),
                            mt.group(2).replace(",", "")])
        f.flush()
        print(f"[perf] mode {m} ({MLABEL[m]}) done @ {time.time()-t0:.0f}s",
              flush=True)

print(f"=== DONE @ {time.time()-t0:.0f}s ===", flush=True)
