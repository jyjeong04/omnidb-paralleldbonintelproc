#!/usr/bin/env python3
"""probe (kid=50) d=1792 throttle-free sweep, mirroring the filter experiment:
6 helper variants, each in its own fresh process (PROBE_MODE), each running
noWAS7c/noWAS8c + WAS wassize 28..57344 at PROBE_ROUNDS=1 PROBE_REPS=100
(build table once, 100 reps per cfg). Two passes:
  pass 1 -> elapsed-time CSV (/tmp/probe_timing.csv)
  pass 2 -> per-core cache-miss CSV via perf -A -C 0-7 (/tmp/probe_percore.csv)
NOTE: requires mainProgram kid loop set to 50 (probe) and a rebuilt ./exec.
"""
import subprocess, os, re, csv, time

HERE = "/home/jy/study/omnidb-paralleldbonintelproc/src/handshaking/DLL_OpenCL"
MODES = [0, 1, 2, 3, 6, 7]
MLABEL = {0: "F-nospin", 1: "B-nospin", 2: "F-spin", 3: "B-spin",
          6: "F-spin-pz", 7: "B-spin-pz"}
ENVBASE = {"PROBE_ROUNDS": "1", "PROBE_REPS": "100"}
EVENTS = "mem_load_retired.l3_miss,mem_load_retired.l3_hit,cycles"
t0 = time.time()

# probe per-cfg line:
# [KID50-PROBE  1/14] d1792 noWAS7c   d=  1792 w=  -1 L3= 0.0% avg= 159.72 ± 1.01 cv= 0.6% min= 158.00 n=50 hits=0
LINE = re.compile(r"\]\s+(.*?)\s+d=\s*\d+\s+w=\s*(-?\d+).*?avg=\s*([\d.]+)\s*"
                  r"\xb1\s*([\d.]+)\s*cv=[^m]*min=\s*([\d.]+).*?hits=(\d+)")

# ---- Pass 1: elapsed time (clean, no perf) ----
with open("/tmp/probe_timing.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "variant", "wassize", "avg_ms", "std_ms", "min_ms", "hits"])
    for m in MODES:
        env = dict(os.environ, PROBE_MODE=str(m), **ENVBASE)
        r = subprocess.run(["./exec"], cwd=HERE, env=env,
                           capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            mt = LINE.search(ln)
            if mt:
                w.writerow([m, MLABEL[m], mt.group(2), mt.group(3),
                            mt.group(4), mt.group(5), mt.group(6)])
        f.flush()
        print(f"[timing] mode {m} ({MLABEL[m]}) done @ {time.time()-t0:.0f}s",
              flush=True)

# ---- Pass 2: per-core cache miss via perf (same conditions) ----
with open("/tmp/probe_percore.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "variant", "cpu", "event", "count"])
    for m in MODES:
        env = dict(os.environ, PROBE_MODE=str(m), **ENVBASE)
        outf = f"/tmp/ppc_m{m}.txt"
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
