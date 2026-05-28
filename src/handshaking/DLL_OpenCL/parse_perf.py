#!/usr/bin/env python3
"""Aggregate perf_w<w>_<rep>.csv (perf stat -x ,) into per-WAS-size cache metrics.

perf -x , line layout: count,unit,event,run_time,run_pct,metric,metric_unit
Derived per wassize (mean over reps):
  L3 miss ratio = l3_miss / (l3_miss + l3_hit)
  L3 MPKI       = l3_miss / instructions * 1000
  L2 MPKI       = l2_miss / instructions * 1000
  L3-stall %    = stalls_l3_miss / cycles * 100
w=0 is the noWAS baseline; Delta columns are vs w=0.
"""
import os, re, csv, glob, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = re.compile(r"perf_w(-?\d+)_(\d+)\.csv$")  # -1 = noWAS 7-core control
WANT = ["mem_load_retired.l3_miss", "mem_load_retired.l3_hit",
        "mem_load_retired.l2_miss", "cycle_activity.stalls_l3_miss",
        "instructions", "cycles"]

def parse_file(path):
    counts = {}
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(",")
        if len(parts) < 3:
            continue
        val, _unit, ev = parts[0], parts[1], parts[2]
        if ev in WANT:
            try:
                counts[ev] = float(val)
            except ValueError:
                pass  # <not counted>/<not supported>
    return counts

def main():
    per_w = defaultdict(lambda: defaultdict(list))  # w -> event -> [counts]
    for path in sorted(glob.glob(os.path.join(HERE, "perf_w*_*.csv"))):
        m = NAME.search(os.path.basename(path))
        if not m:
            continue
        w = int(m.group(1))
        for ev, c in parse_file(path).items():
            per_w[w][ev].append(c)

    if not per_w:
        print("no perf_w*.csv found (run: python3 run_was_experiments.py perf_wassize)")
        return

    def mean(w, ev):
        v = per_w[w].get(ev, [])
        return st.mean(v) if v else float("nan")

    rows = []
    for w in sorted(per_w):
        l3m, l3h = mean(w, "mem_load_retired.l3_miss"), mean(w, "mem_load_retired.l3_hit")
        l2m = mean(w, "mem_load_retired.l2_miss")
        stall = mean(w, "cycle_activity.stalls_l3_miss")
        ins, cyc = mean(w, "instructions"), mean(w, "cycles")
        rows.append(dict(
            wassize=w, n=len(per_w[w].get("cycles", [])),
            l3_miss=l3m, l3_hit=l3h, l2_miss=l2m,
            l3_miss_ratio=l3m / (l3m + l3h) if (l3m + l3h) else float("nan"),
            l3_mpki=l3m / ins * 1000 if ins else float("nan"),
            l2_mpki=l2m / ins * 1000 if ins else float("nan"),
            stall_pct=stall / cyc * 100 if cyc else float("nan"),
            instructions=ins, cycles=cyc))

    out = os.path.join(HERE, "perf_wassize.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    base = next((r for r in rows if r["wassize"] == 0), None)
    print(f"out: {out}  ({len(rows)} wassizes)\n")
    hdr = f"{'wassize':>8} {'n':>2} {'L3_miss':>12} {'L3miss%':>8} {'L3_MPKI':>8} {'L2_MPKI':>8} {'L3stall%':>9}"
    if base:
        hdr += f" {'dL3MPKI':>8}"
    print(hdr)
    for r in rows:
        line = (f"{r['wassize']:>8} {r['n']:>2} {r['l3_miss']:>12.3e} "
                f"{r['l3_miss_ratio']*100:>7.2f}% {r['l3_mpki']:>8.3f} "
                f"{r['l2_mpki']:>8.3f} {r['stall_pct']:>8.2f}%")
        if base and base["l3_mpki"] == base["l3_mpki"]:
            line += f" {r['l3_mpki']-base['l3_mpki']:>+8.3f}"
        print(line)
    if base:
        print("\n(dL3MPKI = L3 MPKI relative to w=0 noWAS baseline; +이면 WAS가 L3 miss 늘림)")

if __name__ == "__main__":
    main()
