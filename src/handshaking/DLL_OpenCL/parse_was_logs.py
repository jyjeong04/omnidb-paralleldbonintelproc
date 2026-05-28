#!/usr/bin/env python3
"""Parse all run_*.log into one long CSV + a wide pivot (mean avg_ms over runs).

Recognised filenames:
  run_high_fs_<i>.log         -> delta=28672, variant=fs (already-done run)
  run_<vc>_d<delta>_<i>.log   -> vc in {fs,fn,bs,bn}, delta from name
Sweep line parsed:
  [KID20-SWEEP ...] ... d= <delta> w= <wassize> L3= ..% avg= <ms>ms hits=<n>
"""
import os, re, csv, glob, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WGSIZE = 256
LINE = re.compile(
    r"d=\s*(\d+)\s+w=\s*(\d+)\s+L3=\s*[\d.]+%\s+avg=\s*([\d.]+)ms\s+hits=(\d+)")
NAME_NEW = re.compile(r"run_(fs|fn|bs|bn)_d(\d+)_(\d+)\.log$")
NAME_OLD = re.compile(r"run_high_fs_(\d+)\.log$")

def classify(fn):
    m = NAME_NEW.search(fn)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = NAME_OLD.search(fn)
    if m:
        return "fs", 28672, int(m.group(1))
    return None

def main():
    rows = []          # long-format records
    for path in sorted(glob.glob(os.path.join(HERE, "run_*.log"))):
        info = classify(os.path.basename(path))
        if not info:
            continue
        vc, _fname_delta, run = info
        direction = "forward" if vc[0] == 'f' else "backward"
        spin = "spin" if vc[1] == 's' else "nospin"
        for ln in open(path, errors="replace"):
            m = LINE.search(ln)
            if not m:
                continue
            # delta is taken from the line's d= column (authoritative): a single
            # run log may mix deltas (e.g. d=114688 WAS cfgs + d=14336/131072 noWAS).
            d, w, avg, hits = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
            mode = "noWAS" if w == 0 else "WAS"
            rows.append(dict(delta=d, variant=vc, direction=direction, spin=spin,
                             mode=mode, wassize=w,
                             was_per_wg=(w // (d // WGSIZE)) if w > 0 else 0,
                             run=run, avg_ms=avg, hits=hits))

    long_csv = os.path.join(HERE, "was_experiments_long.csv")
    with open(long_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["delta", "variant", "direction", "spin",
                                           "mode", "wassize", "was_per_wg", "run", "avg_ms", "hits"])
        wr.writeheader()
        wr.writerows(rows)

    # wide pivot: (delta, variant) x wassize -> mean avg_ms
    agg = defaultdict(list)
    wsizes = sorted({r["wassize"] for r in rows})
    for r in rows:
        agg[(r["delta"], r["variant"], r["wassize"])].append(r["avg_ms"])
    keys = sorted({(r["delta"], r["variant"]) for r in rows})
    wide_csv = os.path.join(HERE, "was_experiments_wide.csv")
    with open(wide_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["delta", "variant", "n_runs"] + [f"w{w}" for w in wsizes])
        for (delta, vc) in keys:
            nrun = max((len(agg[(delta, vc, w)]) for w in wsizes
                        if (delta, vc, w) in agg), default=0)
            row = [delta, vc, nrun]
            for w in wsizes:
                vals = agg.get((delta, vc, w))
                row.append(f"{statistics.mean(vals):.3f}" if vals else "")
            wr.writerow(row)

    print(f"long: {long_csv}  ({len(rows)} rows)")
    print(f"wide: {wide_csv}  ({len(keys)} (delta,variant) groups, {len(wsizes)} wassize cols)")
    # quick console preview of the wide table
    print("\n(delta, variant)         n  " + "  ".join(f"{w:>7}" for w in wsizes))
    for (delta, vc) in keys:
        cells = []
        for w in wsizes:
            vals = agg.get((delta, vc, w))
            cells.append(f"{statistics.mean(vals):7.2f}" if vals else "      .")
        nrun = max((len(agg[(delta, vc, w)]) for w in wsizes if (delta, vc, w) in agg), default=0)
        print(f"d={delta:<6} {vc:<6} {nrun:>3}  " + "  ".join(cells))

if __name__ == "__main__":
    main()
