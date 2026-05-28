# WAS-prefetch experiment results — i7-9700 (no-SMT, PoCL CPU)

All measurements: delta=1792 (7 work-groups), wassize 28..57344 (doubling),
6 helper variants × {noWAS7c, noWAS8c}, FILTER_REPS / PROBE_REPS=100,
each variant in its own fresh process (throttle-free), `perf_event_paranoid=0`.

## Headline CSVs (analysis-ready)
- `filter_timing.csv`    — filter (kid=20) elapsed-time per (variant,wassize).
  cols: mode, variant, wassize, avg_ms, hits.
- `filter_percore.csv`   — filter per-core L3 miss/hit/cycles per variant.
  cols: mode, variant, cpu, event, count.
- `probe_timing.csv`     — probe (kid=50) elapsed-time (avg+std+min) per (variant,wassize).
  cols: mode, variant, wassize, avg_ms, std_ms, min_ms, hits.
- `probe_percore.csv`    — probe per-core L3 miss/hit/cycles per variant.

## Logs
- `filter_driver.log`, `probe_driver.log` — per-variant progress timestamps.

## raw_perf/
Raw `perf stat -A -C 0-7` output per variant (variant-aggregated), pre-CSV.
- `fpc_m{0,1,2,3,6,7}.txt` — filter, helper modes 0..7.
- `ppc_m{0,1,2,3,6,7}.txt` — probe.

mode bits: 0=backward, 1=spin, 2=pause (set), 3=64B-wide, 4=no-deref.
labels: 0=F-nospin 1=B-nospin 2=F-spin 3=B-spin 6=F-spin-pause 7=B-spin-pause.

## per_launch_extrapolation/
Earlier per-wassize perf runs used for the two-point extrapolation
((C50-C1)/49) that isolated per-probe cache miss from process overhead.
- `pp_w<W>_<rep>.csv`  — probe REPS=50.
- `pp1_w<W>_<rep>.csv` — probe REPS=1.
- `ff_*_*.csv`         — filter equivalents.
