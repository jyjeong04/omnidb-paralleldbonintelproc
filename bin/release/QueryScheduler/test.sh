#!/bin/bash
# If CoProcessorApp hangs with multiple threads, try threadnum=1 first.
# GDB showed the hang is in pocl_level0_wait_event (OpenCL GPU); reducing
# concurrency (e.g. ./CoProcessorApp 15 1) may avoid the hang.

count=0
querynum=15
threadnum=2

while [ "$count" -le 0 ]; do
    threadnum=2
    while [ "$threadnum" -le 8 ]; do
        querynum=15
        while [ "$querynum" -le 30 ]; do
            ./CoProcessorApp "$querynum" "$threadnum"
            querynum=$((querynum + 5))
        done
        threadnum=$((threadnum + 2))
    done
    count=$((count + 1))
done
