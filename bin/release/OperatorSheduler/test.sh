#!/bin/bash

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
