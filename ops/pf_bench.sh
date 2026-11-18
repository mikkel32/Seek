#!/bin/bash
# Speed of the engine with the input-matrix rows of each move requested from memory at make time (-DPFROWS=<cache lines per row>):
# alone, and with 76 copies running at once (the load of a match). Positions per second, summed over the copies.
cd /root/loopchess
for pf in 0 4 12 24 48; do f=""; [ $pf != 0 ] && f="-DPFROWS=$pf"; gcc -O3 -march=native -funroll-loops -DHMAX=1536 $f -o calc/t_pf$pf calc/calc_s19_pf.c -lm; done
one() { python3 zero/bench.py $1 $2 ${3:-14} | grep -o "= [0-9]*k" | grep -o "[0-9]*"; }
export -f one
for n in w_BIGX w_BIGM; do
  for pf in 0 4 12 24 48; do
    a=$(one calc/t_pf$pf zero/$n.bin 15); b=$(one calc/t_pf$pf zero/$n.bin 15)
    l=$(seq 76 | xargs -P 76 -I{} bash -c "one calc/t_pf$pf zero/$n.bin 15" | awk '{s+=$1} END {print s}')
    echo "[$(date -u +%H:%M:%S)] $n, rows requested ahead: $pf lines; alone ${a}k and ${b}k per second; 76 copies at once: ${l}k per second in total" >> pf_bench.log
  done
done
echo "[$(date -u +%H:%M:%S)] finished" >> pf_bench.log
