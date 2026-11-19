#!/bin/bash
# Architecture and recipe probes on in-distribution data: every variant from scratch, one pass over the same 1.02 billion
# Lichess positions (shards h0-h59, held in memory), judged on the held-out million (ival). Runs on the match machine's
# graphics card while its processors play. One line per variant in probe.log.
#   bash probe_q.sh "<name>|<trainer options>" ...
cd /root/loopchess; tags=$(seq -s, -f "h%g" 0 59)
for spec in "$@"; do
  n=${spec%%|*}; o=${spec#*|}; t0=$(date +%s)
  python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tags --val ival --epochs 1 --threads 4 --device cuda --pst zero/g101s.npz --out /dev/shm/p_$n.bin $o > /dev/shm/p_$n.log 2>&1
  r=$(grep "^done" /dev/shm/p_$n.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  echo "[$(date -u +%H:%M:%S)] $n ($o): ${r:-FAILED: $(tail -1 /dev/shm/p_$n.log | cut -c1-120)}; $(( $(date +%s) - t0 ))s, $(grep '^step' /dev/shm/p_$n.log | tail -1 | grep -o '[0-9]*k pos/s')" >> probe.log
  rm -f /dev/shm/p_$n.bin.pt
done
echo "[$(date -u +%H:%M:%S)] probes finished" >> probe.log
