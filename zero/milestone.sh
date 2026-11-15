#!/bin/bash
# A training run from scratch on everything the long-memory machine holds, once it holds N shards:
#   bash zero/milestone.sh <N shards> <one-letter name, not b or l> [passes=8]
# The shards are frozen under their own names (hard links), so production may go on deleting and adding meanwhile.
# Two networks: the plain one, and the same with the cheap layered read-out (layers reading 64 units a side).
cd /root/loopchess; N=$1; T=$2; P=${3:-8}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> milestone.log; }
while [ $(ls data/ev_[bl][0-9]*_c.npy 2>/dev/null | wc -l) -lt $N ]; do sleep 60; done
i=0; for c in $(ls -tr data/ev_[bl][0-9]*_c.npy | head -n $N); do y=${c%_c.npy}_y.npy; [ -f $y ] || continue; ln -f $c data/ev_${T}${i}_c.npy; ln -f $y data/ev_${T}${i}_y.npy; i=$((i+1)); done
tags=$(seq -s, -f "${T}%g" 0 $((i-1)))
say "$T: $i shards frozen (about $(( i * 15 ))M positions), $P passes from scratch"
arm() { n=$1; shift; t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $P --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz "$@" --out zero/w_$n.bin > zero/w_$n.train.log 2>&1
  say "arm $n ($*): played held-out $(grep -h '^done' zero/w_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18) in $(( $(date +%s) - t0 ))s"; }
arm ${T}P; arm ${T}K --head 16,32 --hk 64
for j in $(seq 0 $((i-1))); do rm -f data/ev_${T}${j}_c.npy data/ev_${T}${j}_y.npy; done
say "$T finished"
