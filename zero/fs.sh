#!/bin/bash
# Does the layered read-out pay when the network is trained with it from the start? Two networks from scratch on the same
# 270M positions (the 18 newest farm shards, frozen under their own names), 8 passes each, then games at equal positions.
cd /root/loopchess
say() { echo "[$(date -u +%H:%M:%S)] $*" >> fs.log; }
for i in $(seq 0 17); do for k in c y; do ln -f data/ev_b$((i+18))_$k.npy data/ev_f${i}_$k.npy; done; done
tags=$(seq -s, -f "f%g" 0 17)
arm() { n=$1; shift; t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs 8 --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz "$@" --out zero/w_$n.bin > zero/w_$n.train.log 2>&1
  say "arm $n ($*): played held-out $(grep -h '^done' zero/w_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18) in $(( $(date +%s) - t0 ))s"; }
arm S384 & arm S384H --head 16,32 & wait
touch zero/FARM_PAUSE; sleep 12
g() { bash gt.sh "$1" fs.log 76 --us "calc/calc_s13 zero/w_$2.bin" --opp "$3|calc/calc_s13 zero/w_$4.bin|Hash=64" "${@:5}"; }
g "from scratch: with the layers (S384H) against without (S384), 30,000 positions per move" S384H S384 S384 --openings 1000 --nodes 30000
g "from scratch on 270M (S384) against the generational network (N384), 30,000 positions per move" S384 N384 N384 --openings 500 --nodes 30000
rm -f zero/FARM_PAUSE
for i in $(seq 0 17); do rm -f data/ev_f${i}_c.npy data/ev_f${i}_y.npy; done
say finished
