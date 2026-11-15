#!/bin/bash
# Mirrored inputs in games: (1) one more pass from N384 with and without them, (2) from scratch against the plain from-scratch network.
cd /root/loopchess
say() { echo "[$(date -u +%H:%M:%S)] $*" >> mir.log; }
for i in $(seq 0 17); do for k in c y; do ln -f data/ev_b$((i+18))_$k.npy data/ev_m${i}_$k.npy; done; done
tags=$(seq -s, -f "m%g" 0 17)
arm() { n=$1; shift; python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz "$@" --out zero/w_$n.bin > zero/w_$n.train.log 2>&1
  say "arm $n ($*): played held-out $(grep -h '^done' zero/w_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18)"; }
arm M1 --epochs 1 --widen zero/w_N384.bin.pt --hm 1 & arm C1 --epochs 1 --widen zero/w_N384.bin.pt & wait
say "engine against trainer, M1: $(python3 zero/check_net.py calc/calc_s14 zero/w_M1.bin zero/w_M1.bin.ema.pt 300 mirror 2>&1 | tail -1 | cut -c1-200)"
arm S384M --epochs 8 --hm 1
while ! grep -q finished fs.log 2>/dev/null; do sleep 20; done
touch zero/FARM_PAUSE; sleep 12
g() { bash gt.sh "$1" mir.log 76 --us "calc/calc_s14 zero/w_$2.bin" --opp "$3|calc/calc_s14 zero/w_$4.bin|Hash=64" "${@:5}"; }
g "mirrored inputs, one more pass from N384 (M1) against the same pass without (C1), 30,000 positions per move" M1 C1 C1 --openings 1000 --nodes 30000
g "mirrored inputs from scratch (S384M) against plain from scratch (S384), 30,000 positions per move" S384M S384 S384 --openings 1000 --nodes 30000
rm -f zero/FARM_PAUSE
for i in $(seq 0 17); do rm -f data/ev_m${i}_c.npy data/ev_m${i}_y.npy; done
say finished
