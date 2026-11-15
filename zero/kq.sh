#!/bin/bash
# The cheap layered read-out (layers read only the first 64 or 128 units of each side) from scratch on the same 270M
# positions as S384 and S384H, then against plain S384 at equal positions and at the clock.
cd /root/loopchess
say() { echo "[$(date -u +%H:%M:%S)] $*" >> kq.log; }
tags=$(seq -s, -f "k%g" 0 17)
arm() { n=$1; shift; t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs 8 --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz "$@" --out zero/w_$n.bin > zero/w_$n.train.log 2>&1
  say "arm $n ($*): played held-out $(grep -h '^done' zero/w_$n.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18) in $(( $(date +%s) - t0 ))s"; }
arm S384K --head 16,32 --hk 64 & arm S384K2 --head 16,32 --hk 128 & wait
for n in S384K S384K2; do say "engine against trainer, $n: $(python3 zero/check_net.py calc/calc_s16 zero/w_$n.bin zero/w_$n.bin.ema.pt 2>&1 | tail -1 | cut -c1-230)"; done
while ! grep -q finished s15.log 2>/dev/null; do sleep 20; done
touch zero/FARM_PAUSE; sleep 12
for n in S384 S384K S384K2 S384H; do say "speed, $n: $(python3 zero/bench.py calc/calc_s16 zero/w_$n.bin 14 | cut -c1-90)"; done
g() { bash gt.sh "$1" kq.log 76 --us "calc/calc_s16 zero/w_$2.bin" --opp "$3|calc/calc_s16 zero/w_$4.bin|Hash=64" "${@:5}"; }
g "layers reading 64 units a side (S384K) against plain (S384), 30,000 positions per move" S384K S384 S384 --openings 1000 --nodes 30000
g "layers reading 128 units a side (S384K2) against plain (S384), 30,000 positions per move" S384K2 S384 S384 --openings 1000 --nodes 30000
g "layers reading 64 units a side (S384K) against plain (S384) at the clock, 10+0.1" S384K S384 S384 --tc 10+0.1 --openings 400
g "layers reading 128 units a side (S384K2) against plain (S384) at the clock, 10+0.1" S384K2 S384 S384 --tc 10+0.1 --openings 400
g "layers reading 64 units a side (S384K) against plain (S384) at the clock, 60+0.6" S384K S384 S384 --tc 60+0.6 --openings 150
rm -f zero/FARM_PAUSE
for i in $(seq 0 17); do rm -f data/ev_k${i}_c.npy data/ev_k${i}_y.npy; done
say finished
