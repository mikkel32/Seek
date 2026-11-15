#!/bin/bash
# the layered read-out in games (and, while it trains, clean clock results for the width and fresh-data arms)
cd /root/loopchess
tags=$(seq -s, -f "b%g" 0 35)
say() { echo "[$(date -u +%H:%M:%S)] $*" >> head.log; }
g() { bash gt.sh "$1" head.log 76 --us "calc/calc_s13 $2" --opp "$3|calc/calc_s13 $4|Hash=64" "${@:5}"; }
( t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs 2 --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --widen zero/g101.pt --head 16,32 --out zero/w_N384H.bin > zero/w_N384H.train.log 2>&1
  say "N384H (N384's recipe + layered read-out 16-32) trained in $(( $(date +%s) - t0 ))s: $(grep -h '^done' zero/w_N384H.train.log | grep -o 'averaged ([0-9.]*' | cut -c11-18) on played held-out positions (N384: 0.518579)" ) &
g "clean rerun: W768 against N384 at the clock, 10+0.1" zero/w_W768.bin N384 zero/w_N384.bin --tc 10+0.1 --openings 300 --stop 0,20
g "fresh data alone at the clock: N384 against generation 101, 10+0.1" zero/w_N384.bin "generation 101" zero/vol_net.bin --tc 10+0.1 --openings 300 --stop 0,30
g "fresh data alone at the clock: N384 against generation 101, 60+0.6" zero/w_N384.bin "generation 101" zero/vol_net.bin --tc 60+0.6 --openings 150 --stop 0,30
wait
say "engine against trainer, N384H: $(python3 zero/check_net.py calc/calc_s13 zero/w_N384H.bin zero/w_N384H.bin.ema.pt 2>&1 | tail -1 | cut -c1-260)"
say "speed, N384H: $(python3 zero/bench.py calc/calc_s13 zero/w_N384H.bin 14 | cut -c1-90)"; say "speed, N384: $(python3 zero/bench.py calc/calc_s13 zero/w_N384.bin 14 | cut -c1-90)"
g "layered read-out: N384H against N384, 30,000 positions per move" zero/w_N384H.bin N384 zero/w_N384.bin --openings 1000 --nodes 30000 --stop 0,20
g "layered read-out: N384H against N384 at the clock, 10+0.1" zero/w_N384H.bin N384 zero/w_N384.bin --tc 10+0.1 --openings 300 --stop 0,20
g "layered read-out: N384H against N384 at the clock, 60+0.6" zero/w_N384H.bin N384 zero/w_N384.bin --tc 60+0.6 --openings 150 --stop 0,20
say finished
