#!/bin/bash
# The same 4.37 billion Lichess positions, 768 wide, with king-bucketed inputs: 16 mirrored buckets by the view's own king
# square, each bucket's table = a shared table + its own correction. From scratch, 4 passes, batches of 32,768 (the bucketed
# step is cheapest per position there). Then against the plain 768-wide network (BIGW) at equal positions per move.
# Self-play on this machine is paused for the duration (it shares the graphics card).
cd /root/loopchess; EP=${1:-4}; N=${2:-BIGK}; KB=${3:-16}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> bigk.log; }
kh=$(ls data/ev_h[0-9]*_c.npy | wc -l); tags=$(seq -s, -f "e%g" 0 11),$(seq -s, -f "f%g" 0 39),$(seq -s, -f "h%g" 0 $((kh - 1)))
touch zero/FARM_PAUSE
say "training $N: 768 wide, $KB king buckets with a shared table, from scratch, $EP passes over the 4.37 billion positions"
t0=$(date +%s); python3 calc/train_nnue.py --h 768 --buckets 8 --kb $KB --kf 1 --ema 0.999 --files $tags --val bval --epochs $EP --bs 32768 --lr 1.4e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_$N.bin > zero/w_$N.train.log 2>&1
say "network $N trained in $(( $(date +%s) - t0 ))s: $(tail -1 zero/w_$N.train.log | cut -c1-150)"
say "engine against trainer: $(python3 zero/check_net.py calc/calc_s18 zero/w_$N.bin zero/w_$N.bin.ema.pt 1000 | cut -c1-140)"
sleep 5
for n in BIGW $N; do say "speed, $n: $(python3 zero/bench.py calc/calc_s18 zero/w_$n.bin 14 | cut -c1-90)"; done
bash gt.sh "$N (768 wide, $KB king buckets) against BIGW (768 wide, plain), same 4.37 billion positions, 30,000 positions per move" bigk.log 22 --us "calc/calc_s18 zero/w_$N.bin" --opp "BIGW|calc/calc_s18 zero/w_BIGW.bin|Hash=64" --openings 1000 --nodes 30000
rm -f zero/FARM_PAUSE
say "$N finished"
