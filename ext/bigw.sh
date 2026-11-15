#!/bin/bash
# After ext/big.sh: the same 4.37 billion Lichess positions, a 768-wide plain network from scratch, 4 passes; then against
# the 384-wide one (BIG) at equal positions per move.
cd /root/loopchess; EP=${1:-4}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> big.log; }
while ! grep -q "network BIG trained" big.log 2>/dev/null; do sleep 30; done
kh=$(ls data/ev_h[0-9]*_c.npy | wc -l); tags=$(seq -s, -f "e%g" 0 11),$(seq -s, -f "f%g" 0 39),$(seq -s, -f "h%g" 0 $((kh - 1)))
say "training a 768-wide network from scratch on the same positions, $EP passes"
t0=$(date +%s); python3 calc/train_nnue.py --h 768 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_BIGW.bin > zero/w_BIGW.train.log 2>&1
say "network BIGW (768 wide) trained in $(( $(date +%s) - t0 ))s"
while ! grep -q "^\[.*\] finished" big.log; do sleep 20; done
touch zero/FARM_PAUSE; sleep 5
bash gt.sh "BIGW (768 wide, 4.37 billion positions) against BIG (384 wide, same positions), 30,000 positions per move" big.log 22 --us "calc/calc_s17 zero/w_BIGW.bin" --opp "BIG|calc/calc_s17 zero/w_BIG.bin|Hash=64" --openings 1000 --nodes 30000
rm -f zero/FARM_PAUSE
say "wide finished"
