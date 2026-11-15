#!/bin/bash
# One plain 384 network from scratch on all the Lichess positions held (deep evaluations + every converted month of game
# positions), then against the best so far (LFf) at equal positions per move.
cd /root/loopchess; EP=${1:-4}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> big.log; }
while ! grep -q "h finished" fishnet.log 2>/dev/null; do sleep 30; done
while pgrep -f "train_nnue.py.*w_LF[W]" > /dev/null; do sleep 20; done
kh=$(ls data/ev_h[0-9]*_c.npy | wc -l); tags=$(seq -s, -f "e%g" 0 11),$(seq -s, -f "f%g" 0 39),$(seq -s, -f "h%g" 0 $((kh - 1)))
n=$(python3 -c "
import numpy as np
print(round(sum(len(np.load(f'data/ev_{t}_y.npy', mmap_mode='r')) for t in '$tags'.split(',')) / 1e9, 2))")
say "$n billion positions in $(echo $tags | tr ',' '\n' | wc -l) shards; training a plain 384 network from scratch, $EP passes"
t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $tags --val bval --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_BIG.bin > zero/w_BIG.train.log 2>&1
say "network BIG trained in $(( $(date +%s) - t0 ))s"
touch zero/FARM_PAUSE; sleep 5
bash gt.sh "BIG ($n billion Lichess positions) against LFf (0.79 billion), 30,000 positions per move" big.log 22 --us "calc/calc_s17 zero/w_BIG.bin" --opp "LFf|calc/calc_s17 zero/w_LFf.bin|Hash=64" --openings 1000 --nodes 30000
rm -f zero/FARM_PAUSE
say finished
