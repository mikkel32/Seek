#!/bin/bash
# After ext/fishnet.sh: a network on the game positions alone (FN) and one on them plus the deep evaluation database (LF),
# both from scratch; each against the network trained on the deep database alone (LE) at equal positions per move.
cd /root/loopchess; P=${1:-f}; EP=${2:-6}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> fishnet.log; }
while ! grep -q "$P finished" fishnet.log 2>/dev/null; do sleep 30; done
k=$(ls data/ev_${P}[0-9]*_c.npy | wc -l); ftags=$(seq -s, -f "${P}%g" 0 $((k - 1))); etags=$(seq -s, -f "e%g" 0 11)
arm() { n=$1; files=$2; t0=$(date +%s); python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files $files --val bval --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_$n.bin > zero/w_$n.train.log 2>&1; say "network $n trained from scratch, $EP passes, in $(( $(date +%s) - t0 ))s"; }
arm FN$P $ftags; arm LF$P $etags,$ftags
touch zero/FARM_PAUSE; sleep 5
for n in FN$P LF$P; do bash gt.sh "$n against LE (deep evaluation database alone), 30,000 positions per move" ext.log 22 --us "calc/calc_s17 zero/w_$n.bin" --opp "LE|calc/calc_s17 zero/w_LE.bin|Hash=64" --openings 1000 --nodes 30000; done
rm -f zero/FARM_PAUSE
say "$P matches finished"
