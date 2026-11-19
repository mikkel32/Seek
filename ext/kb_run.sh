#!/bin/bash
# One king-bucketed network from scratch on the Lichess positions, then a match at equal positions per move.
#   bash ext/kb_run.sh <name> <passes> <king buckets> <width> <4|7: billions of positions> <opponent network> [extra trainer options]
# 4 = the 4.37 billion (e, f, h shards); 7 = those plus 2.68 billion more (i shards; i139 is the held-out set and never trained on).
# Self-play on this machine is paused for the duration (it shares the graphics card).
cd /root/loopchess; N=$1; EP=$2; KB=$3; W=$4; D=$5; OPP=$6; shift 6
say() { echo "[$(date -u +%H:%M:%S)] $*" >> bigk.log; }
tags=$(seq -s, -f "e%g" 0 11),$(seq -s, -f "f%g" 0 39),$(seq -s, -f "h%g" 0 209); [ "$D" = 7 ] && tags=$tags,$(seq -s, -f "i%g" 0 138)
touch zero/FARM_PAUSE
say "training $N: $W wide, $KB king buckets with a shared table, from scratch, $EP passes over the $D billion positions $*"
t0=$(date +%s); python3 calc/train_nnue.py --h $W --buckets 8 --kb $KB --kf 1 --ema 0.999 --files $tags --val ival --epochs $EP --bs 32768 --lr 1.4e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_$N.bin "$@" > zero/w_$N.train.log 2>&1
say "network $N trained in $(( $(date +%s) - t0 ))s: $(tail -1 zero/w_$N.train.log | cut -c1-150)"
say "engine against trainer: $(python3 zero/check_net.py calc/calc_s18 zero/w_$N.bin zero/w_$N.bin.ema.pt 1000 | cut -c1-140)"
sleep 5
for n in $OPP $N; do say "speed, $n: $(python3 zero/bench.py calc/calc_s18 zero/w_$n.bin 14 | cut -c1-90)"; done
bash gt.sh "$N ($W wide, $KB king buckets, $D billion positions) against $OPP, 30,000 positions per move" bigk.log 22 --us "calc/calc_s18 zero/w_$N.bin" --opp "$OPP|calc/calc_s18 zero/w_$OPP.bin|Hash=64" --openings 1000 --nodes 30000
rm -f zero/FARM_PAUSE
say "$N finished"
