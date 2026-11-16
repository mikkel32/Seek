#!/bin/bash
# One full-scale king-bucketed network from scratch on a named set of shards, then: a match against an earlier network at
# 30,000 positions per move, and matches against Stockfish 14.1 at 10,000 and 30,000 positions per move each.
# US_EXTRA=<options> adds engine options for the new network only (e.g. EvalScale=70 for labels of a larger scale).
#   bash ext/big_run.sh <name> <passes> <king buckets> <width> <shards, e.g. "s:0-147" or "e:0-11,f:0-39,h:0-209,s:0-147"> <opponent network> <engine> [extra trainer options]
# Self-play on this machine is paused for the duration (it shares the graphics card).
cd /root/loopchess; N=$1; EP=$2; KB=$3; W=$4; G=$5; OPP=$6; E=$7; shift 7
say() { echo "[$(date -u +%H:%M:%S)] $*" >> big_run.log; }
tg=""; for r in ${G//,/ }; do pre=${r%%:*}; rg=${r#*:}; tg="$tg,$(seq -s, -f "$pre%g" ${rg%-*} ${rg#*-})"; done; tg=${tg#,}
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; OU="$O${US_EXTRA:+,$US_EXTRA}"
touch zero/FARM_PAUSE; ulimit -n 65536     # one memory map per shard file: more than a thousand of them
say "options of the new network in its matches: $OU"
say "training $N: $W wide, $KB king buckets with a shared table, from scratch, $EP passes over $G $*"
t0=$(date +%s); python3 calc/train_nnue.py --h $W --buckets 8 --kb $KB --kf 1 --ema 0.999 --files $tg --val ival --epochs $EP --bs 32768 --lr 1.4e-3 --threads 4 --device cuda --pst zero/g101s.npz --out zero/w_$N.bin "$@" > zero/w_$N.train.log 2>&1
say "network $N trained in $(( $(date +%s) - t0 ))s: $(tail -1 zero/w_$N.train.log | cut -c1-150)"
say "engine against trainer: $(python3 zero/check_net.py $E zero/w_$N.bin zero/w_$N.bin.ema.pt 1000 | cut -c1-140)"
sleep 5
for n in $OPP $N; do say "speed, $n: $(python3 zero/bench.py $E zero/w_$n.bin 14 | cut -c1-90)"; done
bash gt.sh "$N against $OPP, 30,000 positions per move" big_run.log 22 --us "$E zero/w_$N.bin" --us_opts "$OU" --opp "$OPP|$E zero/w_$OPP.bin|$O" --openings 1000 --nodes 30000
for nd in 10000 30000; do
  bash gt.sh "$N against Stockfish 14.1, $nd positions per move each" big_run.log 22 --us "$E zero/w_$N.bin" --us_opts "$OU" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --nodes $nd
done
rm -f zero/FARM_PAUSE
say "$N finished"
