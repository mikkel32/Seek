#!/bin/bash
# Architecture probes on the current recipe: 768 wide, 16 king buckets with a shared table, half labels and half the two
# teachers (BIGX and Stockfish 14.1's network), one pass over 1.09 billion mixed positions; then against the reference
# probe at 30,000 positions per move (2,000 games) and against Stockfish 14.1 at 10,000 each (1,000 games).
#   bash ext/vault_probe.sh <reference name or -> "<name>|<extra trainer options>" ...
# (later options override earlier ones, e.g. "--buckets 16 --obk 1"). NOPLAY=1: training and held-out loss only.
# With "-" no match against a reference is played (use it to train the reference itself). Matches take the test lock.
cd /root/loopchess; REF=$1; shift
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; E=calc/calc_s19h; G="h:0-33,s:0-33"
T="--distill zero/w_BIGX.bin.ema.pt+sf:/root/ext/sf141_embedded.bin --dw 0.5"
tg=""; for r in ${G//,/ }; do pre=${r%%:*}; rg=${r#*:}; tg="$tg,$(seq -s, -f "$pre%g" ${rg%-*} ${rg#*-})"; done; tg=${tg#,}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> vault_probe.log; }
for spec in "$@"; do
  IFS='|' read n o <<< "$spec"; t0=$(date +%s)
  python3 ${TRAINER:-calc/train_nnue_v4.py} --h 768 --buckets 8 --kb 16 --kf 1 --ema 0.999 --files $tg --val ival --ytag yc --epochs 1 --bs 32768 --lr 1.4e-3 --threads 8 --device cuda --pst zero/g101s.npz --out zero/q_$n.bin $T $o > zero/q_$n.train.log 2>&1
  r=$(grep "^done" zero/q_$n.train.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  say "$n ($o): ${r:-FAILED: $(tail -1 zero/q_$n.train.log | cut -c1-160)}; $(( $(date +%s) - t0 ))s, $(grep '^step' zero/q_$n.train.log | tail -1 | grep -o '[0-9]*k pos/s')"
  [ -z "$r" ] && continue
  [ -n "$NOPLAY" ] && continue
  say "  engine against trainer: $(python3 zero/check_net.py $E zero/q_$n.bin zero/q_$n.bin.ema.pt 1000 2>&1 | tail -1 | cut -c1-170)"
  say "  speed: $(python3 zero/bench.py $E zero/q_$n.bin 13 | cut -c1-80)"
  while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
  [ "$REF" != "-" ] && bash gt.sh "  $n against $REF, 30,000 positions per move" vault_probe.log 22 --us "$E zero/q_$n.bin" --us_opts "$B" --opp "$REF|$E zero/q_$REF.bin|$B" --openings 1000 --nodes 30000
  bash gt.sh "  $n against Stockfish 14.1, 10,000 positions per move each" vault_probe.log 22 --us "$E zero/q_$n.bin" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --nodes 10000
  rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE
done
say "probes finished"
