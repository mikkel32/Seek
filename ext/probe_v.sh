#!/bin/bash
# Architecture probes judged by games, on the training machine: every variant from scratch, one pass over the same 1.02
# billion Lichess positions (shards h0-h59), then 2,000 games at 30,000 positions per move against the reference variant
# trained the same way. Held-out loss, engine-against-trainer check and search speed are logged for each. probe_v.log.
#   bash ext/probe_v.sh <reference name> "<name>|<trainer options>[|<engine binary>[|<shard tags>]]" ...
# Shard tags default to h0-h59 (Lichess game positions); "s:0-35,s:37-68" style ranges name other shards of the same total.
# Self-play on this machine is paused for the duration (it shares the graphics card and the processors).
cd /root/loopchess; REF=$1; shift; tags=$(seq -s, -f "h%g" 0 59)
say() { echo "[$(date -u +%H:%M:%S)] $*" >> probe_v.log; }
touch zero/FARM_PAUSE
for spec in "$@"; do
  IFS='|' read n o e g <<< "$spec"; e=${e:-calc/calc_s18}; t0=$(date +%s); tg=$tags
  if [ -n "$g" ]; then tg=""; for r in ${g//,/ }; do pre=${r%%:*}; rg=${r#*:}; tg="$tg,$(seq -s, -f "$pre%g" ${rg%-*} ${rg#*-})"; done; tg=${tg#,}; fi
  python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tg --val ival --epochs 1 --threads 4 --device cuda --pst zero/g101s.npz --out zero/p_$n.bin $o > zero/p_$n.train.log 2>&1
  r=$(grep "^done" zero/p_$n.train.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  say "$n ($o): ${r:-FAILED: $(tail -1 zero/p_$n.train.log | cut -c1-120)}; $(( $(date +%s) - t0 ))s, $(grep '^step' zero/p_$n.train.log | tail -1 | grep -o '[0-9]*k pos/s')"
  [ -z "$r" ] && continue
  say "  engine against trainer: $(python3 zero/check_net.py $e zero/p_$n.bin zero/p_$n.bin.ema.pt 1000 2>&1 | tail -1 | cut -c1-140)"
  say "  speed ($e): $(python3 zero/bench.py $e zero/p_$n.bin 14 | cut -c1-90)"
  [ "$n" = "$REF" ] || bash gt.sh "  $n against $REF, 30,000 positions per move" probe_v.log 22 --us "$e zero/p_$n.bin" --opp "$REF|$e zero/p_$REF.bin|Hash=64" --openings 1000 --nodes 30000
done
rm -f zero/FARM_PAUSE; say "probes finished"
