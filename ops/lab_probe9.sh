#!/bin/bash
# Third round on the match machine's card: probes with a named opponent, so a finished network can be fine-tuned with a
# teacher and played against itself-before. Same data, batch and matches as lab_probe2.sh; the learning rate comes with the options.
#   bash lab_probe9.sh "<name>|<trainer options (must include --lr)>|<shards, e.g. h:0-33,s:0-33>|<opponent network file>|<opponent name>" ...
cd /root/loopchess
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for spec in "$@"; do
  IFS='|' read n o g opp oname <<< "$spec"; t0=$(date +%s)
  tg=""; for r in ${g//,/ }; do pre=${r%%:*}; rg=${r#*:}; tg="$tg,$(seq -s, -f "$pre%g" ${rg%-*} ${rg#*-})"; done; tg=${tg#,}
  python3 calc/train_nnue_distill2.py --h 768 --buckets 8 --kb 16 --kf 1 --ema 0.999 --files $tg --val ival --ytag yc --epochs 1 --bs 32768 --threads 6 --device cuda --pst zero/g101s.npz --out zero/p_$n.bin $o > zero/p_$n.train.log 2>&1
  r=$(grep "^done" zero/p_$n.train.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  echo "[$(date -u +%H:%M:%S)] $n ($o; $g): ${r:-FAILED: $(tail -1 zero/p_$n.train.log | cut -c1-120)}; $(( $(date +%s) - t0 ))s, $(grep '^step' zero/p_$n.train.log | tail -1 | grep -o '[0-9]*k pos/s')" >> lab_probe.log
  [ -z "$r" ] && continue
  echo "[$(date -u +%H:%M:%S)]   engine against trainer: $(python3 zero/check_net.py calc/calc_s19 zero/p_$n.bin zero/p_$n.bin.ema.pt 1000 2>&1 | tail -1 | cut -c1-200)" >> lab_probe.log
  flock /dev/shm/cpu.lock bash gt.sh "  $n against $oname, 30,000 positions per move" lab_probe.log 76 --us "calc/calc_s19 zero/p_$n.bin" --us_opts "$B" --opp "$oname|calc/calc_s19 $opp|$B" --openings 1000 --nodes 30000
  flock /dev/shm/cpu.lock bash gt.sh "  $n against Stockfish 14.1, 10,000 positions per move each" lab_probe.log 76 --us "calc/calc_s19 zero/p_$n.bin" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 1000 --nodes 10000
done
echo "[$(date -u +%H:%M:%S)] third round of probes finished" >> lab_probe.log
