#!/bin/bash
# Sibling-ranking probes: the standing network (BIGS) is trained on for a short while on its own recipe (labels and the
# two teachers), with or without the sibling loss of calc/train_nnue.py (--sib), and then judged: held-out loss, held-out
# sibling pairs, the static move-choice test (ext/brain_metric.py, as it stands and with captures played out) and, unless
# NOPLAY=1, 2,000 games against BIGS at 30,000 positions per move.
#   bash ext/sib_probe.sh "<name>|<extra trainer options>" ...        G="h:0-5,s:0-5" picks the files, LR the learning rate
cd /root/loopchess
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; E=calc/calc_s19h; G=${G:-"h:0-5,s:0-5"}
T="--distill zero/w_BIGX.bin.ema.pt+sf:/root/ext/sf141_embedded.bin --dw 0.5"
tg=""; for r in ${G//,/ }; do pre=${r%%:*}; rg=${r#*:}; tg="$tg,$(seq -s, -f "$pre%g" ${rg%-*} ${rg#*-})"; done; tg=${tg#,}
say() { echo "[$(date -u +%H:%M:%S)] $*" >> sib_probe.log; }
for spec in "$@"; do
  IFS='|' read n o <<< "$spec"; t0=$(date +%s)
  python3 ${TRAINER:-calc/train_nnue_v6.py} --h 768 --buckets 8 --kb 16 --kf 1 --ema 0.999 --files $tg --val ival --ytag yc --epochs 1 --bs 32768 --lr ${LR:-1.5e-4} --threads 4 --device cuda --pst zero/g101s.npz \
    --resume ${FROM:-zero/w_BIGS.bin.ema.pt} --out zero/q_$n.bin $T $o > zero/q_$n.train.log 2>&1
  r=$(grep "^done" zero/q_$n.train.log | sed 's/.*averaged (\([0-9.]*\), \([0-9.]*\)).*/held-out loss \1, mean error \2/')
  say "$n ($o): ${r:-FAILED: $(tail -1 zero/q_$n.train.log | cut -c1-160)}; $(( $(date +%s) - t0 ))s, $(grep '^step' zero/q_$n.train.log | tail -1 | grep -o '[0-9]*k pos/s')"
  [ -z "$r" ] && continue
  grep -q "^end held-out" zero/q_$n.train.log && say "  $(grep '^start held-out' zero/q_$n.train.log | cut -c1-200)" && say "  $(grep '^end held-out' zero/q_$n.train.log | cut -c1-200)"
  say "  engine against trainer: $(python3 zero/check_net.py $E zero/q_$n.bin zero/q_$n.bin.ema.pt 1000 2>&1 | tail -1 | cut -c1-170)"
  say "  static move choice:       $(python3 ext/brain_metric.py "$n|$E zero/q_$n.bin" 2>&1 | tail -1)"
  say "  with captures played out: $(QEVAL=1 python3 ext/brain_metric.py "$n|$E zero/q_$n.bin" 2>&1 | tail -1)"
  [ -n "$NOPLAY" ] && continue
  while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
  bash gt.sh "  $n against BIGS, 30,000 positions per move" sib_probe.log 22 --us "$E zero/q_$n.bin" --us_opts "$B" --opp "BIGS|$E zero/w_BIGS.bin|$B" --openings 1000 --nodes 30000
  rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE
done
say "sibling probes finished"
