#!/bin/bash
# eqtime_test.sh <checkpoint name> <log> "<LmDepth> <PolW> <PolLmr> <LmWait> <LmAhead>"...
# The transformer steering the search, judged at EQUAL TIME: every configuration gets MS milliseconds a move (default 200)
# on the quality test's positions, so whatever the transformer costs in time is charged. Compared with the search alone at
# the same time (run first, once). WORKERS engines at once (default 3: the card then serves them about as it would serve
# one game; at 22 the answers come too late to judge anything), LIMIT positions (default 8964).
# Trainers on the card are paused for the test (stopped and continued).
cd /root/loopchess; CK=$1; LOG=$2; shift 2; export OURO_SOCK=/tmp/ouro_eq_$CK.sock; BIN=${LMBIN:-calc/calc_s19M}
MS=${MS:-200}; W=${WORKERS:-3}; LIM=${LIMIT:-8964}
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
# the trainer processes only: stopping their tmux shells would make tmux continue the whole process group
TR=$(pgrep -f "^python3 (lm/train_lm|calc/train_nnue)" | tr "\n" " "); for p in $TR; do kill -STOP $p; done
trap 'for p in $TR; do kill -CONT $p; done; rm -f zero/Q_BUSY' EXIT
(OURO_GPU=1 python3 lm/serve.py ckpt/$CK.pt $OURO_SOCK 2 > lm_serve_eq_$CK.log 2>&1 &) ; for i in $(seq 1 90); do grep -q ready lm_serve_eq_$CK.log 2>/dev/null && break; sleep 1; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; base=eq_plain_${MS}_w${W}_$LIM
echo "[$(date -u +%H:%M:%S)] transformer $CK steering the search at equal time: $MS ms a move, $W engines at once, $LIM positions" >> $LOG
[ -e data/q/runs/$base.json ] || python3 ext/quality.py test --name $base --cmd "$BIN zero/w_BIGM.bin" --opts "$B" --nodes -$MS --limit $LIM --workers $W > /dev/null 2>&1
for t in "$@"; do set -- $t; name=eq${CK}_$1_$2_$3_w$4_a$5_${MS}_w${W}_$LIM
  python3 ext/quality.py test --name $name --cmd "$BIN zero/w_BIGM.bin" --opts "$B,LmDepth=$1,PolW=$2,PolLmr=$3,LmWait=$4,LmAhead=$5" --nodes -$MS --limit $LIM --workers $W > /dev/null 2>&1
  echo "LmDepth $1, PolW $2, PolLmr $3, LmWait $4, LmAhead $5: $(python3 ext/quality.py cmp $name $base 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//'); server: $(tail -1 lm_serve_eq_$CK.log | cut -d: -f4- | cut -c1-90)" >> $LOG
done
pkill -f "lm/serve.py ckpt/$CK.pt $OURO_SOCK"; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished $CK" >> $LOG
