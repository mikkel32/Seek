#!/bin/bash
# lmtree_test.sh <checkpoint name> <log> "<nodes> <LmDepth> <PolW> <PolLmr> [LmWait=1] [workers=22] [LmAhead=0]"...
# The transformer's policy orders the quiet moves (PolW) and shifts their reductions (PolLmr) at every node with LmDepth
# or more plies left (calc_s19.c, OURO_SOCK). LmWait=1: the engine waits for each answer; 0: it searches on and uses the
# answer when the node comes round again; LmAhead=k: on entering a deeper node it asks about the first k children
# without waiting. Quality test against the search alone (BIGM, standing options) at the same
# node budget. The large transformer's training, if running, is paused for the test (stopped and continued).
cd /root/loopchess; CK=$1; LOG=$2; shift 2; export OURO_SOCK=/tmp/ouro_tree_$CK.sock; BIN=${LMBIN:-calc/calc_s19M}
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
# the trainer process only: stopping its tmux shell would make tmux continue the whole process group
TR=$(pgrep -f "^python3 lm/train_l[m].py" | tr "\n" " "); for p in $TR; do kill -STOP $p; done
trap 'for p in $TR; do kill -CONT $p; done' EXIT
(OURO_GPU=1 python3 lm/serve.py ckpt/$CK.pt $OURO_SOCK 2 > lm_serve_tree_$CK.log 2>&1 &) ; for i in $(seq 1 60); do grep -q ready lm_serve_tree_$CK.log 2>/dev/null && break; sleep 1; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
echo "[$(date -u +%H:%M:%S)] transformer $CK steering the quiet moves at the top of the tree ($BIN), against the search alone; training paused: $(for p in $TR; do ps -o stat= -p $p; done | tr '\n' ' ')" >> $LOG
for t in "$@"; do set -- $t; w=${5:-1}; ah=${7:-0}; name=lmt${CK}_$2_$3_$4_w${w}_a${ah}_n${6:-22}_$1; t0=$(date +%s)
  python3 ext/quality.py test --name $name --cmd "$BIN zero/w_BIGM.bin" --opts "$B,LmDepth=$2,PolW=$3,PolLmr=$4,LmWait=$w,LmAhead=$ah" --nodes $1 --limit 8964 --workers ${6:-22} > /dev/null 2>&1
  echo "$1 nodes, LmDepth $2, PolW $3, PolLmr $4, LmWait $w, LmAhead $ah, ${6:-22} at once ($(( $(date +%s) - t0 ))s): $(python3 ext/quality.py cmp $name bigm_$1 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> $LOG
done
pkill -f "lm/serve.py ckpt/$CK.pt $OURO_SOCK" ; rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished $CK" >> $LOG
