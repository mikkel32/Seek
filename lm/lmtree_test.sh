#!/bin/bash
# lmtree_test.sh <checkpoint name> <log> "<nodes> <LmDepth> <PolW> <PolLmr>"...     the transformer's move probabilities
# order the quiet moves (PolW) and shift their reductions (PolLmr) at every node with LmDepth or more plies left
# (calc_s19.c, OURO_SOCK); quality test against the search alone (BIGM, standing options) at the same node budget.
# The large transformer's training, if running, is paused for the test (stopped and continued, nothing is lost).
cd /root/loopchess; CK=$1; LOG=$2; shift 2; export OURO_SOCK=/tmp/ouro_tree_$CK.sock
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
TR=$(pgrep -f "lm/train_l[m].py"); [ -n "$TR" ] && kill -STOP $TR
trap '[ -n "$TR" ] && kill -CONT $TR' EXIT
(OURO_GPU=1 python3 lm/serve.py ckpt/$CK.pt $OURO_SOCK 2 > lm_serve_tree_$CK.log 2>&1 &) ; for i in $(seq 1 60); do grep -q ready lm_serve_tree_$CK.log 2>/dev/null && break; sleep 1; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
echo "[$(date -u +%H:%M:%S)] transformer $CK steering the quiet moves at the top of the tree, against the search alone" >> $LOG
for t in "$@"; do set -- $t; name=lmt${CK}_$2_$3_$4_$1; t0=$(date +%s)
  python3 ext/quality.py test --name $name --cmd "calc/calc_s19L zero/w_BIGM.bin" --opts "$B,LmDepth=$2,PolW=$3,PolLmr=$4" --nodes $1 --limit 8964 --workers 22 > /dev/null 2>&1
  echo "$1 nodes, LmDepth $2, PolW $3, PolLmr $4 ($(( $(date +%s) - t0 ))s): $(python3 ext/quality.py cmp $name bigm_$1 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> $LOG
done
pkill -f "lm/serve.py ckpt/$CK.pt $OURO_SOCK" ; rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished $CK" >> $LOG
