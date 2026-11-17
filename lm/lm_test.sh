#!/bin/bash
# lm_test.sh <checkpoint name> <log> "<nodes> <prior> <cap>"...     a transformer as root prior on the search (BIGM, standing
# options), against the search alone at the same node budget, on the quality test. The transformer runs on the graphics card.
cd /root/loopchess; CK=$1; LOG=$2; shift 2; export CALC_BIN=calc/calc_s19h OURO_SERVER=/tmp/ouro_$CK.sock
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
(OURO_GPU=1 python3 lm/serve.py ckpt/$CK.pt $OURO_SERVER 2 > lm_serve_$CK.log 2>&1 &) ; for i in $(seq 1 60); do grep -q ready lm_serve_$CK.log 2>/dev/null && break; sleep 1; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
echo "[$(date -u +%H:%M:%S)] transformer $CK as root prior, against the search alone" >> $LOG
for t in "$@"; do set -- $t; name=duo${CK}_${2}_${3}_$1
  python3 ext/quality.py test --name $name --cmd "python3 duo_uci.py zero/w_BIGM.bin ckpt/$CK.pt" --opts "$B,Prior=$2,PriorCap=$3" --nodes $1 --limit 8964 --workers 22 > /dev/null 2>&1
  echo "$1 nodes, prior $2 per unit, cap $3: $(python3 ext/quality.py cmp $name bigm_$1 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> $LOG
done
pkill -f "lm/serve.py ckpt/$CK.pt" ; rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished $CK" >> $LOG
