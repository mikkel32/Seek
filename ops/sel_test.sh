#!/bin/bash
# sel_test.sh <log> <model file> "<SelA SelB>"...     the learned judgment of late moves against the standing rule, 300,000 nodes
cd /root/loopchess; LOG=$1; MODEL=$2; shift 2
while [ -e zero/MATCH_BUSY ]; do sleep 5; done; touch zero/MATCH_BUSY zero/GEN_PAUSE; sleep 2
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; M="calc/calc_s19g zero/w_BIGM.bin"; N=${NODES:-300000}; T=$(basename $MODEL .txt)
echo "[$(date -u +%H:%M:%S)] $MODEL at $N positions per move" >> $LOG
for ab in "$@"; do set -- $ab; n=${T}_${1}_${2}_$N
  python3 ext/quality.py test --name $n --cmd "$M" --opts "$B,SelFile=$MODEL,Sel=1,SelA=$1,SelB=$2" --nodes $N --limit 8964 > /dev/null 2>&1
  r=$(python3 ext/quality.py cmp $n bigm_$N 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')
  echo "SelA=$1 SelB=$2: $r" >> $LOG
done
rm -f zero/MATCH_BUSY zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> $LOG
