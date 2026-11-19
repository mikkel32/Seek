#!/bin/bash
# the two-brain player on today's engine: the old 2M-parameter transformer's policy as a prior on the root moves of the
# search (BIGM, standing options), against the search alone at the same node budget, on the quality test
cd /root/loopchess; export CALC_BIN=calc/calc_s19h
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
echo "[$(date -u +%H:%M:%S)] two-brain player (old transformer as root prior), against the search alone" >> lm_duo.log
for n in 30000 300000; do for pc in "8 40" "16 80" "30 150"; do set -- $pc; name=duo_${1}_${2}_$n
  python3 ext/quality.py test --name $name --cmd "python3 duo_uci.py zero/w_BIGM.bin ckpt/B.pt" --opts "$B,Prior=$1,PriorCap=$2" --nodes $n --limit 8964 --workers 16 > /dev/null 2>&1
  echo "$n nodes, prior $1 per unit, cap $2: $(python3 ext/quality.py cmp $name bigm_$n 2>&1 | tail -1 | sed -e 's/^.*difference //' -e 's/; negative = the first gives away less//')" >> lm_duo.log
done; done
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> lm_duo.log
