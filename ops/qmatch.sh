#!/bin/bash
# the same network with a finer integer scale in the engine against the standing scale, 30,000 nodes a move
cd /root/loopchess
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for q in 9 8; do
  bash gt.sh "BIGS with integer scale 2^$q against 2^7, 30,000 nodes" brain.log 22 --us "env QBITS=$q calc/calc_s19q zero/w_BIGS.bin" --us_opts "$O" --opp "BIGS 2^7|calc/calc_s19q zero/w_BIGS.bin|$O" --openings 2000 --nodes 30000
done
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> brain.log
