#!/bin/bash
# the per-unit integer scale against the old single worst-case scale, same network on both sides, 30,000 nodes a move
cd /root/loopchess
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
bash gt.sh "BIGS, per-unit integer scale against the single scale, 30,000 nodes" scale_match.log 22 --us "calc/calc_s19q zero/w_BIGS.bin" --us_opts "$O" --opp "BIGS old scale|env QBITS=0 calc/calc_s19q zero/w_BIGS.bin|$O" --openings 2000 --nodes 30000
bash gt.sh "BIGX, per-unit integer scale against the single scale, 30,000 nodes" scale_match.log 22 --us "calc/calc_s19qx zero/w_BIGX.bin" --us_opts "$O" --opp "BIGX old scale|env QBITS=0 calc/calc_s19qx zero/w_BIGX.bin|$O" --openings 1500 --nodes 30000
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> scale_match.log
