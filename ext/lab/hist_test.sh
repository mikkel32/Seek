#!/bin/bash
# The history batch (option HistV2, bit flags) against the standing search, same network (BIGM), 10,000 positions per move.
cd /root/loopchess
while pgrep -f "[a]fter_clock.sh" > /dev/null; do sleep 10; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for x in 31 1 2 4 8 16; do
  bash gt.sh "BIGM with HistV2=$x against BIGM with the standing search, 10,000 positions per move" hist_test.log 76 --us "calc/calc_s19h zero/w_BIGM.bin" --us_opts "$B,HistV2=$x" --opp "standing search|calc/calc_s19h zero/w_BIGM.bin|$B" --openings 2000 --nodes 10000
done
echo "[$(date -u +%H:%M:%S)] finished" >> hist_test.log
