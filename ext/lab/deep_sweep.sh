#!/bin/bash
# Search options at 100,000 positions per move against the standing search (same network, BIGM): what a 10,000-node test
# likes is not what a deep search likes (the history batch: +19 at 10,000 nodes, -9 at 100,000, -18 at the clock).
cd /root/loopchess
while pgrep -f "[d]epth_test.sh" > /dev/null; do sleep 10; done
S="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for x in LmrDiv=180 CutLmr=1 LmrPrune=1 CutLmr=2 LmrDiv=180,CutLmr=1 PruneV2=2 NmpBase=4 LmpDepth=12 ProbCut=150 PruneV2=1 NegExt=1 TtBucket=1; do
  bash gt.sh "BIGM [$x] against BIGM with the standing search, 100,000 positions per move" deep_sweep.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$S,$x" --opp "standing search|calc/calc_s19i zero/w_BIGM.bin|$S" --openings 500 --nodes 100000
done
echo "[$(date -u +%H:%M:%S)] finished" >> deep_sweep.log
