#!/bin/bash
# search suggestions at equal positions per move (after the mirrored-input matches; the farm is paused meanwhile)
cd /root/loopchess
while ! grep -q finished mir.log 2>/dev/null; do sleep 20; done
touch zero/FARM_PAUSE; sleep 12
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
t() { bash gt.sh "$1" s15.log 76 --us "calc/calc_s15 zero/w_N384.bin" --us_opts "$B,$2" --opp "pruning half|calc/calc_s15 zero/w_N384.bin|$B" --nodes 30000 --openings 1000; }
t "correction tables for minor and major pieces (MmCorr=128), 30,000 positions per move" "MmCorr=128"
t "correction tables for minor and major pieces (MmCorr=256), 30,000 positions per move" "MmCorr=256"
t "wider reverse-futility margin when a cheaper piece attacks ours (ThrRfp=80), 30,000 positions per move" "ThrRfp=80"
t "no reverse futility at all in that case (ThrRfp=2000), 30,000 positions per move" "ThrRfp=2000"
t "one ply less reduction in that case (ThrLmr=1), 30,000 positions per move" "ThrLmr=1"
rm -f zero/FARM_PAUSE
echo "[$(date -u +%H:%M:%S)] finished" >> s15.log
