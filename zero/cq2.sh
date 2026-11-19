#!/bin/bash
# the two search candidates together, at the clock (after the cheap-layers matches; the farm is paused meanwhile)
cd /root/loopchess
while ! grep -q finished kq.log 2>/dev/null; do sleep 20; done
touch zero/FARM_PAUSE; sleep 15
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
t() { bash gt.sh "$1" s15.log 76 --us "calc/calc_s15 zero/w_N384.bin" --us_opts "$B,MmCorr=256,ThrRfp=2000" --opp "pruning half|calc/calc_s15 zero/w_N384.bin|$B" "${@:2}"; }
t "minor/major correction tables + no reverse futility under a cheap threat (MmCorr=256, ThrRfp=2000), 10+0.1" --tc 10+0.1 --openings 500
t "minor/major correction tables + no reverse futility under a cheap threat (MmCorr=256, ThrRfp=2000), 60+0.6" --tc 60+0.6 --openings 150
rm -f zero/FARM_PAUSE
echo "[$(date -u +%H:%M:%S)] clock tests finished" >> s15.log
