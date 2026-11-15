#!/bin/bash
# 768 wide against 384 wide, both trained on the same outside data (LFW, LFf): at the clock. The farm is paused meanwhile.
cd /root/loopchess; touch zero/FARM_PAUSE; sleep 8
for n in LFf LFW; do echo "[$(date -u +%H:%M:%S)] speed, $n: $(python3 zero/bench.py calc/calc_s17 zero/w_$n.bin 14 | cut -c1-90)" >> lw.log; done
g() { bash gt.sh "$1" lw.log 76 --us "calc/calc_s17 zero/w_LFW.bin" --opp "LFf (384 wide)|calc/calc_s17 zero/w_LFf.bin|Hash=64" "${@:2}"; }
g "768 wide (LFW) against 384 wide (LFf) at the clock, 10+0.1" --tc 10+0.1 --openings 200
g "768 wide (LFW) against 384 wide (LFf) at the clock, 60+0.6" --tc 60+0.6 --openings 100
rm -f zero/FARM_PAUSE
echo "[$(date -u +%H:%M:%S)] finished" >> lw.log
