#!/bin/bash
# Does the 768-wide network use the extra data? BIGW (4.37 billion positions) against LFW (0.79 billion), both 768 wide, equal positions per move.
cd /root/loopchess
touch zero/FARM_PAUSE; sleep 5
bash gt.sh "BIGW (768 wide, 4.37 billion positions) against LFW (768 wide, 0.79 billion), 30,000 positions per move" big.log 22 --us "calc/calc_s17 zero/w_BIGW.bin" --opp "LFW|calc/calc_s17 zero/w_LFW.bin|Hash=64" --openings 1000 --nodes 30000
rm -f zero/FARM_PAUSE
echo "[$(date -u +%H:%M:%S)] data-768 finished" >> big.log

