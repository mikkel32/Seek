#!/bin/bash
# Does the gain of the history batch at 10,000 nodes hold with more nodes? The same pair at 30,000 and 100,000 nodes, then per flag.
cd /root/loopchess
S="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
g() { bash gt.sh "BIGM with HistV2=$1 against BIGM with the standing search, $2 positions per move" hist_test.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$S,HistV2=$1" --opp "standing search|calc/calc_s19i zero/w_BIGM.bin|$S" --openings $3 --nodes $2; }
g 7 30000 1000
g 7 100000 500
g 1 100000 500
g 4 100000 500
g 2 100000 500
echo "[$(date -u +%H:%M:%S)] depth test finished" >> hist_test.log
