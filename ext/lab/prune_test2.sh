#!/bin/bash
cd /root/loopchess
S="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; B="$S,HistV2=7"
bash gt.sh "BIGM with HistV2=7 against BIGM with the standing search, 10+0.1" hist_test.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$B" --opp "standing search|calc/calc_s19i zero/w_BIGM.bin|$S" --tc 10+0.1 --openings 500
g() { bash gt.sh "$1" hist_test.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$2" --opp "$3|calc/calc_s19i zero/w_BIGM.bin|$4" --openings 2000 --nodes 10000; }
for x in 4 8 16; do
  g "BIGM with HistV2=7 and PruneV2=$x against BIGM with HistV2=7, 10,000 positions per move" "$B,PruneV2=$x" "history flags 7 only" "$B"
done
bash gt.sh "BIGM with HistV2=7 against BIGM with the standing search, 60+0.6" hist_test.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$B" --opp "standing search|calc/calc_s19i zero/w_BIGM.bin|$S" --tc 60+0.6 --openings 150
echo "[$(date -u +%H:%M:%S)] finished" >> hist_test.log
