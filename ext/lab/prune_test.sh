#!/bin/bash
# History flags without the harmful one, then the second batch (option PruneV2, bit flags) on top; same network, equal nodes.
cd /root/loopchess
S="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; B="$S,HistV2=15"
g() { bash gt.sh "$1" hist_test.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$2" --opp "$3|calc/calc_s19i zero/w_BIGM.bin|$4" --openings 2000 --nodes 10000; }
g "BIGM with HistV2=15 against BIGM with the standing search, 10,000 positions per move" "$S,HistV2=15" "standing search" "$S"
g "BIGM with HistV2=7 against BIGM with the standing search, 10,000 positions per move" "$S,HistV2=7" "standing search" "$S"
for x in 15 1 2 4 8 16; do
  g "BIGM with HistV2=15 and PruneV2=$x against BIGM with HistV2=15, 10,000 positions per move" "$B,PruneV2=$x" "history flags 15 only" "$B"
done
bash gt.sh "BIGM with HistV2=15 against BIGM with the standing search, 10+0.1" hist_test.log 76 --us "calc/calc_s19i zero/w_BIGM.bin" --us_opts "$B" --opp "standing search|calc/calc_s19i zero/w_BIGM.bin|$S" --tc 10+0.1 --openings 500
echo "[$(date -u +%H:%M:%S)] second batch finished" >> hist_test.log
