#!/bin/bash
# Search options re-checked under the new evaluator family, with enough games to see 15 Elo: the mixed-data probe against
# Stockfish 14.1 at 10,000 positions per move each, 4,000 games per setting, on the match machine's processors.
#   bash lab_sweep.sh <network stem> "<options>" ...
cd /root/loopchess; N=$1; shift
while pgrep -f "[m]ixc_clock.sh" > /dev/null; do sleep 10; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for x in "$@"; do
  bash gt.sh "$N [${x:-as it is}] against Stockfish 14.1, 10,000 positions per move each" lab_sweep.log 70 --us "calc/calc_s19 zero/$N.bin" --us_opts "$B${x:+,$x}" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 2000 --nodes 10000
done
echo "[$(date -u +%H:%M:%S)] sweep finished" >> lab_sweep.log
