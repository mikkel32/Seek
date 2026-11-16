#!/bin/bash
# Where is the gap to Stockfish 14.1: in the evaluator or in the search? The same pair at equal positions per move, from
# almost no search (300) upward. A gap that is already there at 300 is the evaluator's; one that grows is the search's.
#   bash ext/node_ladder.sh <network> <engine> <workers>
cd /root/loopchess; N=$1; E=$2; W=$3
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for spec in "300 1000" "3000 1000" "30000 500" "300000 250"; do set -- $spec
  for opp in "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false"; do
    bash gt.sh "$N against ${opp%%|*}, $1 positions per move each" ladder.log $W --us "$E zero/w_$N.bin" --us_opts "$O" --opp "$opp" --openings $2 --nodes $1
  done
done
echo "[$(date -u +%H:%M:%S)] ladder finished" >> ladder.log
