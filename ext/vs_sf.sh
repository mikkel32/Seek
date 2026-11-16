#!/bin/bash
# Networks against Stockfish 14.1 at equal positions per move (the measure the ladder showed to matter): one match each.
#   bash ext/vs_sf.sh <engine> <workers> <positions per move> <openings> <network file stem> ...
cd /root/loopchess; E=$1; W=$2; NODES=$3; OP=$4; shift 4
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for n in "$@"; do
  bash gt.sh "$n against Stockfish 14.1, $NODES positions per move each" vs_sf.log $W --us "$E zero/$n.bin" --us_opts "$O" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings $OP --nodes $NODES
done
