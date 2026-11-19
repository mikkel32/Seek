#!/bin/bash
# Search settings against Stockfish 14.1 at equal positions per move: one match per setting, same openings, same network.
#   bash ext/sweep.sh <network> <engine> <workers> <positions per move> <openings> "<extra options>" ...
cd /root/loopchess; N=$1; E=$2; W=$3; NODES=$4; OP=$5; shift 5
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
for x in "$@"; do
  bash gt.sh "$N [${x:-as it is}] against Stockfish 14.1, $NODES positions per move each" sweep.log $W --us "$E zero/w_$N.bin" --us_opts "$O${x:+,$x}" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings $OP --nodes $NODES
done
echo "[$(date -u +%H:%M:%S)] sweep finished" >> sweep.log
