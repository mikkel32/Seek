#!/bin/bash
# The diagnostic: this engine's search with Stockfish 14.1's own network (build -DSFNET, ext/sfnet_extract.py,
# ext/sfnet_check.py) against Stockfish 14.1 at equal nodes: the evaluator is the same on both sides, so the result is the
# difference between the two searches. Then the same build against this engine with its own network: the difference
# between the two evaluators inside one search. Three settings of EvalScale, since our search's margins are in our units.
#   bash ext/sfnet_matches.sh <workers> <nodes> <openings>
cd /root/loopchess; W=$1; ND=$2; OP=$3; export SFNET_FILE=/root/ext/sf141_embedded.bin
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
g() { bash gt.sh "$1" sfnet.log $W --us "calc/calc_s19sf zero/w_BIGM.bin" --us_opts "$O,EvalScale=$2" --opp "$3" --openings $OP --nodes $ND; }
for es in 100 70 140; do
  g "our search with Stockfish 14.1's network [EvalScale=$es] against Stockfish 14.1, $ND nodes each" $es "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"
done
g "our search with Stockfish 14.1's network [EvalScale=100] against our search with BIGM, $ND nodes each" 100 "BIGM|calc/calc_s19c zero/w_BIGM.bin|$O"
echo "[$(date -u +%H:%M:%S)] finished" >> sfnet.log
