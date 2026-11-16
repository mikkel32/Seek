#!/bin/bash
# How much does ten times the nodes buy each engine against itself? Same engine and evaluator on both sides, one side with
# ten times the node budget (gauntlet_odds.py). If our search gained much less than Stockfish's from the same step, the
# search would be the problem whatever the evaluator.
#   bash ext/scaling.sh <workers> <openings>
cd /root/loopchess; W=$1; OP=$2; O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
g() { r=$(OPP_NODES=$4 python3 gauntlet_odds.py --us "$2" --us_opts "$3" --opp "$5" --openings $OP --nodes $6 --workers $W | tail -1 | cut -c1-300); echo "[$(date -u +%H:%M:%S)] $1: $r" >> scaling.log; }
for pair in "3000 300" "30000 3000"; do set -- $pair
  g "BIGM at $1 nodes against BIGM at $2" "calc/calc_s19c zero/w_BIGM.bin" "$O" $2 "itself|calc/calc_s19c zero/w_BIGM.bin|$O" $1
  g "self-taught N384 at $1 nodes against itself at $2" "calc/calc_s19c zero/w_N384.bin" "$O" $2 "itself|calc/calc_s19c zero/w_N384.bin|$O" $1
  g "Stockfish 14.1 at $1 nodes against itself at $2" "/usr/games/stockfish" "Hash=64,Threads=1" $2 "itself|/usr/games/stockfish|Hash=64,Threads=1" $1
  g "Stockfish 14.1 hand-written evaluator at $1 nodes against itself at $2" "/usr/games/stockfish" "Hash=64,Threads=1,Use NNUE=false" $2 "itself|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false" $1
done
echo "[$(date -u +%H:%M:%S)] finished" >> scaling.log
