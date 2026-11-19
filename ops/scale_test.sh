#!/bin/bash
# Do networks taught by Stockfish's network tolerate a larger evaluation scale (= tighter pruning margins), as their
# teacher does in our search (EvalScale 140 was its best)? Against Stockfish 14.1 at 10,000 positions per move each.
cd /root/loopchess
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; SF="Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"
for spec in "p_distill_sf_half calc/calc_s19i 120" "p_distill_sf_half calc/calc_s19i 140" "p_distill_sf calc/calc_s19i 120" "p_distill_sf calc/calc_s19i 140" "w_BIGX calc/calc_s19x 120" "w_BIGX calc/calc_s19x 85" "p_mixc calc/calc_s19i 120"; do
  set -- $spec
  flock /dev/shm/cpu.lock bash gt.sh "$1 with EvalScale=$3 against Stockfish 14.1, 10,000 positions per move each" scale_test.log 76 --us "$2 zero/$1.bin" --us_opts "$B,EvalScale=$3" --opp "$SF" --openings 1000 --nodes 10000
done
echo "[$(date -u +%H:%M:%S)] finished" >> scale_test.log
