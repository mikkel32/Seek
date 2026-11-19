#!/bin/bash
# The network trained on Stockfish-labelled Lichess positions (LE): at the clock against the self-taught network, and a
# fresh standing against Stockfish 14.1 with and without its network. The farm is paused meanwhile.
cd /root/loopchess; touch zero/FARM_PAUSE; sleep 8
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
g() { bash gt.sh "$1" le.log 76 --us "calc/calc_s17 zero/w_LE.bin" "${@:2}"; }
g "LE against the self-taught network (N384) at the clock, 10+0.1" --opp "N384 self-taught|calc/calc_s17 zero/w_N384.bin|Hash=64" --tc 10+0.1 --openings 200
g "standing, LE with the confirmed options: against Stockfish 14.1 on its hand-written evaluator, 10+0.1" --us_opts "$B" --opp "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false" --tc 10+0.1 --openings 100
g "standing, LE with the confirmed options: against Stockfish 14.1, 10+0.1" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings 100
g "LE against the self-taught network (N384) at the clock, 60+0.6" --opp "N384 self-taught|calc/calc_s17 zero/w_N384.bin|Hash=64" --tc 60+0.6 --openings 100
g "standing, LE with the confirmed options: against Stockfish 14.1 on its hand-written evaluator, 60+0.6" --us_opts "$B" --opp "Stockfish 14.1 hand-written evaluator|/usr/games/stockfish|Hash=64,Threads=1,Use NNUE=false" --tc 60+0.6 --openings 75
rm -f zero/FARM_PAUSE
echo "[$(date -u +%H:%M:%S)] finished" >> le.log
