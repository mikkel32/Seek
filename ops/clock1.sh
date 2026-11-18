#!/bin/bash
# the standing network and BIGS against Stockfish 14.1 at the clock on this machine (22 boards), 1,000 games each
cd /root/loopchess
while [ -e zero/MATCH_BUSY ]; do sleep 5; done; touch zero/MATCH_BUSY zero/GEN_PAUSE; sleep 3
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; SF="Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"
for n in BIGS BIGM; do
  bash gt.sh "$n against Stockfish 14.1, 10+0.1, 22 boards" clock.log 22 --us "calc/calc_s19c zero/w_$n.bin" --us_opts "$O" --opp "$SF" --openings 500 --tc 10+0.1
done
rm -f zero/MATCH_BUSY; echo "[$(date -u +%H:%M:%S)] finished" >> clock.log
