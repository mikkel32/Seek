#!/bin/bash
# when BIGQ's own matches are done: BIGQ at the clock against Stockfish 14.1, the same conditions as BIGS's -110
cd /root/loopchess
while ! grep -q "BIGQ finished" big_run.log; do sleep 20; done
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY
GAMES_LOG=/root/loopchess/games_bigq.jsonl bash gt.sh "BIGQ against Stockfish 14.1, 10+0.1, 22 boards" clock.log 22 --us "calc/calc_s19h zero/w_BIGQ.bin" --us_opts "Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --tc 10+0.1
rm -f zero/Q_BUSY; echo "[$(date -u +%H:%M:%S)] finished BIGQ clock" >> clock.log
