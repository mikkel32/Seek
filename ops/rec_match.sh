#!/bin/bash
# the standing network against Stockfish 14.1 at the clock, every game kept (moves and time per move)
cd /root/loopchess; rm -f games_bigs.jsonl; touch zero/Q_BUSY
GAMES_LOG=/root/loopchess/games_bigs.jsonl bash gt.sh "BIGS against Stockfish 14.1, 10+0.1, 22 boards, games recorded" clock.log 22 --us "calc/calc_s19c zero/w_BIGS.bin" --us_opts "Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --tc 10+0.1
rm -f zero/Q_BUSY
