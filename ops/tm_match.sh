#!/bin/bash
# tm_match.sh <TmCurve value> ...     BIGS with the move-number time curve against Stockfish 14.1 at the clock, games kept
cd /root/loopchess
for c in "$@"; do
  while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY; rm -f games_tm$c.jsonl
  GAMES_LOG=/root/loopchess/games_tm$c.jsonl bash gt.sh "BIGS with TmCurve=$c against Stockfish 14.1, 10+0.1, 22 boards" clock.log 22 --us "calc/calc_s19T zero/w_BIGS.bin" --us_opts "Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1,TmCurve=$c" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --tc 10+0.1
  python3 timefrac.py games_tm$c.jsonl > tm_profile_$c.log 2>&1
  rm -f zero/Q_BUSY
done
