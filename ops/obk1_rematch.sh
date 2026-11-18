#!/bin/bash
# the queen-aware 16-head probe again, with an engine that reads its bucket key (the first match was played by one that did not)
cd /root/loopchess; B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; E=calc/calc_s19h
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
bash gt.sh "  obk1 against r0, 30,000 positions per move (replayed with bucket support)" vault_probe.log 22 --us "$E zero/q_obk1.bin" --us_opts "$B" --opp "r0|$E zero/q_r0.bin|$B" --openings 1000 --nodes 30000
bash gt.sh "  obk1 against Stockfish 14.1, 10,000 positions per move each (replayed)" vault_probe.log 22 --us "$E zero/q_obk1.bin" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings 500 --nodes 10000
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE
