#!/bin/bash
# duo_clock.sh <checkpoint name> <prior> <cap> <openings>     the two-brain player (BIGS + transformer on the card as root
# prior) against Stockfish 14.1 at 10+0.1, 22 boards; the transformer's thinking is charged to our clock
cd /root/loopchess; CK=$1; export CALC_BIN=calc/calc_s19c OURO_SERVER=/tmp/ouro_clock_$CK.sock
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
(OURO_GPU=1 python3 lm/serve.py ckpt/$CK.pt $OURO_SERVER 2 > lm_serve_clock_$CK.log 2>&1 &) ; for i in $(seq 1 60); do grep -q ready lm_serve_clock_$CK.log 2>/dev/null && break; sleep 1; done
O="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
bash gt.sh "BIGS + transformer $CK as root prior ($2 per unit, cap $3) against Stockfish 14.1, 10+0.1, 22 boards" clock.log 22 --us "python3 duo_uci.py zero/w_BIGS.bin ckpt/$CK.pt" --us_opts "$O,Prior=$2,PriorCap=$3" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --openings $4 --tc 10+0.1
pkill -f "lm/serve.py ckpt/$CK.pt"; rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished duo clock $CK" >> clock.log
