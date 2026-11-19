#!/bin/bash
# q_fill.sh <games> <openings to skip>     more test positions (ext/quality.py collect) and Stockfish's analysis of them, done
# in the gaps between tests: a batch of 440 positions at a time, giving way whenever zero/Q_BUSY appears.
cd /root/loopchess; G=$1; SK=$2
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; M="calc/calc_s19d zero/w_BIGM.bin"
lock() { while true; do touch zero/FILL_BUSY; [ -e zero/Q_BUSY ] || return; rm -f zero/FILL_BUSY; sleep 5; done; }
touch zero/FILL_ON zero/GEN_PAUSE; sleep 3
lock; python3 ext/quality.py collect --us "$M" --us_opts "$B" --nodes 30000 --games $G --per 8 --workers 22 --skip $SK 2>&1 | tail -1 >> fill.log; rm -f zero/FILL_BUSY
while true; do
  lock; out=$(python3 ext/quality.py truth --nodes 4000000 --workers 22 --limit 440 2>&1 | tail -1); rm -f zero/FILL_BUSY
  echo "[$(date -u +%H:%M:%S)] $out" >> fill.log
  case "$out" in *"nothing left"*) break;; esac
done
rm -f zero/FILL_ON; [ -e zero/Q_BUSY ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> fill.log
