#!/bin/bash
# The position set for ext/quality.py and its first measurements. Self-play generation is paused meanwhile (GEN_PAUSE).
cd /root/loopchess; touch zero/GEN_PAUSE; sleep 2
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; M="calc/calc_s19c zero/w_BIGM.bin"; X="calc/calc_s19x zero/w_BIGX.bin"
q() { python3 ext/quality.py "$@" 2>&1 | tail -1 >> quality.log; }
echo "[$(date -u +%H:%M:%S)] collecting" >> quality.log
q collect --us "$M" --us_opts "$B" --nodes 30000 --games 1600 --per 8 --workers 22
echo "[$(date -u +%H:%M:%S)] Stockfish's analysis, 4,000,000 nodes, two lines" >> quality.log
q truth --nodes 4000000 --workers 22
echo "[$(date -u +%H:%M:%S)] tests" >> quality.log
for n in 30000 100000 300000 1000000; do q test --name bigm_$n --cmd "$M" --opts "$B" --nodes $n; done
for n in 30000 100000 300000 1000000; do q test --name sf_$n --cmd /usr/games/stockfish --opts "Hash=64,Threads=1" --nodes $n; done
for n in 30000 300000; do q test --name bigx_$n --cmd "$X" --opts "$B" --nodes $n; done
for n in 30000 300000; do q test --name hist7_$n --cmd "$M" --opts "$B,HistV2=7" --nodes $n; done
q test --name scale140_100000 --cmd "$M" --opts "$B,EvalScale=140" --nodes 100000
for p in "bigx_30000 bigm_30000" "bigx_300000 bigm_300000" "hist7_30000 bigm_30000" "hist7_300000 bigm_300000" "scale140_100000 bigm_100000" "bigm_100000 bigm_30000" "bigm_300000 bigm_100000" "bigm_1000000 bigm_300000" "sf_30000 bigm_30000" "sf_100000 bigm_100000" "sf_300000 bigm_300000" "sf_1000000 bigm_1000000" "sf_30000 bigm_100000" "sf_100000 bigm_300000" "sf_300000 bigm_1000000"; do q cmp $p; done
rm -f zero/GEN_PAUSE
echo "[$(date -u +%H:%M:%S)] finished" >> quality.log
