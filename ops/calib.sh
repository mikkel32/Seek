#!/bin/bash
# calib.sh: the quality test's controls with the current engine source, then games that put its units on the Elo scale
cd /root/loopchess; touch zero/GEN_PAUSE; sleep 2
ts() { date -u +%H:%M:%S; }
gcc -O3 -march=native -funroll-loops -DHMAX=1024 -o calc/calc_s19d calc/calc_s19.c -lm || { echo "[$(ts)] build failed" >> calib.log; rm -f zero/GEN_PAUSE; exit 1; }
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"; M="calc/calc_s19d zero/w_BIGM.bin"
q() { python3 ext/quality.py "$@" 2>&1 | tail -1 >> calib.log; }
echo "[$(ts)] controls" >> calib.log
q test --name d_30000 --cmd "$M" --opts "$B" --nodes 30000
q cmp d_30000 bigm_30000
for n in 30000 300000; do q test --name hist7_$n --cmd "$M" --opts "$B,HistV2=7" --nodes $n; q cmp hist7_$n bigm_$n; done
q test --name hist1_300000 --cmd "$M" --opts "$B,HistV2=1" --nodes 300000; q cmp hist1_300000 bigm_300000
echo "[$(ts)] games" >> calib.log
SF="Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1"
g() { label=$1; shift; r=$(python3 gauntlet.py --us "$M" --us_opts "$B" --opp "$SF" --workers 22 "$@" | tail -1 | cut -c1-330); echo "[$(ts)] $label: $r" >> calib.log; }
g "30k v 30k"   --nodes 30000  --openings 500
g "100k v 30k"  --nodes 100000 --opp_nodes 30000  --openings 500
g "100k v 100k" --nodes 100000 --openings 500
g "300k v 100k" --nodes 300000 --opp_nodes 100000 --openings 400
rm -f zero/GEN_PAUSE; echo "[$(ts)] finished" >> calib.log
