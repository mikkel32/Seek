#!/bin/bash
cd /root/loopchess
gcc -O3 -march=native -funroll-loops -DHMAX=1024 -o calc/calc_s19h calc/calc_s19.c -lm || exit 1
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
python3 ext/q_fail.py --sf sf_100000 --ours bigm_1000000 --cmd "calc/calc_s19h zero/w_BIGM.bin" --base "$B" --nodes 1000000 \
  "standing|" "standing, 3 million|nodes=3000000" "standing, 10 million|nodes=10000000" \
  "no null move|Off=1" "no reverse futility|Off=2" "no razoring|Off=4" "no late-move reductions|Off=8" "no late-move pruning|Off=16" "no quiet futility|Off=32" \
  "no losing-exchange pruning|Off=64" "no bad-history pruning|Off=128" "no singular tests|Off=256" "no depth cut without table move|Off=512" "no delta pruning|Off=1024" \
  "no move pruning at all|Off=240" "no static pruning|Off=6" "check extension|ChkExt=1" "half the reductions|LmrDiv=480,LmrBase=34" "quiet checks kept|SearchV3=1" > fail1.log 2>&1
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> fail1.log
