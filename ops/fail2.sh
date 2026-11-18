#!/bin/bash
# the same diagnosis with Stockfish's network in our search: what is left is the search alone
cd /root/loopchess; export SFNET_FILE=/root/ext/sf141_embedded.bin
gcc -O3 -march=native -funroll-loops -DHMAX=1024 -DSFNET -o calc/calc_s19sf3 calc/calc_s19.c -lm || exit 1
while [ -e zero/Q_BUSY ]; do sleep 5; done; touch zero/Q_BUSY zero/GEN_PAUSE; while [ -e zero/FILL_BUSY ]; do sleep 2; done
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1,EvalScale=140"
python3 ext/q_fail.py --sf sf_100000 --ours sfnet_1000000 --cmd "calc/calc_s19sf3 zero/w_BIGM.bin" --base "$B" --nodes 1000000 \
  "standing|" "standing, 3 million|nodes=3000000" \
  "no null move|Off=1" "no reverse futility|Off=2" "no late-move reductions|Off=8" "no late-move pruning|Off=16" "no quiet futility|Off=32" \
  "no losing-exchange pruning|Off=64" "no bad-history pruning|Off=128" "no singular tests|Off=256" "no depth cut without table move|Off=512" \
  "no move pruning at all|Off=240" "check extension|ChkExt=1" "half the reductions|LmrDiv=480,LmrBase=34" "another hash size|Hash=32" > fail2.log 2>&1
rm -f zero/Q_BUSY; [ -e zero/FILL_ON ] || rm -f zero/GEN_PAUSE; echo "[$(date -u +%H:%M:%S)] finished" >> fail2.log
