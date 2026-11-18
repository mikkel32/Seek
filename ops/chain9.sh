#!/bin/bash
cd /root/loopchess
while ! grep -q finished calib.log; do sleep 10; done
# our search with Stockfish's own network: the evaluator no longer differs from the judge's, what is left is the search
touch zero/Q_BUSY zero/GEN_PAUSE; export SFNET_FILE=/root/ext/sf141_embedded.bin
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
q() { python3 ext/quality.py "$@" 2>&1 | tail -1 >> sfnet_q.log; }
for n in 30000 100000 300000 1000000; do q test --name sfnet_$n --cmd "calc/calc_s19sf zero/w_BIGM.bin" --opts "$B,EvalScale=140" --nodes $n --limit 8964; q cmp sfnet_$n sf_$n; q cmp sfnet_$n bigm_$n; done
q cmp sf_30000 sfnet_100000; q cmp sf_100000 sfnet_300000; q cmp sf_300000 sfnet_1000000
unset SFNET_FILE; rm -f zero/Q_BUSY; echo "[$(date -u +%H:%M:%S)] finished" >> sfnet_q.log
# the reduction and null-move amounts, judged where the search is deep
bash ext/q_screen.sh calc/calc_s19f 300000 screen300k.log CutLmr=1 CutLmr=2 LmrBase=108 LmrBase=148 LmrDiv=200 LmrDiv=170 NmpBase=4 NmpDiv=3 "NmpBase=4,NmpDiv=3" "CutLmr=1,LmrBase=108" "CutLmr=2,LmrBase=108,NmpBase=4" SearchV3=48
# the new switches and the pruning-depth ones, also at 300,000 (at 100,000 this test resolves little more than games do)
bash ext/q_screen.sh calc/calc_s19f 300000 screen300k.log SearchV3=16 SearchV3=64 SearchV3=1 SearchV3=2 SearchV3=4 SearchV3=8 LmrPrune=1 LmpDepth=8 LmpDepth=63 NegExt=1 ProbCut=150 TtCapLmr=1 NmpMargin=80 PruneV2=8 HistV2=1
bash ext/q_fill.sh 5600 800
