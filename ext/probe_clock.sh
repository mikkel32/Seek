#!/bin/bash
# A network from the training machine at the clock on the match machine: standing against Stockfish 14.1 at 10+0.1,
# with the engine that has the EvalScale option. Waits for the tuned-values match at 10+0.1, skips its 60+0.6 repeat.
#   bash probe_clock.sh <network stem on the training machine, e.g. p_mixr> <EvalScale> [openings=200]
cd /root/loopchess; N=$1; ES=$2; OP=${3:-200}; . /root/.vault
[ -s zero/$N.bin ] || curl -s --fail -m 300 -H "X-Token: $T3" "$A3/f?p=/root/loopchess/zero/$N.bin" -o zero/$N.bin
while ! grep -q "10+0.1: {" spsa_test.log 2>/dev/null; do sleep 10; done
tmux kill-window -t lab:st 2>/dev/null; sleep 1; for p in $(pgrep -f "tuned search values"); do kill $p 2>/dev/null; done; sleep 5
B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
bash gt.sh "standing, $N with EvalScale=$ES (engine s19): against Stockfish 14.1, 10+0.1" probe_clock.log 76 --us "calc/calc_s19 zero/$N.bin" --us_opts "$B,EvalScale=$ES" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings $OP
bash gt.sh "standing, BIGK (engine s19) again: against Stockfish 14.1, 10+0.1" probe_clock.log 76 --us "calc/calc_s19 zero/w_BIGK.bin" --us_opts "$B" --opp "Stockfish 14.1|/usr/games/stockfish|Hash=64,Threads=1" --tc 10+0.1 --openings $OP
echo "[$(date -u +%H:%M:%S)] finished" >> probe_clock.log
